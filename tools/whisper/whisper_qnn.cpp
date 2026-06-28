// ============================================================================
// whisper_qnn.cpp — standalone QNN HTP runner for Qualcomm AI Hub's
//                   Whisper-tiny (separate encoder + decoder QNN context bins).
//
// Runs encoder once + the autoregressive greedy decode loop on the Snapdragon
// Hexagon NPU (HTP backend) via the QNN C API, and prints the decoded token IDs
// as one space-separated line. Detokenization to text is a later step.
//
// Usage (modes):
//   whisper_qnn <encoder.bin> <decoder.bin> <input_features.raw> [backend]
//       Legacy/positional mode. Prints the decoded token IDs (one line).
//
//   whisper_qnn --pcm <enc.bin> <dec.bin> <vocab.bin> <mel_filters.bin> <pcm.raw> [backend]
//       PCM -> mel front-end -> encoder -> decode -> detokenize -> print TEXT.
//
//   whisper_qnn --dump-mel <mel_filters.bin> <pcm.raw> <out.raw>
//       Compute the fp16 mel [1,80,3000] (480000 bytes) and exit. No NPU.
//       Validation hook: diff against the Python WhisperFeatureExtractor reference.
//
//   whisper_qnn --watch <enc.bin> <dec.bin> <vocab.bin> <mel_filters.bin> <channel_dir> [backend]
//       Resident daemon: load contexts + assets ONCE, then poll <channel_dir>
//       every 50 ms for the app<->daemon file channel and transcribe each request.
//
// The QNN API call sequence and the version-aware Qnn_Tensor_t accessors are
// mirrored from the canonical QAIRT 2.45.0 SampleApp source:
//   DynamicLoadUtil.cpp   — dlopen + QnnInterface_getProviders / QnnSystemInterface_getProviders
//   QnnSampleApp.cpp      — backendCreate/deviceCreate/logCreate, createFromBinary, graphRetrieve, graphExecute
//   QnnSampleAppUtils.cpp — copyMetadataToGraphsInfo (binaryInfo version switch)
//   QnnTypeMacros.hpp     — QNN_TENSOR_GET_*/SET_* accessors (inlined here as free helpers)
//   IOTensor.cpp          — client-buffer alloc + bind via clientBuf, output read via clientBuf.data
//
// Model I/O contract (whisper_tiny metadata.json):
//   encoder in : input_features [1,80,3000] fp16
//   encoder out: k_cache_cross_{0..3} [6,1,64,1500] fp16, v_cache_cross_{0..3} [6,1,1500,64] fp16
//   decoder in : input_ids [1,1] int32, attention_mask [1,1,1,200] fp16,
//                k/v_cache_self_{i}_in (fp16), k/v_cache_cross_{i} (fp16), position_ids [1] int32
//   decoder out: logits [1,51865,1,1] fp16, k/v_cache_self_{i}_out (fp16)
//
// All buffers are populated/read BY TENSOR NAME (not positional order).
// ============================================================================

#include <dlfcn.h>
#include <unistd.h>     // usleep (daemon poll), unlink
#include <sys/stat.h>   // stat (channel marker presence)

#include <cmath>        // cosf, sinf, log10, fmaxf
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "QnnInterface.h"
#include "QnnContext.h"
#include "QnnGraph.h"
#include "QnnTensor.h"
#include "QnnTypes.h"
#include "QnnLog.h"
#include "QnnBackend.h"
#include "QnnDevice.h"
#include "System/QnnSystemInterface.h"
#include "System/QnnSystemContext.h"

// fp16 type. clang on aarch64 supports _Float16 natively (ARMv8.2-A FP16).
using fp16 = _Float16;

// ----------------------------------------------------------------------------
// Whisper-tiny constants (DESIGN.md::decode_loop, verified against metadata.json)
// ----------------------------------------------------------------------------
static constexpr int   VOCAB_SIZE        = 51865;
static constexpr int   MEAN_DECODE_LEN   = 200;       // attention_mask length / decode budget
static constexpr float MASK_NEG          = -100.0f;

static constexpr int   SOT_TOKEN          = 50258;    // <|startoftranscript|>
static constexpr int   EOT_TOKEN          = 50257;    // <|endoftext|>
static constexpr int   LANG_EN_TOKEN      = 50259;    // <|en|>
static constexpr int   TASK_TRANSCRIBE    = 50359;    // <|transcribe|>
static constexpr int   NOTIMESTAMPS_TOKEN = 50363;    // <|notimestamps|>

// ----------------------------------------------------------------------------
// Mel front-end constants (OpenAI Whisper / transformers WhisperFeatureExtractor).
// PCM float32 16 kHz -> log-Mel input_features [1,80,3000] fp16.
// ----------------------------------------------------------------------------
[[maybe_unused]] static constexpr int MEL_SAMPLE_RATE = 16000;  // documents the contract
static constexpr int    MEL_N_FFT       = 400;      // window length
static constexpr int    MEL_HOP         = 160;      // 10 ms hop
static constexpr int    MEL_N_MELS      = 80;       // mel bins
static constexpr int    MEL_N_SAMPLES   = 480000;   // 30 s
static constexpr int    MEL_N_FRAMES    = 3000;     // 480000/160
static constexpr int    MEL_FFT_BINS    = MEL_N_FFT / 2 + 1;   // 201 one-sided bins
static constexpr int    MEL_PAD         = MEL_N_FFT / 2;       // 200 (torch center=True)
static constexpr size_t MEL_FEAT_ELEMS  = (size_t)MEL_N_MELS * MEL_N_FRAMES;   // 240000
static constexpr int    VOCAB_TOKENS    = 50257;   // entries in vocab.bin (byte-level BPE)

// ============================================================================
// Function-pointer signatures of the two entry symbols (DynamicLoadUtil.cpp:18-22)
// ============================================================================
typedef Qnn_ErrorHandle_t (*QnnInterfaceGetProvidersFn_t)(
    const QnnInterface_t*** providerList, uint32_t* numProviders);
typedef Qnn_ErrorHandle_t (*QnnSystemInterfaceGetProvidersFn_t)(
    const QnnSystemInterface_t*** providerList, uint32_t* numProviders);

// ============================================================================
// Inlined, version-aware Qnn_Tensor_t accessors.
//
// The public QNN headers ship NO accessor macros (those live in the sample's
// QnnTypeMacros.hpp). We reproduce only the ones we need. Per the SampleApp,
// the leading fields of Qnn_TensorV1_t and Qnn_TensorV2_t share an identical
// in-memory layout, so reading/writing them through `.v1.*` is correct for both
// versions (QnnTypeMacros.hpp does exactly this for id/name/type/dataType/rank/
// dimensions/memType/clientBuf). Only V2-only trailing fields would need the
// version branch, and we don't touch those.
// ============================================================================
static inline const char*       T_NAME(const Qnn_Tensor_t& t)      { return t.v1.name; }
static inline Qnn_DataType_t    T_DTYPE(const Qnn_Tensor_t& t)     { return t.v1.dataType; }
static inline uint32_t          T_RANK(const Qnn_Tensor_t& t)      { return t.v1.rank; }
static inline uint32_t*         T_DIMS(const Qnn_Tensor_t& t)      { return t.v1.dimensions; }
static inline void              T_SET_MEMTYPE(Qnn_Tensor_t& t, Qnn_TensorMemType_t m) { t.v1.memType = m; }
static inline void              T_SET_CLIENTBUF(Qnn_Tensor_t& t, const Qnn_ClientBuffer_t& b) { t.v1.clientBuf = b; }
static inline Qnn_ClientBuffer_t T_CLIENTBUF(const Qnn_Tensor_t& t) { return t.v1.clientBuf; }

// Bytes-per-element for the dtypes this model uses (DataUtil.cpp g_dataTypeToSize).
static size_t dtypeSizeBytes(Qnn_DataType_t dt) {
  switch (dt) {
    case QNN_DATATYPE_FLOAT_16: return 2;
    case QNN_DATATYPE_INT_32:   return 4;
    case QNN_DATATYPE_UINT_32:  return 4;
    case QNN_DATATYPE_FLOAT_32: return 4;
    case QNN_DATATYPE_INT_64:   return 8;
    case QNN_DATATYPE_UINT_64:  return 8;
    case QNN_DATATYPE_INT_8:    return 1;
    case QNN_DATATYPE_UINT_8:   return 1;
    case QNN_DATATYPE_INT_16:   return 2;
    case QNN_DATATYPE_UINT_16:  return 2;
    default:                    return 0;  // unsupported -> caller treats as error
  }
}

// Element count = product of dims (DataUtil.cpp calculateElementCount).
static size_t elementCount(const Qnn_Tensor_t& t) {
  size_t n = 1;
  uint32_t rank = T_RANK(t);
  uint32_t* dims = T_DIMS(t);
  for (uint32_t r = 0; r < rank; ++r) n *= (size_t)dims[r];
  return n;
}

// Byte length of a tensor's client buffer (DataUtil.cpp calculateLength).
static size_t tensorByteLength(const Qnn_Tensor_t& t) {
  return elementCount(t) * dtypeSizeBytes(T_DTYPE(t));
}

// ============================================================================
// Resolved QNN vtables (filled by loadBackend / loadSystem).
// ============================================================================
struct QnnVtables {
  QNN_INTERFACE_VER_TYPE        core{};    // == v2_34 for QAIRT 2.45.0
  QNN_SYSTEM_INTERFACE_VER_TYPE sys{};     // == v1_9
  void* backendLib = nullptr;
  void* systemLib  = nullptr;
};

#define CHECK_QNN(call, msg)                                                  \
  do {                                                                        \
    Qnn_ErrorHandle_t _e = (call);                                           \
    if (_e != QNN_SUCCESS) {                                                  \
      fprintf(stderr, "[ERROR] %s failed (0x%llx)\n", (msg),                  \
              (unsigned long long)_e);                                        \
      return false;                                                          \
    }                                                                         \
  } while (0)

// ----------------------------------------------------------------------------
// (a) dlopen backend .so + resolve the interface vtable (DynamicLoadUtil.cpp).
// ----------------------------------------------------------------------------
static bool loadBackend(const std::string& backendPath, QnnVtables& v) {
  v.backendLib = dlopen(backendPath.c_str(), RTLD_NOW | RTLD_GLOBAL);
  if (!v.backendLib) {
    fprintf(stderr, "[ERROR] dlopen(%s): %s\n", backendPath.c_str(), dlerror());
    return false;
  }
  auto getProviders = (QnnInterfaceGetProvidersFn_t)dlsym(
      v.backendLib, "QnnInterface_getProviders");
  if (!getProviders) {
    fprintf(stderr, "[ERROR] dlsym QnnInterface_getProviders: %s\n", dlerror());
    return false;
  }
  const QnnInterface_t** providers = nullptr;
  uint32_t num = 0;
  if (getProviders(&providers, &num) != QNN_SUCCESS || !providers || num == 0) {
    fprintf(stderr, "[ERROR] QnnInterface_getProviders returned no providers\n");
    return false;
  }
  bool found = false;
  for (uint32_t i = 0; i < num; ++i) {
    if (QNN_API_VERSION_MAJOR == providers[i]->apiVersion.coreApiVersion.major &&
        QNN_API_VERSION_MINOR <= providers[i]->apiVersion.coreApiVersion.minor) {
      v.core = providers[i]->QNN_INTERFACE_VER_NAME;  // the function table
      found = true;
      break;
    }
  }
  if (!found) {
    fprintf(stderr, "[ERROR] No compatible QNN interface provider (need core %d.%d)\n",
            QNN_API_VERSION_MAJOR, QNN_API_VERSION_MINOR);
    return false;
  }
  return true;
}

// ----------------------------------------------------------------------------
// (a') dlopen libQnnSystem.so + resolve the system (reflection) vtable.
// ----------------------------------------------------------------------------
static bool loadSystem(const std::string& systemPath, QnnVtables& v) {
  v.systemLib = dlopen(systemPath.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!v.systemLib) {
    fprintf(stderr, "[ERROR] dlopen(%s): %s\n", systemPath.c_str(), dlerror());
    return false;
  }
  auto getProviders = (QnnSystemInterfaceGetProvidersFn_t)dlsym(
      v.systemLib, "QnnSystemInterface_getProviders");
  if (!getProviders) {
    fprintf(stderr, "[ERROR] dlsym QnnSystemInterface_getProviders: %s\n", dlerror());
    return false;
  }
  const QnnSystemInterface_t** providers = nullptr;
  uint32_t num = 0;
  if (getProviders(&providers, &num) != QNN_SUCCESS || !providers || num == 0) {
    fprintf(stderr, "[ERROR] QnnSystemInterface_getProviders returned no providers\n");
    return false;
  }
  bool found = false;
  for (uint32_t i = 0; i < num; ++i) {
    if (QNN_SYSTEM_API_VERSION_MAJOR == providers[i]->systemApiVersion.major &&
        QNN_SYSTEM_API_VERSION_MINOR <= providers[i]->systemApiVersion.minor) {
      v.sys = providers[i]->QNN_SYSTEM_INTERFACE_VER_NAME;
      found = true;
      break;
    }
  }
  if (!found) {
    fprintf(stderr, "[ERROR] No compatible QNN system interface provider\n");
    return false;
  }
  return true;
}

// ============================================================================
// A loaded graph: its handle plus name->Qnn_Tensor_t maps for I/O, each tensor
// already bound to a heap-allocated client buffer of the right byte length.
// ============================================================================
struct LoadedGraph {
  Qnn_GraphHandle_t graph = nullptr;
  std::string name;

  // Contiguous arrays handed to graphExecute (input array is const-cast at call).
  std::vector<Qnn_Tensor_t> inputs;
  std::vector<Qnn_Tensor_t> outputs;

  // name -> index into inputs / outputs (so we can populate/read BY NAME).
  std::map<std::string, size_t> inIdx;
  std::map<std::string, size_t> outIdx;

  // Stable backing storage for tensor names/dims copied out of the metadata
  // (whose memory is freed by systemContextFree). std::deque is used because it
  // never invalidates references to existing elements on push_back, so the
  // pointers we hand to Qnn_Tensor_t.v1.name / .v1.dimensions stay valid.
  std::deque<std::string>             nameStore;
  std::deque<std::vector<uint32_t>>   dimStore;

  // Backing client buffers we own and must free.
  std::vector<void*> ownedBuffers;

  void* inBuf(const std::string& n) {
    auto it = inIdx.find(n);
    if (it == inIdx.end()) return nullptr;
    return T_CLIENTBUF(inputs[it->second]).data;
  }
  void* outBuf(const std::string& n) {
    auto it = outIdx.find(n);
    if (it == outIdx.end()) return nullptr;
    return T_CLIENTBUF(outputs[it->second]).data;
  }
  size_t inBytes(const std::string& n) {
    return tensorByteLength(inputs.at(inIdx.at(n)));
  }
  size_t outBytes(const std::string& n) {
    return tensorByteLength(outputs.at(outIdx.at(n)));
  }
};

// Allocate a client buffer for one tensor (copied as a template from the
// metadata wrapper) and bind it (IOTensor.cpp setupTensors).
// We copy only the fields graphExecute needs: name/dtype/rank/dimensions/type/
// quantizeParams come straight from the metadata wrapper (already in `dst` via
// the shallow struct copy), then we attach a raw client buffer.
static bool bindBuffer(Qnn_Tensor_t& dst, std::vector<void*>& owned) {
  size_t len = tensorByteLength(dst);
  if (len == 0) {
    fprintf(stderr, "[ERROR] tensor '%s' has unsupported dtype 0x%x or zero length\n",
            T_NAME(dst) ? T_NAME(dst) : "(null)", (unsigned)T_DTYPE(dst));
    return false;
  }
  void* mem = calloc(1, len);  // zero-initialized; matters for self-KV-cache init
  if (!mem) {
    fprintf(stderr, "[ERROR] calloc(%zu) failed\n", len);
    return false;
  }
  owned.push_back(mem);
  T_SET_MEMTYPE(dst, QNN_TENSORMEMTYPE_RAW);
  Qnn_ClientBuffer_t cb = QNN_CLIENT_BUFFER_INIT;
  cb.data     = mem;
  cb.dataSize = (uint32_t)len;
  T_SET_CLIENTBUF(dst, cb);
  return true;
}

// ----------------------------------------------------------------------------
// (c)+(d)+(e)+(f) Load one .bin: parse metadata, create context, retrieve the
// graph handle, and allocate+bind a client buffer for each input/output tensor.
// ----------------------------------------------------------------------------
static bool loadGraphFromBinary(QnnVtables& v,
                                Qnn_BackendHandle_t backend,
                                Qnn_DeviceHandle_t device,
                                const std::string& binPath,
                                LoadedGraph& lg) {
  // --- read the serialized context binary into memory ---
  std::ifstream in(binPath, std::ios::binary | std::ios::ate);
  if (!in) {
    fprintf(stderr, "[ERROR] cannot open %s\n", binPath.c_str());
    return false;
  }
  uint64_t bufferSize = (uint64_t)in.tellg();
  in.seekg(0);
  std::vector<uint8_t> buffer(bufferSize);
  if (!in.read(reinterpret_cast<char*>(buffer.data()), (std::streamsize)bufferSize)) {
    fprintf(stderr, "[ERROR] cannot read %s\n", binPath.c_str());
    return false;
  }

  // --- (c) parse graph + tensor metadata via the System library ---
  QnnSystemContext_Handle_t sysCtx = nullptr;
  CHECK_QNN(v.sys.systemContextCreate(&sysCtx), "systemContextCreate");

  const QnnSystemContext_BinaryInfo_t* binaryInfo = nullptr;
  Qnn_ContextBinarySize_t binaryInfoSize = 0;
  CHECK_QNN(v.sys.systemContextGetBinaryInfo(sysCtx,
                                             static_cast<void*>(buffer.data()),
                                             bufferSize,
                                             &binaryInfo,
                                             &binaryInfoSize),
            "systemContextGetBinaryInfo");

  // Walk the version-tagged binaryInfo union to find the (single) graph and its
  // input/output Qnn_Tensor_t arrays (copyMetadataToGraphsInfo logic).
  const QnnSystemContext_GraphInfo_t* graphs = nullptr;
  uint32_t numGraphs = 0;
  if (binaryInfo->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1) {
    graphs    = binaryInfo->contextBinaryInfoV1.graphs;
    numGraphs = binaryInfo->contextBinaryInfoV1.numGraphs;
  } else if (binaryInfo->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2) {
    graphs    = binaryInfo->contextBinaryInfoV2.graphs;
    numGraphs = binaryInfo->contextBinaryInfoV2.numGraphs;
  } else if (binaryInfo->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_3) {
    graphs    = binaryInfo->contextBinaryInfoV3.graphs;
    numGraphs = binaryInfo->contextBinaryInfoV3.numGraphs;
  } else {
    fprintf(stderr, "[ERROR] unrecognized binary info version %d\n", (int)binaryInfo->version);
    return false;
  }
  if (!graphs || numGraphs == 0) {
    fprintf(stderr, "[ERROR] no graphs in %s\n", binPath.c_str());
    return false;
  }
  if (numGraphs != 1) {
    // Whisper encoder/decoder bins each contain exactly one graph; warn but use [0].
    fprintf(stderr, "[WARN] %s has %u graphs; using graph[0]\n", binPath.c_str(), numGraphs);
  }

  // Extract graphName + input/output tensor arrays from the per-graph version union.
  const char* graphName = nullptr;
  const Qnn_Tensor_t* graphInputs = nullptr;
  uint32_t numGraphInputs = 0;
  const Qnn_Tensor_t* graphOutputs = nullptr;
  uint32_t numGraphOutputs = 0;
  if (graphs[0].version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1) {
    const auto& g     = graphs[0].graphInfoV1;
    graphName         = g.graphName;
    graphInputs       = g.graphInputs;
    numGraphInputs    = g.numGraphInputs;
    graphOutputs      = g.graphOutputs;
    numGraphOutputs   = g.numGraphOutputs;
  } else if (graphs[0].version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_2) {
    const auto& g     = graphs[0].graphInfoV2;
    graphName         = g.graphName;
    graphInputs       = g.graphInputs;
    numGraphInputs    = g.numGraphInputs;
    graphOutputs      = g.graphOutputs;
    numGraphOutputs   = g.numGraphOutputs;
  } else if (graphs[0].version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_3) {
    const auto& g     = graphs[0].graphInfoV3;
    graphName         = g.graphName;
    graphInputs       = g.graphInputs;
    numGraphInputs    = g.numGraphInputs;
    graphOutputs      = g.graphOutputs;
    numGraphOutputs   = g.numGraphOutputs;
  } else {
    fprintf(stderr, "[ERROR] unrecognized graph info version %d\n", (int)graphs[0].version);
    return false;
  }
  if (!graphName) {
    fprintf(stderr, "[ERROR] graph[0] has no name in %s\n", binPath.c_str());
    return false;
  }
  lg.name = graphName;

  // Deep-copy the tensor templates (struct + heap-copied dims) BEFORE freeing the
  // system context, since systemContextFree invalidates binaryInfo's memory.
  lg.inputs.resize(numGraphInputs);
  lg.outputs.resize(numGraphOutputs);

  // Deep-copy each metadata tensor template into our own storage, owning the
  // name + dims so they survive systemContextFree below.
  auto copyTensorTemplate = [&](const Qnn_Tensor_t& src, Qnn_Tensor_t& dst) {
    dst = src;  // shallow copy of all scalar fields (version, dtype, rank, type, quant...)
    // Own the name.
    const char* nm = T_NAME(src);
    if (nm) {
      lg.nameStore.emplace_back(nm);
      dst.v1.name = lg.nameStore.back().c_str();
    }
    // Own the dims array.
    uint32_t rank = T_RANK(src);
    if (rank > 0 && T_DIMS(src)) {
      lg.dimStore.emplace_back(T_DIMS(src), T_DIMS(src) + rank);
      dst.v1.dimensions = lg.dimStore.back().data();
    }
  };
  for (uint32_t i = 0; i < numGraphInputs; ++i)  copyTensorTemplate(graphInputs[i],  lg.inputs[i]);
  for (uint32_t o = 0; o < numGraphOutputs; ++o) copyTensorTemplate(graphOutputs[o], lg.outputs[o]);

  // System context handle no longer needed once metadata is copied out.
  v.sys.systemContextFree(sysCtx);
  sysCtx = nullptr;

  // --- (d) create the executable context from the binary (HTP deserialization) ---
  Qnn_ContextHandle_t context = nullptr;
  CHECK_QNN(v.core.contextCreateFromBinary(backend,
                                           device,
                                           /*config*/ nullptr,
                                           static_cast<void*>(buffer.data()),
                                           bufferSize,
                                           &context,
                                           /*profile*/ nullptr),
            "contextCreateFromBinary");

  // --- (e) retrieve the graph handle by name ---
  CHECK_QNN(v.core.graphRetrieve(context, lg.name.c_str(), &lg.graph), "graphRetrieve");

  // --- (f) allocate + bind a client buffer for each input/output tensor ---
  for (size_t i = 0; i < lg.inputs.size(); ++i) {
    if (!bindBuffer(lg.inputs[i], lg.ownedBuffers)) return false;
    lg.inIdx[T_NAME(lg.inputs[i])] = i;
  }
  for (size_t o = 0; o < lg.outputs.size(); ++o) {
    if (!bindBuffer(lg.outputs[o], lg.ownedBuffers)) return false;
    lg.outIdx[T_NAME(lg.outputs[o])] = o;
  }
  return true;
}

// ----------------------------------------------------------------------------
// (g) Execute a loaded graph (synchronous, on the NPU).
// ----------------------------------------------------------------------------
static bool runGraph(QnnVtables& v, LoadedGraph& lg) {
  Qnn_ErrorHandle_t e = v.core.graphExecute(
      lg.graph,
      lg.inputs.data(),  (uint32_t)lg.inputs.size(),
      lg.outputs.data(), (uint32_t)lg.outputs.size(),
      /*profile*/ nullptr,
      /*signal*/  nullptr);
  if (e != QNN_SUCCESS) {
    fprintf(stderr, "[ERROR] graphExecute('%s') failed (0x%llx)\n",
            lg.name.c_str(), (unsigned long long)e);
    return false;
  }
  return true;
}

// argmax over fp16 logits, casting each element to float for comparison.
static int argmaxFp16(const fp16* logits, int len) {
  int best = 0;
  float bestVal = (float)logits[0];
  for (int i = 1; i < len; ++i) {
    float val = (float)logits[i];
    if (val > bestVal) { bestVal = val; best = i; }
  }
  return best;
}

// ============================================================================
// MEL FRONT-END — PCM float32 -> log-Mel input_features [1,80,3000] fp16.
//
// Matches OpenAI Whisper / transformers WhisperFeatureExtractor exactly:
//   * zero-pad / truncate PCM to MEL_N_SAMPLES (480000),
//   * reflect-pad by n_fft/2=200 each side (torch.stft center=True),
//   * periodic Hann window length 400,
//   * hop 160 -> compute 3001 frames, drop the last -> 3000 (torch.stft[...,:-1]),
//   * direct one-sided DFT (201 bins) via precomputed cos/sin tables,
//   * power = re*re + im*im,
//   * mel[m][t] = sum_k mel_filters[k][m] * power[k][t],
//   * log10(max(mel,1e-10)); clamp to (global_max - 8.0); (x + 4.0)/4.0.
// Output is row-major [mel][time] => [1,80,3000], emitted as fp16 — the SAME
// layout the encoder's input_features buffer expects.
// ============================================================================

// Precomputed DFT/window tables, built once at startup (or first --dump-mel use).
struct MelFrontEnd {
  // mel_filters[k*80 + m], row-major [freq_bin=201][mel=80], float32.
  std::vector<float> melFilters;   // size 201*80 = 16080
  // periodic Hann window, length 400.
  std::vector<float> hann;         // size 400
  // cos/sin tables for the direct DFT: [201][400] each, flat row-major [k*400 + n].
  std::vector<float> cosTab;       // size 201*400
  std::vector<float> sinTab;       // size 201*400
  bool ready = false;

  // Load mel_filters.bin (201*80 float32 = 64320 bytes) and build the tables.
  bool init(const std::string& melFiltersPath) {
    std::ifstream f(melFiltersPath, std::ios::binary | std::ios::ate);
    if (!f) {
      fprintf(stderr, "[ERROR] cannot open mel filters %s\n", melFiltersPath.c_str());
      return false;
    }
    size_t bytes = (size_t)f.tellg();
    f.seekg(0);
    const size_t want = (size_t)MEL_FFT_BINS * MEL_N_MELS * sizeof(float);  // 64320
    if (bytes != want) {
      fprintf(stderr, "[ERROR] %s is %zu bytes, expected %zu (201*80 float32)\n",
              melFiltersPath.c_str(), bytes, want);
      return false;
    }
    melFilters.resize((size_t)MEL_FFT_BINS * MEL_N_MELS);
    if (!f.read(reinterpret_cast<char*>(melFilters.data()), (std::streamsize)want)) {
      fprintf(stderr, "[ERROR] failed reading %s\n", melFiltersPath.c_str());
      return false;
    }

    // Periodic Hann: w[n] = 0.5 - 0.5*cos(2*pi*n / N), n = 0..N-1 (NOT N-1).
    hann.resize(MEL_N_FFT);
    for (int n = 0; n < MEL_N_FFT; ++n) {
      hann[n] = 0.5f - 0.5f * cosf(2.0f * (float)M_PI * (float)n / (float)MEL_N_FFT);
    }

    // cos/sin tables for the one-sided DFT: angle = -2*pi*k*n / N_FFT.
    cosTab.resize((size_t)MEL_FFT_BINS * MEL_N_FFT);
    sinTab.resize((size_t)MEL_FFT_BINS * MEL_N_FFT);
    for (int k = 0; k < MEL_FFT_BINS; ++k) {
      for (int n = 0; n < MEL_N_FFT; ++n) {
        // Use double for the angle to keep the table accurate, store float.
        double ang = -2.0 * M_PI * (double)k * (double)n / (double)MEL_N_FFT;
        cosTab[(size_t)k * MEL_N_FFT + n] = (float)cos(ang);
        sinTab[(size_t)k * MEL_N_FFT + n] = (float)sin(ang);
      }
    }
    ready = true;
    return true;
  }

  // Compute the log-Mel features for a PCM buffer of ANY length.
  // Writes MEL_FEAT_ELEMS (240000) fp16 values row-major [mel][time] into `out`.
  void compute(const float* pcm, size_t pcmLen, fp16* out) const {
    // --- 1. zero-pad / truncate to exactly MEL_N_SAMPLES ---
    std::vector<float> x((size_t)MEL_N_SAMPLES, 0.0f);
    size_t n = pcmLen < (size_t)MEL_N_SAMPLES ? pcmLen : (size_t)MEL_N_SAMPLES;
    if (n > 0) memcpy(x.data(), pcm, n * sizeof(float));

    // --- 2. reflect-pad by MEL_PAD (200) on each side (torch reflect: edge sample
    //        excluded). Padded length = MEL_N_SAMPLES + 2*MEL_PAD. ---
    std::vector<float> xp((size_t)MEL_N_SAMPLES + 2 * MEL_PAD, 0.0f);
    // body
    memcpy(xp.data() + MEL_PAD, x.data(), (size_t)MEL_N_SAMPLES * sizeof(float));
    // left: xp[i] = x[MEL_PAD - i] for i=0..MEL_PAD-1  (mirror around index 0)
    for (int i = 0; i < MEL_PAD; ++i) xp[i] = x[(size_t)(MEL_PAD - i)];
    // right: mirror around the last index (MEL_N_SAMPLES-1)
    for (int i = 0; i < MEL_PAD; ++i) {
      xp[(size_t)MEL_PAD + MEL_N_SAMPLES + i] =
          x[(size_t)(MEL_N_SAMPLES - 2 - i)];
    }

    // --- 3. per-frame power spectrum, then mel, tracked for the global max ---
    // log_spec stored transposed-friendly as [mel][time] directly into a float
    // scratch so we can apply the global-max clamp before writing fp16.
    std::vector<float> logspec(MEL_FEAT_ELEMS);
    std::vector<float> frame(MEL_N_FFT);
    std::vector<float> power(MEL_FFT_BINS);
    float globalMax = -1e30f;

    for (int t = 0; t < MEL_N_FRAMES; ++t) {
      const float* seg = xp.data() + (size_t)t * MEL_HOP;
      // windowed frame
      for (int i = 0; i < MEL_N_FFT; ++i) frame[i] = seg[i] * hann[i];
      // direct DFT for the 201 one-sided bins
      for (int k = 0; k < MEL_FFT_BINS; ++k) {
        const float* cr = &cosTab[(size_t)k * MEL_N_FFT];
        const float* ci = &sinTab[(size_t)k * MEL_N_FFT];
        float re = 0.0f, im = 0.0f;
        for (int i = 0; i < MEL_N_FFT; ++i) {
          re += frame[i] * cr[i];
          im += frame[i] * ci[i];
        }
        power[k] = re * re + im * im;
      }
      // mel projection + log10, per mel bin
      for (int m = 0; m < MEL_N_MELS; ++m) {
        float acc = 0.0f;
        // mel_filters row-major [freq_bin][mel] => index k*80 + m
        for (int k = 0; k < MEL_FFT_BINS; ++k) {
          acc += melFilters[(size_t)k * MEL_N_MELS + m] * power[k];
        }
        float v = acc < 1e-10f ? 1e-10f : acc;
        float ls = log10f(v);
        logspec[(size_t)m * MEL_N_FRAMES + t] = ls;
        if (ls > globalMax) globalMax = ls;
      }
    }

    // --- 4. global-max clamp + affine normalize, then emit fp16 [mel][time] ---
    float floorVal = globalMax - 8.0f;
    for (size_t i = 0; i < MEL_FEAT_ELEMS; ++i) {
      float ls = logspec[i];
      if (ls < floorVal) ls = floorVal;
      ls = (ls + 4.0f) / 4.0f;
      out[i] = (fp16)ls;
    }
  }
};

// ============================================================================
// DETOKENIZER — token IDs -> UTF-8 text via vocab.bin.
//
// vocab.bin = VOCAB_TOKENS (50257) token byte-strings separated by '\0' (a token
// may be empty), with a trailing '\0'. Whisper is byte-level BPE, so raw byte
// concatenation of the matched token strings yields valid UTF-8.
// Special tokens (id >= 50257: EOT 50257 / SOT 50258 / lang 50259 / task 50359 /
// notimestamps 50363 / timestamps ...) carry no bytes and are skipped.
// ============================================================================
struct Detokenizer {
  std::vector<std::string> tokens;   // indexed by id 0..50256
  bool ready = false;

  bool init(const std::string& vocabPath) {
    std::ifstream f(vocabPath, std::ios::binary | std::ios::ate);
    if (!f) {
      fprintf(stderr, "[ERROR] cannot open vocab %s\n", vocabPath.c_str());
      return false;
    }
    size_t bytes = (size_t)f.tellg();
    f.seekg(0);
    std::vector<char> raw(bytes);
    if (bytes && !f.read(raw.data(), (std::streamsize)bytes)) {
      fprintf(stderr, "[ERROR] failed reading %s\n", vocabPath.c_str());
      return false;
    }
    // Split on '\0'. There are VOCAB_TOKENS strings, each followed by a '\0'
    // (including a trailing '\0' after the last token), so we read exactly
    // VOCAB_TOKENS records and stop.
    tokens.clear();
    tokens.reserve(VOCAB_TOKENS);
    size_t start = 0;
    for (size_t i = 0; i < bytes && (int)tokens.size() < VOCAB_TOKENS; ++i) {
      if (raw[i] == '\0') {
        tokens.emplace_back(raw.data() + start, i - start);
        start = i + 1;
      }
    }
    if ((int)tokens.size() != VOCAB_TOKENS) {
      fprintf(stderr, "[ERROR] vocab.bin parsed %zu tokens, expected %d\n",
              tokens.size(), VOCAB_TOKENS);
      return false;
    }
    ready = true;
    return true;
  }

  // Concatenate the raw bytes of non-special tokens; trim a single leading space.
  std::string decode(const std::vector<int>& ids) const {
    std::string out;
    for (int id : ids) {
      if (id < 0 || id >= VOCAB_TOKENS) continue;   // special tokens carry no bytes
      out += tokens[(size_t)id];
    }
    if (!out.empty() && out[0] == ' ') out.erase(out.begin());
    return out;
  }
};

// ============================================================================
// Resident decode pipeline: encoder + decoder graphs + assets, reused per request.
// Encapsulates one transcription so --pcm and --watch share the exact same path.
// ============================================================================
struct WhisperRunner {
  QnnVtables*  v   = nullptr;
  LoadedGraph* enc = nullptr;
  LoadedGraph* dec = nullptr;
  const MelFrontEnd*  mel    = nullptr;
  const Detokenizer*  detok  = nullptr;

  // Run the full pipeline on a PCM window: mel -> encoder -> decode -> detok.
  // Returns true on success and writes the transcript into `text`.
  bool transcribe(const float* pcm, size_t pcmLen, std::string& text) {
    // ---- mel front-end straight into the encoder's input_features buffer ----
    fp16* feat = (fp16*)enc->inBuf("input_features");
    if (!feat) {
      fprintf(stderr, "[ERROR] encoder has no 'input_features' input tensor\n");
      return false;
    }
    if (enc->inBytes("input_features") != MEL_FEAT_ELEMS * sizeof(fp16)) {
      fprintf(stderr, "[ERROR] input_features is %zu bytes, expected %zu\n",
              enc->inBytes("input_features"), MEL_FEAT_ELEMS * sizeof(fp16));
      return false;
    }
    mel->compute(pcm, pcmLen, feat);

    // ---- encoder once -> cross-KV caches; copy into decoder by name ----
    if (!runGraph(*v, *enc)) return false;
    for (int i = 0; i < 4; ++i) {
      for (const char* kv : {"k", "v"}) {
        std::string nm = std::string(kv) + "_cache_cross_" + std::to_string(i);
        void* src = enc->outBuf(nm);
        void* dst = dec->inBuf(nm);
        if (!src || !dst) {
          fprintf(stderr, "[ERROR] cross-KV name mismatch for '%s'\n", nm.c_str());
          return false;
        }
        memcpy(dst, src, dec->inBytes(nm));
      }
    }

    // ---- decode loop (forced English-transcribe-no-timestamps prefix) ----
    std::vector<int> output_ids =
        { SOT_TOKEN, LANG_EN_TOKEN, TASK_TRANSCRIBE, NOTIMESTAMPS_TOKEN };
    const int output_length = (int)output_ids.size();

    fp16* attn = (fp16*)dec->inBuf("attention_mask");
    int32_t* posIds = (int32_t*)dec->inBuf("position_ids");
    int32_t* inIds  = (int32_t*)dec->inBuf("input_ids");
    fp16* logits    = (fp16*)dec->outBuf("logits");
    if (!attn || !posIds || !inIds || !logits) {
      fprintf(stderr, "[ERROR] decoder missing a required tensor "
                      "(attention_mask/position_ids/input_ids/logits)\n");
      return false;
    }

    // Re-seed per-request state: each transcribe() call must start clean because
    // the daemon reuses these buffers across requests.
    for (int j = 0; j < MEAN_DECODE_LEN; ++j) attn[j] = (fp16)MASK_NEG;
    posIds[0] = 0;
    // self-KV *_in buffers must be zeroed before each fresh decode.
    for (int i = 0; i < 4; ++i) {
      for (const char* kv : {"k", "v"}) {
        std::string inNm = std::string(kv) + "_cache_self_" + std::to_string(i) + "_in";
        void* dst = dec->inBuf(inNm);
        if (!dst) {
          fprintf(stderr, "[ERROR] decoder missing '%s'\n", inNm.c_str());
          return false;
        }
        memset(dst, 0, dec->inBytes(inNm));
      }
    }

    for (int nstep = 0; nstep < MEAN_DECODE_LEN - 1; ++nstep) {
      inIds[0] = (int32_t)output_ids[nstep];
      attn[MEAN_DECODE_LEN - nstep - 1] = (fp16)0.0f;

      if (!runGraph(*v, *dec)) return false;

      int output_id = argmaxFp16(logits, VOCAB_SIZE);

      bool last_step = (nstep == MEAN_DECODE_LEN - 2);
      if (last_step || output_id == EOT_TOKEN) {
        output_ids.push_back(output_id);
        break;
      }
      if (nstep >= output_length - 1) output_ids.push_back(output_id);

      for (int i = 0; i < 4; ++i) {
        for (const char* kv : {"k", "v"}) {
          std::string inNm  = std::string(kv) + "_cache_self_" + std::to_string(i) + "_in";
          std::string outNm = std::string(kv) + "_cache_self_" + std::to_string(i) + "_out";
          void* src = dec->outBuf(outNm);
          void* dst = dec->inBuf(inNm);
          if (!src || !dst) {
            fprintf(stderr, "[ERROR] self-KV name mismatch ('%s'/'%s')\n",
                    outNm.c_str(), inNm.c_str());
            return false;
          }
          memcpy(dst, src, dec->inBytes(inNm));
        }
      }
      posIds[0] += 1;
    }

    text = detok->decode(output_ids);
    return true;
  }
};

// ----------------------------------------------------------------------------
// Small file helpers (raw POSIX, no extra libs) for the daemon channel.
// ----------------------------------------------------------------------------
static bool fileExists(const std::string& path) {
  struct stat st;
  return stat(path.c_str(), &st) == 0;
}

// Read an entire file into a byte vector. Returns false if it can't be opened.
static bool readWholeFile(const std::string& path, std::vector<uint8_t>& out) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) return false;
  size_t bytes = (size_t)f.tellg();
  f.seekg(0);
  out.resize(bytes);
  if (bytes && !f.read(reinterpret_cast<char*>(out.data()), (std::streamsize)bytes)) return false;
  return true;
}

// Write a string fully to a file (truncating). Flushes + closes before return.
static bool writeWholeFile(const std::string& path, const std::string& data) {
  std::ofstream f(path, std::ios::binary | std::ios::trunc);
  if (!f) return false;
  if (!data.empty()) f.write(data.data(), (std::streamsize)data.size());
  f.flush();
  return (bool)f;
}

// Create an empty marker file (presence => signal).
static bool touchFile(const std::string& path) {
  std::ofstream f(path, std::ios::binary | std::ios::trunc);
  return (bool)f;
}

// Channel filenames — MUST match com.scamshield.app.runtime.WhisperConfig exactly.
static constexpr const char* CH_IN_RAW    = "whisper_in.raw";
static constexpr const char* CH_IN_READY  = "whisper_in.ready";
static constexpr const char* CH_OUT_TXT   = "whisper_out.txt";
static constexpr const char* CH_OUT_READY = "whisper_out.ready";
static constexpr const char* CH_ERR_READY = "whisper_err.ready";
static constexpr int CH_POLL_US = 50 * 1000;   // 50 ms

static std::string joinPath(const std::string& dir, const char* name) {
  if (dir.empty()) return std::string(name);
  if (dir.back() == '/') return dir + name;
  return dir + "/" + name;
}

// ----------------------------------------------------------------------------
// Mode: --dump-mel <mel_filters.bin> <pcm.raw> <out.raw>  (no NPU; validation).
// ----------------------------------------------------------------------------
static int runDumpMel(const std::string& melFiltersPath,
                      const std::string& pcmPath,
                      const std::string& outPath) {
  MelFrontEnd mel;
  if (!mel.init(melFiltersPath)) return 5;

  std::vector<uint8_t> raw;
  if (!readWholeFile(pcmPath, raw)) {
    fprintf(stderr, "[ERROR] cannot open %s\n", pcmPath.c_str());
    return 5;
  }
  size_t nSamples = raw.size() / sizeof(float);
  const float* pcm = reinterpret_cast<const float*>(raw.data());

  std::vector<fp16> feat(MEL_FEAT_ELEMS);
  mel.compute(pcm, nSamples, feat.data());

  std::ofstream out(outPath, std::ios::binary | std::ios::trunc);
  if (!out) { fprintf(stderr, "[ERROR] cannot open %s for write\n", outPath.c_str()); return 5; }
  out.write(reinterpret_cast<const char*>(feat.data()),
            (std::streamsize)(MEL_FEAT_ELEMS * sizeof(fp16)));
  out.flush();
  if (!out) { fprintf(stderr, "[ERROR] failed writing %s\n", outPath.c_str()); return 5; }

  fprintf(stderr, "[dump-mel] %zu samples -> %s (%zu fp16, %zu bytes)\n",
          nSamples, outPath.c_str(), MEL_FEAT_ELEMS, MEL_FEAT_ELEMS * sizeof(fp16));
  return 0;
}

// ----------------------------------------------------------------------------
// Shared QNN bring-up for the NPU modes: load vtables + log/backend/device +
// both graphs. Returns 0 on success; fills the out-params.
// ----------------------------------------------------------------------------
static int bringUpQnn(const std::string& encoderPath,
                      const std::string& decoderPath,
                      const std::string& backendSo,
                      QnnVtables& v,
                      Qnn_BackendHandle_t& backend,
                      Qnn_DeviceHandle_t& device,
                      Qnn_LogHandle_t& logHandle,
                      LoadedGraph& enc,
                      LoadedGraph& dec) {
  const std::string systemSo = "libQnnSystem.so";
  if (!loadBackend(backendSo, v)) return 2;
  if (!loadSystem(systemSo, v))   return 2;

  logHandle = nullptr;
  if (v.core.logCreate) v.core.logCreate(nullptr, QNN_LOG_LEVEL_ERROR, &logHandle);

  backend = nullptr;
  if (v.core.backendCreate(logHandle, nullptr, &backend) != QNN_BACKEND_NO_ERROR) {
    fprintf(stderr, "[ERROR] backendCreate failed\n");
    return 3;
  }

  device = nullptr;
  if (v.core.deviceCreate) {
    Qnn_ErrorHandle_t de = v.core.deviceCreate(logHandle, nullptr, &device);
    if (de != QNN_SUCCESS && de != QNN_DEVICE_ERROR_UNSUPPORTED_FEATURE) {
      fprintf(stderr, "[ERROR] deviceCreate failed (0x%llx)\n", (unsigned long long)de);
      return 3;
    }
  }

  if (!loadGraphFromBinary(v, backend, device, encoderPath, enc)) return 4;
  if (!loadGraphFromBinary(v, backend, device, decoderPath, dec)) return 4;

  fprintf(stderr, "[INFO] encoder graph '%s': %zu inputs, %zu outputs\n",
          enc.name.c_str(), enc.inputs.size(), enc.outputs.size());
  fprintf(stderr, "[INFO] decoder graph '%s': %zu inputs, %zu outputs\n",
          dec.name.c_str(), dec.inputs.size(), dec.outputs.size());
  return 0;
}

// ----------------------------------------------------------------------------
// Mode: --pcm <enc.bin> <dec.bin> <vocab.bin> <mel_filters.bin> <pcm.raw> [backend]
//       PCM -> mel -> encoder -> decode -> detok -> print TEXT to stdout.
// ----------------------------------------------------------------------------
static int runPcm(const std::string& encoderPath,
                  const std::string& decoderPath,
                  const std::string& vocabPath,
                  const std::string& melFiltersPath,
                  const std::string& pcmPath,
                  const std::string& backendSo) {
  MelFrontEnd mel;
  if (!mel.init(melFiltersPath)) return 5;
  Detokenizer detok;
  if (!detok.init(vocabPath)) return 5;

  std::vector<uint8_t> raw;
  if (!readWholeFile(pcmPath, raw)) {
    fprintf(stderr, "[ERROR] cannot open %s\n", pcmPath.c_str());
    return 5;
  }
  size_t nSamples = raw.size() / sizeof(float);
  const float* pcm = reinterpret_cast<const float*>(raw.data());

  QnnVtables v;
  Qnn_BackendHandle_t backend = nullptr;
  Qnn_DeviceHandle_t  device  = nullptr;
  Qnn_LogHandle_t     logHandle = nullptr;
  LoadedGraph enc, dec;
  int rc = bringUpQnn(encoderPath, decoderPath, backendSo, v,
                      backend, device, logHandle, enc, dec);
  if (rc != 0) return rc;

  WhisperRunner runner;
  runner.v = &v; runner.enc = &enc; runner.dec = &dec;
  runner.mel = &mel; runner.detok = &detok;

  std::string text;
  if (!runner.transcribe(pcm, nSamples, text)) {
    fprintf(stderr, "[ERROR] transcription failed\n");
    return 8;
  }

  // Transcript to stdout (one line). stderr already carries the diagnostics.
  printf("%s\n", text.c_str());
  fflush(stdout);

  for (void* p : enc.ownedBuffers) free(p);
  for (void* p : dec.ownedBuffers) free(p);
  if (v.core.deviceFree && device)   v.core.deviceFree(device);
  if (v.core.backendFree && backend) v.core.backendFree(backend);
  if (v.core.logFree && logHandle)   v.core.logFree(logHandle);
  if (v.systemLib)  dlclose(v.systemLib);
  if (v.backendLib) dlclose(v.backendLib);
  return 0;
}

// ----------------------------------------------------------------------------
// Mode: --watch <enc.bin> <dec.bin> <vocab.bin> <mel_filters.bin> <channel_dir> [backend]
//       DAEMON. Load contexts + assets ONCE, then poll <channel_dir> every 50 ms.
//
// Channel contract (com.scamshield.app.runtime.WhisperConfig):
//   whisper_in.raw    app->daemon  mono float32 LE PCM window
//   whisper_in.ready  app->daemon  request marker (presence => request ready)
//   whisper_out.txt   daemon->app  UTF-8 transcript
//   whisper_out.ready daemon->app  result marker
//   whisper_err.ready daemon->app  error marker
//
// Per request: read whisper_in.raw -> transcribe -> write whisper_out.txt fully,
// THEN create whisper_out.ready (or whisper_err.ready on any failure). Finally
// delete whisper_in.ready and whisper_in.raw. Contexts stay resident; each
// request is ~150 ms. Loops forever.
// ----------------------------------------------------------------------------
static int runWatch(const std::string& encoderPath,
                    const std::string& decoderPath,
                    const std::string& vocabPath,
                    const std::string& melFiltersPath,
                    const std::string& channelDir,
                    const std::string& backendSo) {
  MelFrontEnd mel;
  if (!mel.init(melFiltersPath)) return 5;
  Detokenizer detok;
  if (!detok.init(vocabPath)) return 5;

  QnnVtables v;
  Qnn_BackendHandle_t backend = nullptr;
  Qnn_DeviceHandle_t  device  = nullptr;
  Qnn_LogHandle_t     logHandle = nullptr;
  LoadedGraph enc, dec;
  int rc = bringUpQnn(encoderPath, decoderPath, backendSo, v,
                      backend, device, logHandle, enc, dec);
  if (rc != 0) return rc;

  WhisperRunner runner;
  runner.v = &v; runner.enc = &enc; runner.dec = &dec;
  runner.mel = &mel; runner.detok = &detok;

  const std::string inRaw    = joinPath(channelDir, CH_IN_RAW);
  const std::string inReady  = joinPath(channelDir, CH_IN_READY);
  const std::string outTxt   = joinPath(channelDir, CH_OUT_TXT);
  const std::string outReady = joinPath(channelDir, CH_OUT_READY);
  const std::string errReady = joinPath(channelDir, CH_ERR_READY);

  fprintf(stderr, "[watch] resident; polling %s every %d ms\n",
          channelDir.c_str(), CH_POLL_US / 1000);

  for (;;) {
    if (!fileExists(inReady)) {
      usleep(CH_POLL_US);
      continue;
    }

    // A request is ready. Clear any stale result/error markers so the app never
    // sees this turn's output paired with the previous marker.
    unlink(outReady.c_str());
    unlink(errReady.c_str());

    bool ok = false;
    std::string text;
    std::vector<uint8_t> raw;
    if (!readWholeFile(inRaw, raw)) {
      fprintf(stderr, "[watch] ERROR: cannot read %s\n", inRaw.c_str());
    } else {
      size_t nSamples = raw.size() / sizeof(float);
      const float* pcm = reinterpret_cast<const float*>(raw.data());
      if (runner.transcribe(pcm, nSamples, text)) {
        ok = true;
        fprintf(stderr, "[watch] %zu samples -> \"%s\"\n", nSamples, text.c_str());
      } else {
        fprintf(stderr, "[watch] ERROR: transcription failed (%zu samples)\n", nSamples);
      }
    }

    if (ok) {
      // Write the transcript FULLY before creating the result marker.
      if (writeWholeFile(outTxt, text)) {
        touchFile(outReady.c_str());
      } else {
        fprintf(stderr, "[watch] ERROR: cannot write %s\n", outTxt.c_str());
        touchFile(errReady.c_str());
      }
    } else {
      touchFile(errReady.c_str());
    }

    // Consume the request: delete the marker first (so a fast app re-arm can't be
    // lost), then the payload.
    unlink(inReady.c_str());
    unlink(inRaw.c_str());
  }

  // Unreachable (loops forever); teardown kept for completeness / future signals.
  // for (void* p : enc.ownedBuffers) free(p);
  // for (void* p : dec.ownedBuffers) free(p);
  // return 0;
}

static void printUsage(const char* prog) {
  fprintf(stderr,
    "Usage:\n"
    "  %s <encoder.bin> <decoder.bin> <input_features.raw> [backend=libQnnHtp.so]\n"
    "      Legacy positional mode: prints decoded token IDs (one line).\n"
    "  %s --pcm <enc.bin> <dec.bin> <vocab.bin> <mel_filters.bin> <pcm.raw> [backend]\n"
    "      PCM -> mel -> encoder -> decode -> detokenize -> print TEXT.\n"
    "  %s --dump-mel <mel_filters.bin> <pcm.raw> <out.raw>\n"
    "      Compute fp16 mel [1,80,3000] (480000 bytes) and exit (no NPU).\n"
    "  %s --watch <enc.bin> <dec.bin> <vocab.bin> <mel_filters.bin> <channel_dir> [backend]\n"
    "      Resident daemon: poll <channel_dir> every 50 ms and transcribe.\n",
    prog, prog, prog, prog);
}

int main(int argc, char** argv) {
  // ===== Mode dispatch (new modes are opt-in via a leading flag) =====
  if (argc >= 2 && std::string(argv[1]) == "--dump-mel") {
    if (argc != 5) { printUsage(argv[0]); return 1; }
    return runDumpMel(argv[2], argv[3], argv[4]);
  }
  if (argc >= 2 && std::string(argv[1]) == "--pcm") {
    if (argc < 7) { printUsage(argv[0]); return 1; }
    const std::string backendSo = (argc >= 8) ? argv[7] : "libQnnHtp.so";
    return runPcm(argv[2], argv[3], argv[4], argv[5], argv[6], backendSo);
  }
  if (argc >= 2 && std::string(argv[1]) == "--watch") {
    if (argc < 7) { printUsage(argv[0]); return 1; }
    const std::string backendSo = (argc >= 8) ? argv[7] : "libQnnHtp.so";
    return runWatch(argv[2], argv[3], argv[4], argv[5], argv[6], backendSo);
  }

  // ===== Legacy positional mode: <enc.bin> <dec.bin> <input_features.raw> [backend] =====
  if (argc < 4) {
    printUsage(argv[0]);
    return 1;
  }
  const std::string encoderPath  = argv[1];
  const std::string decoderPath  = argv[2];
  const std::string featuresPath = argv[3];
  const std::string backendSo    = (argc >= 5) ? argv[4] : "libQnnHtp.so";
  const std::string systemSo     = "libQnnSystem.so";

  // ===== Load both QNN vtables (backend + system) =====
  QnnVtables v;
  if (!loadBackend(backendSo, v)) return 2;
  if (!loadSystem(systemSo, v))   return 2;

  // ===== (b) log, backend, device — created once, reused for both graphs =====
  Qnn_LogHandle_t logHandle = nullptr;
  if (v.core.logCreate) {
    v.core.logCreate(/*callback*/ nullptr, QNN_LOG_LEVEL_ERROR, &logHandle);
  }

  Qnn_BackendHandle_t backend = nullptr;
  if (v.core.backendCreate(logHandle, /*config*/ nullptr, &backend) != QNN_BACKEND_NO_ERROR) {
    fprintf(stderr, "[ERROR] backendCreate failed\n");
    return 3;
  }

  Qnn_DeviceHandle_t device = nullptr;
  if (v.core.deviceCreate) {
    Qnn_ErrorHandle_t de = v.core.deviceCreate(logHandle, /*config*/ nullptr, &device);
    if (de != QNN_SUCCESS && de != QNN_DEVICE_ERROR_UNSUPPORTED_FEATURE) {
      fprintf(stderr, "[ERROR] deviceCreate failed (0x%llx)\n", (unsigned long long)de);
      return 3;
    }
  }

  // ===== Load encoder + decoder ONCE =====
  LoadedGraph enc, dec;
  if (!loadGraphFromBinary(v, backend, device, encoderPath, enc)) return 4;
  if (!loadGraphFromBinary(v, backend, device, decoderPath, dec)) return 4;

  fprintf(stderr, "[INFO] encoder graph '%s': %zu inputs, %zu outputs\n",
          enc.name.c_str(), enc.inputs.size(), enc.outputs.size());
  fprintf(stderr, "[INFO] decoder graph '%s': %zu inputs, %zu outputs\n",
          dec.name.c_str(), dec.inputs.size(), dec.outputs.size());

  // ===== 1. Load input_features.raw straight into the encoder's fp16 input =====
  {
    void* dst = enc.inBuf("input_features");
    if (!dst) {
      fprintf(stderr, "[ERROR] encoder has no 'input_features' input tensor\n");
      return 5;
    }
    size_t need = enc.inBytes("input_features");      // [1,80,3000] fp16 = 480000 bytes
    std::ifstream fin(featuresPath, std::ios::binary | std::ios::ate);
    if (!fin) { fprintf(stderr, "[ERROR] cannot open %s\n", featuresPath.c_str()); return 5; }
    size_t have = (size_t)fin.tellg();
    fin.seekg(0);
    if (have != need) {
      fprintf(stderr, "[ERROR] %s is %zu bytes, expected %zu (fp16 [1,80,3000])\n",
              featuresPath.c_str(), have, need);
      return 5;
    }
    if (!fin.read(reinterpret_cast<char*>(dst), (std::streamsize)need)) {
      fprintf(stderr, "[ERROR] failed reading %s\n", featuresPath.c_str());
      return 5;
    }
  }

  // ===== Run the encoder once -> cross-attention KV caches =====
  if (!runGraph(v, enc)) return 6;

  // Copy each encoder cross-KV output into the matching decoder cross-KV input,
  // BY NAME (encoder out names == decoder in names: k/v_cache_cross_{0..3}).
  for (int i = 0; i < 4; ++i) {
    for (const char* kv : {"k", "v"}) {
      std::string nm = std::string(kv) + "_cache_cross_" + std::to_string(i);
      void* src = enc.outBuf(nm);
      void* dst = dec.inBuf(nm);
      if (!src || !dst) {
        fprintf(stderr, "[ERROR] cross-KV name mismatch for '%s' (enc out=%p, dec in=%p)\n",
                nm.c_str(), src, dst);
        return 6;
      }
      memcpy(dst, src, dec.inBytes(nm));
    }
  }

  // ===== 2. Seed the forced English-transcribe-no-timestamps prefix =====
  std::vector<int> output_ids = { SOT_TOKEN, LANG_EN_TOKEN, TASK_TRANSCRIBE, NOTIMESTAMPS_TOKEN };
  const int output_length = (int)output_ids.size();   // 4

  // attention_mask: fp16 length 200, init to MASK_NEG (-100.0).
  fp16* attn = (fp16*)dec.inBuf("attention_mask");
  if (!attn) { fprintf(stderr, "[ERROR] decoder has no 'attention_mask'\n"); return 7; }
  for (int j = 0; j < MEAN_DECODE_LEN; ++j) attn[j] = (fp16)MASK_NEG;

  // position_ids: int32 [1], starts at 0.
  int32_t* posIds = (int32_t*)dec.inBuf("position_ids");
  if (!posIds) { fprintf(stderr, "[ERROR] decoder has no 'position_ids'\n"); return 7; }
  posIds[0] = 0;

  // input_ids: int32 [1,1].
  int32_t* inIds = (int32_t*)dec.inBuf("input_ids");
  if (!inIds) { fprintf(stderr, "[ERROR] decoder has no 'input_ids'\n"); return 7; }

  // Self-KV caches: *_in buffers start zeroed (calloc already did this). We keep
  // input and output as SEPARATE buffers (no read/write aliasing) and copy each
  // *_out into the matching *_in after every step.
  // logits output buffer:
  fp16* logits = (fp16*)dec.outBuf("logits");
  if (!logits) { fprintf(stderr, "[ERROR] decoder has no 'logits' output\n"); return 7; }

  // ===== 3. Greedy decode loop =====
  for (int n = 0; n < MEAN_DECODE_LEN - 1; ++n) {
    // current input token = output_ids[n]
    inIds[0] = (int32_t)output_ids[n];

    // open one causal slot from the right end: index = 200 - n - 1
    attn[MEAN_DECODE_LEN - n - 1] = (fp16)0.0f;

    // decode step on the NPU
    if (!runGraph(v, dec)) return 8;

    // greedy argmax over the 51865 vocab dim (logits laid out [1,51865,1,1]).
    int output_id = argmaxFp16(logits, VOCAB_SIZE);

    // termination: decode budget exhausted OR EOT produced.
    bool last_step = (n == MEAN_DECODE_LEN - 2);
    if (last_step || output_id == EOT_TOKEN) {
      output_ids.push_back(output_id);
      break;
    }

    // adopt prediction only once we're past the forced prefix; otherwise the
    // forced token already sitting at output_ids[n+1] is fed next.
    if (n >= output_length - 1) {
      output_ids.push_back(output_id);
    }

    // KV-cache feedback: copy each self *_out into the matching self *_in for n+1.
    for (int i = 0; i < 4; ++i) {
      for (const char* kv : {"k", "v"}) {
        std::string inNm  = std::string(kv) + "_cache_self_" + std::to_string(i) + "_in";
        std::string outNm = std::string(kv) + "_cache_self_" + std::to_string(i) + "_out";
        void* src = dec.outBuf(outNm);
        void* dst = dec.inBuf(inNm);
        if (!src || !dst) {
          fprintf(stderr, "[ERROR] self-KV name mismatch ('%s'/'%s')\n", outNm.c_str(), inNm.c_str());
          return 8;
        }
        memcpy(dst, src, dec.inBytes(inNm));
      }
    }

    // advance position
    posIds[0] += 1;
  }

  // ===== Print token IDs: one space-separated line =====
  for (size_t i = 0; i < output_ids.size(); ++i) {
    if (i) printf(" ");
    printf("%d", output_ids[i]);
  }
  printf("\n");

  // ===== Teardown (best-effort; OS reclaims on exit anyway) =====
  for (void* p : enc.ownedBuffers) free(p);
  for (void* p : dec.ownedBuffers) free(p);
  if (v.core.deviceFree && device)  v.core.deviceFree(device);
  if (v.core.backendFree && backend) v.core.backendFree(backend);
  if (v.core.logFree && logHandle)   v.core.logFree(logHandle);
  if (v.systemLib)  dlclose(v.systemLib);
  if (v.backendLib) dlclose(v.backendLib);

  return 0;
}
