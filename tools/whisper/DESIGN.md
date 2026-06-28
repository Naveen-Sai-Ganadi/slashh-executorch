# Native QNN Whisper runner — design specs (deep design study, 2026-06-28)


# ================= decode_loop =================

All token IDs confirmed from the actual tokenizer. One important note: the reference `_transcribe_single_chunk` only seeds `output_ids` with `decoder_start_token_id` (50258) — it does **not** prepend language/task/notimestamps tokens; the model predicts those itself. I'll capture both the exact reference behavior and the optional forced-prefix variant.

Below is the complete spec.

---

# Whisper-tiny (multilingual) greedy decode — C++-ready spec

Source of truth: `qai_hub_models/models/_shared/hf_whisper/app.py::_transcribe_single_chunk` and `model.py`. Token IDs resolved from `openai/whisper-tiny` (`WhisperConfig`/`WhisperTokenizer`).

## Constants (whisper-tiny multilingual)

```cpp
// --- config (model.py + WhisperConfig for openai/whisper-tiny) ---
constexpr int   NUM_MEL_BINS      = 80;      // config.num_mel_bins
constexpr int   MELS_AUDIO_LEN    = 3000;    // AUDIO_EMB_LEN(1500) * 2
constexpr int   AUDIO_EMB_LEN     = 1500;    // cross-attn time dim
constexpr int   NUM_DECODER_BLOCKS= 4;       // config.decoder_layers
constexpr int   D_MODEL           = 384;     // config.d_model
constexpr int   NUM_HEADS         = 6;       // config.decoder_attention_heads
constexpr int   HEAD_DIM          = D_MODEL / NUM_HEADS;   // 64
constexpr int   VOCAB_SIZE        = 51865;   // config.vocab_size
constexpr int   MEAN_DECODE_LEN   = 200;     // MEAN_DECODE_LEN
constexpr int   SELF_CACHE_LEN    = MEAN_DECODE_LEN - 1;   // 199
constexpr float MASK_NEG          = -100.0f; // MASK_NEG

// --- token ids (WhisperTokenizer openai/whisper-tiny) ---
constexpr int SOT_TOKEN          = 50258;  // <|startoftranscript|> = decoder_start_token_id
constexpr int EOT_TOKEN          = 50257;  // <|endoftext|> = eos_token_id = pad_token_id
constexpr int LANG_EN_TOKEN      = 50259;  // <|en|>   (language tokens are 50259 + lang_index)
constexpr int TASK_TRANSCRIBE    = 50359;  // <|transcribe|>
constexpr int TASK_TRANSLATE     = 50358;  // <|translate|>
constexpr int NOTIMESTAMPS_TOKEN = 50363;  // <|notimestamps|>
constexpr int TIMESTAMP_BEGIN    = 50364;  // <|0.00|>  (timestamp tokens >= this)
```

Dtype note: the Python reference allocates fp32 tensors, but the exported `.pte`/QNN graph here is fp16 I/O (`--quantize_full_type float16 --quantize_io` in `get_hf_whisper_options`, model.py L143/L303). Build buffers as **fp16** to match the on-device tensors; treat values as fp16 throughout.

## Encoder — single call

```
encoder(input_features) -> 8 cross-KV tensors
```

| | name | shape | dtype |
|---|---|---|---|
| in | `input_features` | `[1, 80, 3000]` | fp16 |
| out | `k_cache_cross_{i}` (i=0..3) | `[6, 1, 64, 1500]` | fp16 |
| out | `v_cache_cross_{i}` (i=0..3) | `[6, 1, 1500, 64]` | fp16 |

Output order (`get_output_names`, model.py L99): interleaved per layer — `k_cache_cross_0, v_cache_cross_0, k_cache_cross_1, v_cache_cross_1, …, k_cache_cross_3, v_cache_cross_3` (8 tensors total). These are computed once and are **constant** across all decode steps.

## Decoder — one call per step

Input order is exactly: `input_ids`, `attention_mask`, then the **8 self-cache tensors interleaved** (k0,v0,k1,v1,k2,v2,k3,v3), then the **8 cross-cache tensors interleaved** (k0,v0,…,k3,v3), then `position_ids` (app.py L182–188; model.py `forward` slices `args[2:-1]` and `position_ids = args[-1]`).

| | name | shape | dtype |
|---|---|---|---|
| in | `input_ids` | `[1, 1]` | int32 |
| in | `attention_mask` | `[1, 1, 1, 200]` | fp16 |
| in | `k_cache_self_{i}_in` (i=0..3) | `[6, 1, 64, 199]` | fp16 |
| in | `v_cache_self_{i}_in` (i=0..3) | `[6, 1, 199, 64]` | fp16 |
| in | `k_cache_cross_{i}` (i=0..3) | `[6, 1, 64, 1500]` | fp16 |
| in | `v_cache_cross_{i}` (i=0..3) | `[6, 1, 1500, 64]` | fp16 |
| in | `position_ids` | `[1]` | int32 |
| out | `logits` | `[1, 51865]` (HF graph emits `[1,51865,1,1]`; argmax over dim of size 51865) | fp16 |
| out | `k_cache_self_{i}_out` (i=0..3) | `[6, 1, 64, 199]` | fp16 |
| out | `v_cache_self_{i}_out` (i=0..3) | `[6, 1, 199, 64]` | fp16 |

Output order (`get_output_names`, model.py L275): `logits, k_cache_self_0_out, v_cache_self_0_out, k_cache_self_1_out, v_cache_self_1_out, …, k_cache_self_3_out, v_cache_self_3_out`.

## Token seeding — IMPORTANT (matches the reference exactly)

The reference seeds the loop with **only** `SOT_TOKEN` (50258) (app.py L132: `output_ids = torch.tensor([[sot]])`). It does **not** force-feed language/task/notimestamps. The model predicts `<|en|>`, `<|transcribe|>`, `<|notimestamps|>` itself as the first few greedy steps, and they appear in `output_ids`; they're stripped later by `tokenizer.decode(..., skip_special_tokens=True)`.

For a deterministic forced English-transcribe-no-timestamps prefix (recommended for the ScamShield use case to skip language detection and timestamp tokens), seed instead:
`[SOT_TOKEN, LANG_EN_TOKEN, TASK_TRANSCRIBE, NOTIMESTAMPS_TOKEN]` = `[50258, 50259, 50359, 50363]`. The loop logic below handles a prefix of any length via `output_length` (it only adopts the argmax once `n >= output_length - 1`, feeding the forced tokens before that).

## Full greedy loop (C++-ready pseudocode)

```cpp
std::vector<int> transcribe_single_chunk(const fp16* input_features /*[1,80,3000]*/) {

    // ---- 1. Encoder (once) -> cross KV (constant across steps) ----
    fp16 k_cross[4][6*1*64*1500], v_cross[4][6*1*1500*64];
    encoder.run(/*in*/ {input_features},
                /*out*/ {k_cross[0],v_cross[0], k_cross[1],v_cross[1],
                         k_cross[2],v_cross[2], k_cross[3],v_cross[3]});

    // ---- 2. Seed start-of-transcript ----
    // Reference behavior: just [SOT]. Forced variant: [SOT, EN, TRANSCRIBE, NOTIMESTAMPS].
    std::vector<int> output_ids = { SOT_TOKEN };           // or the 4-token forced prefix
    const int output_length = (int)output_ids.size();      // L134

    int32_t position_ids[1] = { 0 };                       // L136

    // attention_mask: all MASK_NEG, length 200 (causal slots opened over time). L137-141
    fp16 attention_mask[1*1*1*200];
    for (int j = 0; j < MEAN_DECODE_LEN; ++j) attention_mask[j] = (fp16)MASK_NEG;

    // self KV caches: start at zeros. k:[6,1,64,199] v:[6,1,199,64]. L144-164
    fp16 k_self[4][6*1*64*199]  = {0};
    fp16 v_self[4][6*1*199*64]  = {0};

    fp16 logits[1*51865];

    // ---- 3. Decode loop: n = 0 .. MEAN_DECODE_LEN-2 (199 iterations max) ---- L166
    for (int n = 0; n < MEAN_DECODE_LEN - 1; ++n) {

        // current input token = output_ids[n]  (shape [1,1] int32). L168
        int32_t input_ids[1] = { (int32_t)output_ids[n] };

        // open one causal slot, filling from the RIGHT end of the mask. L171
        //   index = MEAN_DECODE_LEN - n - 1  -> 199, 198, 197, ...
        attention_mask[MEAN_DECODE_LEN - n - 1] = (fp16)0.0f;

        // decoder call: order = input_ids, attention_mask,
        //   self(k0,v0,k1,v1,k2,v2,k3,v3),
        //   cross(k0,v0,k1,v1,k2,v2,k3,v3),
        //   position_ids                                              L182-188 / model.py L213-227
        decoder.run(
            /*in*/ { input_ids, attention_mask,
                     k_self[0],v_self[0], k_self[1],v_self[1],
                     k_self[2],v_self[2], k_self[3],v_self[3],
                     k_cross[0],v_cross[0], k_cross[1],v_cross[1],
                     k_cross[2],v_cross[2], k_cross[3],v_cross[3],
                     position_ids },
            /*out*/ { logits,
                      k_self[0],v_self[0], k_self[1],v_self[1],   // self_out overwrites self_in
                      k_self[2],v_self[2], k_self[3],v_self[3] }   // L191 / model.py L275-280
        );
        // KV-cache feedback: self_*_out buffers ARE the self_*_in for step n+1
        // (decoder internally writes the new token's k/v into the freed slot).

        // greedy argmax over the 51865 vocab dim. L202
        int output_id = argmax(logits, /*len=*/VOCAB_SIZE);

        // termination: hit decode budget OR produced EOT. L204-206
        bool last_step = (n == MEAN_DECODE_LEN - 2);   // len(output_logits)==MEAN_DECODE_LEN-1
        if (last_step || output_id == EOT_TOKEN) {
            output_ids.push_back(output_id);
            break;
        }

        // adopt the prediction only once we're past any forced prefix. L207-208
        if (n >= output_length - 1) {
            output_ids.push_back(output_id);
        }
        // else: a forced-prefix token at output_ids[n+1] is fed next instead.

        // advance position. L211
        position_ids[0] += 1;
    }

    return output_ids;   // L213 ; caller strips special tokens when decoding to text
}
```

### Per-step bookkeeping rules (exact)
- **argmax (greedy):** `output_id = argmax_v logits[0, v]` over `v ∈ [0, 51865)`. No sampling, no temperature, no penalties (app.py L202).
- **KV feedback:** `k/v_cache_self_i_out` of step `n` becomes `k/v_cache_self_i_in` of step `n+1`. In-place reuse of the same buffer is correct (app.py L191 reassigns `kv_cache_self`). Cross caches never change.
- **attention_mask update:** before each decoder call, set `attention_mask[0,0,0, 200 - n - 1] = 0.0`. So step `n=0` opens index 199, `n=1` opens 198, … filling right-to-left; all other entries stay `MASK_NEG = -100.0` (app.py L171). The 199-wide active region mirrors the 199-deep self-cache.
- **position_ids:** starts at 0; incremented by 1 after each non-terminal step (app.py L136, L211). Passed as int32 `[1]`; the graph internally casts to int64 (model.py L228).
- **input_ids:** `output_ids[n]` (the token decoded at the previous step, or the seed/forced token), shape `[1,1]` int32 (app.py L168).
- **Termination:** stop when `output_id == EOT_TOKEN` (50257) or when 199 tokens have been produced (`n == 198`). The terminating token (including EOT) is appended to `output_ids` (app.py L204-208).
- **Forced-prefix gating:** predictions are appended only when `n >= output_length - 1`; before that the loop feeds `output_ids[n+1]` (the forced token already present), implementing teacher-forcing of the seed prefix (app.py L207). With the bare 1-token seed, `output_length = 1` so this is always true from `n = 0`.

## Files referenced
- `/Users/sumakatabattuni/Documents/Personal/slashh-executorch/.venv/lib/python3.11/site-packages/qai_hub_models/models/_shared/hf_whisper/app.py` (decode loop: `_transcribe_single_chunk`, L96–213)
- `/Users/sumakatabattuni/Documents/Personal/slashh-executorch/.venv/lib/python3.11/site-packages/qai_hub_models/models/_shared/hf_whisper/model.py` (I/O specs + constants: encoder L86–104, decoder L167–280, constants L39–57)

Caveats: (1) The reference seeds **only** `[SOT]`; language/task/notimestamps are model-predicted, not forced — the 4-token forced prefix is the optional deterministic variant. (2) HF decoder emits `logits` as `[1,51865,1,1]`; argmax is over the 51865 axis (Python does `torch.argmax(logits, 1)`). (3) Reference allocates fp32, but the exported QNN graph is fp16 I/O — use fp16 buffers on-device. (4) Token IDs above are the standard multilingual Whisper IDs and were verified against the live `openai/whisper-tiny` tokenizer.


# ================= capi_reference =================

I have everything required. Here is the precise reference.

---

# QNN C-API Reference (QAIRT 2.45.0.260326 headers)

Core API version: **major=2, minor=34, patch=0** (`QnnCommon.h`) → interface struct type is `QnnInterface_ImplementationV2_34_t`, union member name `v2_34`.
System API version: **1.9.0** (`System/QnnSystemCommon.h`) → `QnnSystemInterface_ImplementationV1_9_t`, union member name `v1_9`.
Target SoC: S25 Ultra `SM8750` = `QNN_SOC_MODEL_SM8750 = 69` (`QnnTypes.h:1780`).

Handle typedefs (all `void*` aliases, `QnnCommon.h`): `Qnn_BackendHandle_t`, `Qnn_DeviceHandle_t`, `Qnn_ContextHandle_t`, `Qnn_GraphHandle_t`, `Qnn_ProfileHandle_t`, `Qnn_SignalHandle_t`, `Qnn_LogHandle_t`, `Qnn_MemHandle_t`. `Qnn_ErrorHandle_t = uint64_t`; success is `QNN_SUCCESS == 0`.

---

## (a) Create context from binary + enumerate graphs/tensors

### QnnContext_createFromBinary (`QnnContext.h:922`)
```c
Qnn_ErrorHandle_t QnnContext_createFromBinary(Qnn_BackendHandle_t backend,
                                              Qnn_DeviceHandle_t device,
                                              const QnnContext_Config_t** config,
                                              const void* binaryBuffer,
                                              Qnn_ContextBinarySize_t binaryBufferSize,
                                              Qnn_ContextHandle_t* context,
                                              Qnn_ProfileHandle_t profile);
```
- `device` may be NULL (equiv. to default `QnnDevice_create`). `config` may be NULL (NULL-terminated array of `const QnnContext_Config_t*`). `profile` NULL disables profiling.
- `Qnn_ContextBinarySize_t` is `uint64_t` (`QnnTypes.h:326`).
- Function-table typedef `QnnContext_CreateFromBinaryFn_t` (`QnnInterface.h:182`) has identical parameter order; table field `contextCreateFromBinary` (`QnnInterface.h:533`).

After creating the context, get each graph handle for execute by name:
### QnnGraph_retrieve (`QnnGraph.h:548`)
```c
Qnn_ErrorHandle_t QnnGraph_retrieve(Qnn_ContextHandle_t contextHandle,
                                    const char* graphName,
                                    Qnn_GraphHandle_t* graphHandle);
```
Table field `graphRetrieve` (`QnnInterface.h:541`).

### Enumerating graphs / per-tensor name, dims, dataType — via the System library
Use the **System interface** (`libQnnSystem.so`), which parses the binary's metadata without a backend.

Create a system context (`System/QnnSystemContext.h:445`):
```c
typedef void* QnnSystemContext_Handle_t;
Qnn_ErrorHandle_t QnnSystemContext_create(QnnSystemContext_Handle_t* sysCtxHandle);
```

Get binary info (`System/QnnSystemContext.h:480`):
```c
Qnn_ErrorHandle_t QnnSystemContext_getBinaryInfo(QnnSystemContext_Handle_t sysCtxHandle,
                                                 void* binaryBuffer,
                                                 uint64_t binaryBufferSize,
                                                 const QnnSystemContext_BinaryInfo_t** binaryInfo,
                                                 Qnn_ContextBinarySize_t* binaryInfoSize);
```
(The returned `binaryInfo` memory is owned by QNN; free with `QnnSystemContext_free`, `:531`. A non-deprecated alternative is `QnnSystemContext_getMetadata`, `:514`, which omits the size out-param and takes `const void*`.)

Walk the returned struct (`System/QnnSystemContext.h`). It is a versioned tagged union:
```c
typedef struct {                                  // :399
  QnnSystemContext_BinaryInfoVersion_t version;   // V1=0x01, V2=0x02, V3=0x03  (:70)
  union {
    QnnSystemContext_BinaryInfoV1_t contextBinaryInfoV1;
    QnnSystemContext_BinaryInfoV2_t contextBinaryInfoV2;
    QnnSystemContext_BinaryInfoV3_t contextBinaryInfoV3;
  };
} QnnSystemContext_BinaryInfo_t;
```
All three BinaryInfo versions carry graph enumeration via these fields (V1 `:217`, V2 `:277`, V3 `:340`):
```c
  uint32_t numGraphs;
  QnnSystemContext_GraphInfo_t* graphs;   // array of length numGraphs
```
Each `QnnSystemContext_GraphInfo_t` (`:193`) is itself versioned (`GraphInfoVersion_t`: V1=0x01, V2=0x02, V3=0x03, `:62`):
```c
typedef struct {
  QnnSystemContext_GraphInfoVersion_t version;
  union { QnnSystemContext_GraphInfoV1_t graphInfoV1;
          QnnSystemContext_GraphInfoV2_t graphInfoV2;
          QnnSystemContext_GraphInfoV3_t graphInfoV3; };
} QnnSystemContext_GraphInfo_t;
```
`QnnSystemContext_GraphInfoV1_t` (`:86`) — present (with the same first 5 fields) in V2 (`:115`) and V3 (`:150`):
```c
  const char*   graphName;
  uint32_t      numGraphInputs;
  Qnn_Tensor_t* graphInputs;     // array, length numGraphInputs
  uint32_t      numGraphOutputs;
  Qnn_Tensor_t* graphOutputs;    // array, length numGraphOutputs
```
For each `Qnn_Tensor_t` in `graphInputs`/`graphOutputs`, read **name**, **dims**, **dataType** from the V1/V2 body (see section b): `name`, `rank` + `dimensions[rank]` (uint32 array), `dataType`. Use these to allocate your client buffers before execute.

---

## (b) `Qnn_Tensor_t` across versions (V1/V2) — clientBuf, memType, dataType

### Versioned wrapper (`QnnTypes.h:1505`)
```c
typedef enum {                       // :1493
  QNN_TENSOR_VERSION_1 = 1,
  QNN_TENSOR_VERSION_2 = 2,
  QNN_TENSOR_VERSION_UNDEFINED = 0x7FFFFFFF
} Qnn_TensorVersion_t;

typedef struct {
  Qnn_TensorVersion_t version;       // tag — ALWAYS read this first
  union { Qnn_TensorV1_t v1; Qnn_TensorV2_t v2; };
} Qnn_Tensor_t;
```
Initializer `QNN_TENSOR_INIT` (`:1517`) defaults `version = QNN_TENSOR_VERSION_1`. The System library may hand you either V1 or V2 tensors, so branch on `version`.

### Qnn_TensorV1_t (`QnnTypes.h:1314`)
```c
typedef struct {
  uint32_t                id;
  const char*             name;
  Qnn_TensorType_t        type;          // e.g. QNN_TENSOR_TYPE_APP_WRITE=0 / APP_READ=1
  Qnn_TensorDataFormat_t  dataFormat;    // uint32; DENSE/FLAT_BUFFER = 0
  Qnn_DataType_t          dataType;
  Qnn_QuantizeParams_t    quantizeParams;
  uint32_t                rank;
  uint32_t*               dimensions;    // length == rank
  Qnn_TensorMemType_t     memType;
  union {
    Qnn_ClientBuffer_t    clientBuf;     // when memType == QNN_TENSORMEMTYPE_RAW
    Qnn_MemHandle_t       memHandle;     // when memType == QNN_TENSORMEMTYPE_MEMHANDLE
  };
} Qnn_TensorV1_t;
```

### Qnn_TensorV2_t (`QnnTypes.h:1420`)
Same leading fields as V1, then a 3-way data union plus extra trailing fields:
```c
typedef struct {
  uint32_t                id;
  const char*             name;
  Qnn_TensorType_t        type;
  Qnn_TensorDataFormat_t  dataFormat;
  Qnn_DataType_t          dataType;
  Qnn_QuantizeParams_t    quantizeParams;
  uint32_t                rank;
  uint32_t*               dimensions;
  Qnn_TensorMemType_t     memType;
  union {
    Qnn_ClientBuffer_t      clientBuf;     // QNN_TENSORMEMTYPE_RAW
    Qnn_MemHandle_t         memHandle;     // QNN_TENSORMEMTYPE_MEMHANDLE
    Qnn_TensorRetrieveRaw_t* retrieveRaw;  // QNN_TENSORMEMTYPE_RETRIEVE_RAW (V2 only)
  };
  uint8_t*                isDynamicDimensions;  // optional, length rank or NULL
  Qnn_SparseParams_t      sparseParams;
  uint8_t                 isProduced;
} Qnn_TensorV2_t;
```

### Setting clientBuf.data / clientBuf.dataSize
`Qnn_ClientBuffer_t` (`QnnTypes.h:1208`):
```c
typedef struct {
  void*    data;        // app-owned raw pointer
  uint32_t dataSize;    // size in BYTES (note: uint32, not uint64)
} Qnn_ClientBuffer_t;
```
For a V1 tensor:
```c
t.version = QNN_TENSOR_VERSION_1;
t.v1.memType            = QNN_TENSORMEMTYPE_RAW;
t.v1.clientBuf.data     = myBufferPtr;
t.v1.clientBuf.dataSize = (uint32_t) byteCount;
```
For a V2 tensor use `t.v2.memType` / `t.v2.clientBuf.data` / `t.v2.clientBuf.dataSize`. `dataSize` is bytes = (product of `dimensions[0..rank-1]`) × (bytes-per-element of `dataType`).

### memType enum — `Qnn_TensorMemType_t` (`QnnTypes.h:1193`)
```c
QNN_TENSORMEMTYPE_RAW          = 0   // use clientBuf
QNN_TENSORMEMTYPE_MEMHANDLE    = 1   // use memHandle (shared/ION/dmabuf)
QNN_TENSORMEMTYPE_RETRIEVE_RAW = 2   // V2 only, use retrieveRaw callback
QNN_TENSORMEMTYPE_UNDEFINED    = 0x7FFFFFFF
```

### Qnn_DataType_t — required enum values (`QnnTypes.h:127`)
```c
QNN_DATATYPE_FLOAT_16 = 0x0216   // 16-bit float  (2 bytes/elem)
QNN_DATATYPE_INT_32   = 0x0032   // 32-bit signed int (4 bytes/elem)
```
Other commonly needed ones: `QNN_DATATYPE_FLOAT_32 = 0x0232`, `QNN_DATATYPE_UINT_32 = 0x0132`, `QNN_DATATYPE_INT_8 = 0x0008`, `QNN_DATATYPE_UFIXED_POINT_8 = 0x0408`, `QNN_DATATYPE_UFIXED_POINT_16 = 0x0416`, `QNN_DATATYPE_UNDEFINED = 0x7FFFFFFF`. Note the bit-width is encoded in the low byte in **decimal** (e.g. INT_16=0x0016, INT_32=0x0032, INT_64=0x0064) — do not assume hex bit counts.

The headers expose **no** `QNN_TENSOR_GET_*`/`SET_*` accessor macros in this SDK (they live in the sample `QnnTypeMacros.hpp`, not these public headers); access the union members directly after checking `.version`.

---

## (c) QnnGraph_execute (`QnnGraph.h:704`)
```c
Qnn_ErrorHandle_t QnnGraph_execute(Qnn_GraphHandle_t graphHandle,
                                   const Qnn_Tensor_t* inputs,
                                   uint32_t numInputs,
                                   Qnn_Tensor_t* outputs,
                                   uint32_t numOutputs,
                                   Qnn_ProfileHandle_t profileHandle,
                                   Qnn_SignalHandle_t signalHandle);
```
- `inputs` is a contiguous array of `Qnn_Tensor_t` (length `numInputs`); `outputs` likewise (length `numOutputs`, written by the call). `inputs` is `const`, `outputs` is mutable.
- `profileHandle` / `signalHandle` may be NULL.
- Execute accepts only `QNN_TENSOR_TYPE_APP_READ`, `_APP_WRITE`, `_APP_READWRITE` (and NULL/optional) tensors (`QnnGraph.h:642`). Synchronous, blocking.
- Table field `graphExecute`; typedef `QnnGraph_ExecuteFn_t` (`QnnInterface.h:337`, same param order).

---

## (d) Loading the function table from libQnnHtp.so (dlopen + getProviders)

The backend lib (`libQnnHtp.so`) exports `QnnInterface_getProviders` (`QnnInterface.h:730`):
```c
Qnn_ErrorHandle_t QnnInterface_getProviders(const QnnInterface_t*** providerList,
                                            uint32_t* numProviders);
```

`QnnInterface_t` (`QnnInterface.h:680`):
```c
typedef struct {
  uint32_t          backendId;
  const char*       providerName;
  Qnn_ApiVersion_t  apiVersion;     // coreApiVersion + backendApiVersion
  union {
    QNN_INTERFACE_VER_TYPE  QNN_INTERFACE_VER_NAME;  // == v2_34 for this SDK
  };
} QnnInterface_t;
```
Because core API = 2.34, the macros resolve the union member to **`v2_34`** of type `QnnInterface_ImplementationV2_34_t`. That struct (`QnnInterface.h:529`+) is the function table; relevant fields:
```c
  QnnContext_CreateFn_t            contextCreate;             // :529
  QnnContext_CreateFromBinaryFn_t  contextCreateFromBinary;   // :533
  QnnContext_FreeFn_t              contextFree;               // :534
  QnnGraph_RetrieveFn_t            graphRetrieve;             // :541
  QnnGraph_ExecuteFn_t             graphExecute;              // :542
  QnnTensor_CreateGraphTensorFn_t  tensorCreateGraphTensor;   // :546
  ... (plus backend create, device, log, profile, mem, etc.)
```

Typical load sequence:
```c
void* lib = dlopen("libQnnHtp.so", RTLD_NOW | RTLD_LOCAL);
typedef Qnn_ErrorHandle_t (*GetProvidersFn)(const QnnInterface_t***, uint32_t*);
GetProvidersFn getProviders =
    (GetProvidersFn) dlsym(lib, "QnnInterface_getProviders");

const QnnInterface_t** providers = NULL;
uint32_t numProviders = 0;
getProviders(&providers, &numProviders);          // QNN_SUCCESS == 0

// Pick the provider whose apiVersion.coreApiVersion.major == QNN_API_VERSION_MAJOR (2)
// and .minor >= QNN_API_VERSION_MINOR you compiled against (34).
const QnnInterface_t* iface = providers[0];
QNN_INTERFACE_VER_TYPE api = iface->QNN_INTERFACE_VER_NAME;  // i.e. iface->v2_34

api.contextCreateFromBinary(backend, device, cfg, buf, bufSize, &ctx, NULL);
api.graphRetrieve(ctx, "graphName", &graph);
api.graphExecute(graph, inputs, numInputs, outputs, numOutputs, NULL, NULL);
```
Provider pointers are valid only while the library stays loaded. Validate the version before use rather than blindly indexing `[0]`.

For the System-side enumeration (section a) the parallel symbol in **`libQnnSystem.so`** is `QnnSystemInterface_getProviders` (`System/QnnSystemInterface.h:321`):
```c
Qnn_ErrorHandle_t QnnSystemInterface_getProviders(const QnnSystemInterface_t*** providerList,
                                                  uint32_t* numProviders);
```
`QnnSystemInterface_t` (`:270`) union member resolves to **`v1_9`** (`QnnSystemInterface_ImplementationV1_9_t`), whose fields include `systemContextCreate`, `systemContextGetBinaryInfo`, `systemContextGetMetaData`, `systemContextFree` (`:220`–`223`).

---

Header file paths (all under `/private/tmp/claude-501/-Users-sumakatabattuni-Documents-Personal-slashh-edge/0573850d-791b-4a19-ab79-95d1c47bf752/scratchpad/qairt245/qairt/2.45.0.260326/include/QNN/`):
- `QnnContext.h` (createFromBinary :922; CreateFromBinaryFn_t typedef in QnnInterface.h:182)
- `QnnGraph.h` (execute :704; retrieve :548)
- `QnnTensor.h` (create/update ops; tensor structs defined in QnnTypes.h)
- `QnnTypes.h` (Qnn_Tensor_t :1505, V1 :1314, V2 :1420; Qnn_ClientBuffer_t :1208; Qnn_TensorMemType_t :1193; Qnn_DataType_t :127; Qnn_TensorType_t :229)
- `QnnInterface.h` (QnnInterface_t :680; table type :529; getProviders :730)
- `QnnCommon.h` (API version 2.34.0 :46–48; handle typedefs)
- `System/QnnSystemContext.h` (BinaryInfo/GraphInfo structs; getBinaryInfo :480; getMetadata :514)
- `System/QnnSystemInterface.h` (system table :270; getProviders :321; System API 1.9.0)


# ================= netrun_skeleton =================

I now have a complete and verified understanding of the full sequence. I have all the real function names, struct fields, and the exact orchestration order from the canonical QNN SampleApp source. Let me compile the annotated skeleton.

---

## Source reality check

The `examples/QNN/NetRun/` directory in QAIRT 2.45.0 contains **only shell wrappers** (`android-qnn-net-run.sh`, etc.) that shell out to the prebuilt `qnn-net-run` binary — there is no `.cpp`/`.hpp` there. The actual C-API reference implementation of the exact sequence you asked for lives in the sibling example **`examples/QNN/SampleApp/SampleApp/src/`**, which is the canonical "load-from-context-binary + execute" source. Everything below is quoted from those files (paths are absolute):

- `.../SampleApp/SampleApp/src/Utils/DynamicLoadUtil.cpp` — (a) dlopen + `QnnInterface_getProviders` / `QnnSystemInterface_getProviders`
- `.../SampleApp/SampleApp/src/SampleApp.hpp` — `QnnFunctionPointers` struct (the resolved vtable)
- `.../SampleApp/SampleApp/src/QnnSampleApp.cpp` — (b) backend/device, (c) `systemContextGetBinaryInfo`, (d) `contextCreateFromBinary`, (e) `graphRetrieve`, (g) `graphExecute`
- `.../SampleApp/SampleApp/src/Utils/QnnSampleAppUtils.cpp` — (c→e) `copyMetadataToGraphsInfo` → `QnnSystemContext_GraphInfoV{1,3}_t` → `Qnn_Tensor_t` handles
- `.../SampleApp/SampleApp/src/Utils/IOTensor.cpp` — (f) client-buffer alloc/`QNN_TENSOR_SET_CLIENT_BUF`, (h) output read via `QNN_TENSOR_GET_CLIENT_BUF(tensor).data`
- `.../SampleApp/SampleApp/src/WrapperUtils/QnnWrapperUtils.hpp` — `GraphInfo_t` struct

---

## Annotated minimal C++ skeleton (real names & fields quoted from source)

```cpp
// ============================================================================
// Minimal QNN HTP "load saved context binary + execute" sequence.
// Field/function names are quoted verbatim from QAIRT 2.45.0 SampleApp source.
// Headers: QnnInterface.h, System/QnnSystemInterface.h, QnnContext.h,
//          QnnGraph.h, QnnTensor.h, QnnTypes.h, QnnBackend.h, QnnDevice.h
// ============================================================================
#include <dlfcn.h>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>

#include "QnnInterface.h"
#include "System/QnnSystemInterface.h"
#include "QnnTypes.h"   // Qnn_Tensor_t, Qnn_ClientBuffer_t, QNN_TENSOR_INIT, QNN_CLIENT_BUFFER_INIT

// Function-pointer signatures of the two entry symbols (DynamicLoadUtil.cpp:18-22)
typedef Qnn_ErrorHandle_t (*QnnInterfaceGetProvidersFn_t)(
    const QnnInterface_t*** providerList, uint32_t* numProviders);
typedef Qnn_ErrorHandle_t (*QnnSystemInterfaceGetProvidersFn_t)(
    const QnnSystemInterface_t*** providerList, uint32_t* numProviders);

int main() {
  // ==========================================================================
  // (a) LOAD BACKEND libQnnHtp.so AND RESOLVE THE INTERFACE VTABLE
  //     DynamicLoadUtil.cpp:42-88 — getQnnFunctionPointers()
  // ==========================================================================
  void* libBackendHandle = dlopen("libQnnHtp.so", RTLD_NOW | RTLD_GLOBAL);   // DL_NOW|DL_GLOBAL

  // Resolve the one exported symbol every QNN backend .so provides:
  auto getInterfaceProviders =
      (QnnInterfaceGetProvidersFn_t)dlsym(libBackendHandle, "QnnInterface_getProviders");

  const QnnInterface_t** interfaceProviders{nullptr};
  uint32_t numProviders{0};
  getInterfaceProviders(&interfaceProviders, &numProviders);   // QNN_SUCCESS expected

  // Pick the provider whose core API version is compatible, then grab the
  // versioned interface struct member 'QNN_INTERFACE_VER_NAME' (== .v2.10 etc).
  // This QNN_INTERFACE_VER_TYPE struct holds ALL the function pointers below.
  // (DynamicLoadUtil.cpp:75-82)
  QNN_INTERFACE_VER_TYPE qnnInterface;
  for (uint32_t i = 0; i < numProviders; i++) {
    if (QNN_API_VERSION_MAJOR == interfaceProviders[i]->apiVersion.coreApiVersion.major &&
        QNN_API_VERSION_MINOR <= interfaceProviders[i]->apiVersion.coreApiVersion.minor) {
      qnnInterface = interfaceProviders[i]->QNN_INTERFACE_VER_NAME;   // <-- the vtable
      break;
    }
  }

  // --- Same dance for libQnnSystem.so to get the reflection/metadata API ---
  //     DynamicLoadUtil.cpp:124-176 — getQnnSystemFunctionPointers()
  void* libSystemHandle = dlopen("libQnnSystem.so", RTLD_NOW | RTLD_LOCAL);
  auto getSystemInterfaceProviders =
      (QnnSystemInterfaceGetProvidersFn_t)dlsym(libSystemHandle, "QnnSystemInterface_getProviders");

  const QnnSystemInterface_t** systemInterfaceProviders{nullptr};
  uint32_t numSysProviders{0};
  getSystemInterfaceProviders(&systemInterfaceProviders, &numSysProviders);

  QNN_SYSTEM_INTERFACE_VER_TYPE qnnSystemInterface;
  for (uint32_t i = 0; i < numSysProviders; i++) {
    if (QNN_SYSTEM_API_VERSION_MAJOR == systemInterfaceProviders[i]->systemApiVersion.major &&
        QNN_SYSTEM_API_VERSION_MINOR <= systemInterfaceProviders[i]->systemApiVersion.minor) {
      qnnSystemInterface = systemInterfaceProviders[i]->QNN_SYSTEM_INTERFACE_VER_NAME;
      break;
    }
  }

  // ==========================================================================
  // (b) CREATE BACKEND + (optional) DEVICE
  //     log first, then backend, then device. (QnnSampleApp.cpp)
  // ==========================================================================
  // Logging (initialize():114) — optional but this is the documented order:
  Qnn_LogHandle_t  logHandle{nullptr};
  qnnInterface.logCreate(/*callback*/nullptr, QNN_LOG_LEVEL_ERROR, &logHandle);

  // Backend (initializeBackend():213):
  Qnn_BackendHandle_t backendHandle{nullptr};
  const QnnBackend_Config_t** backendConfig{nullptr};
  qnnInterface.backendCreate(logHandle, backendConfig, &backendHandle);   // QNN_BACKEND_NO_ERROR

  // Device (createDevice():782-790) — guard on the fn ptr; HTP returns SUCCESS
  // or QNN_DEVICE_ERROR_UNSUPPORTED_FEATURE, both acceptable:
  Qnn_DeviceHandle_t deviceHandle{nullptr};
  if (nullptr != qnnInterface.deviceCreate) {
    qnnInterface.deviceCreate(logHandle, /*QnnDevice_Config_t**/nullptr, &deviceHandle);
  }

  // ==========================================================================
  // (c) READ THE .bin AND PARSE GRAPH+TENSOR METADATA
  //     createFromBinary():429-489
  // ==========================================================================
  // Read serialized context binary into a byte buffer:
  std::ifstream in("model.bin", std::ios::binary | std::ios::ate);
  uint64_t bufferSize = (uint64_t)in.tellg();
  in.seekg(0);
  std::vector<uint8_t> buffer(bufferSize);
  in.read(reinterpret_cast<char*>(buffer.data()), bufferSize);

  // Create a SYSTEM context handle, then ask it to describe the binary:
  QnnSystemContext_Handle_t sysCtxHandle{nullptr};
  qnnSystemInterface.systemContextCreate(&sysCtxHandle);                    // QNN_SUCCESS

  const QnnSystemContext_BinaryInfo_t* binaryInfo{nullptr};
  Qnn_ContextBinarySize_t binaryInfoSize{0};
  qnnSystemInterface.systemContextGetBinaryInfo(                           // QnnSampleApp.cpp:472
      sysCtxHandle,
      static_cast<void*>(buffer.data()),
      bufferSize,
      &binaryInfo,          // OUT: pointer to const metadata struct (owned by sysCtx)
      &binaryInfoSize);

  // ---- Translate binaryInfo -> our own GraphInfo_t[] (QnnSampleAppUtils.cpp) ----
  // GraphInfo_t is (QnnWrapperUtils.hpp:118-125):
  //   struct GraphInfo { Qnn_GraphHandle_t graph; char* graphName;
  //                      Qnn_Tensor_t* inputTensors;  uint32_t numInputTensors;
  //                      Qnn_Tensor_t* outputTensors; uint32_t numOutputTensors; };
  //
  // binaryInfo is a version-tagged union; copyMetadataToGraphsInfo():391 switches on
  // binaryInfo->version ∈ { QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1/2/3 } and reads
  //   binaryInfo->contextBinaryInfoV{1,2,3}.graphs   (QnnSystemContext_GraphInfo_t*)
  //   binaryInfo->contextBinaryInfoV{1,2,3}.numGraphs
  // Each graph entry (copyGraphsInfo():357-363) switches on graphsInput[g].version
  // ∈ { QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1 / _3 } and reads
  //   .graphInfoV1 / .graphInfoV3, whose fields are (copyGraphsInfoV1():281):
  //     graphInfoSrc->graphName
  //     graphInfoSrc->graphInputs   (Qnn_Tensor_t*)  / graphInfoSrc->numGraphInputs
  //     graphInfoSrc->graphOutputs  (Qnn_Tensor_t*)  / graphInfoSrc->numGraphOutputs
  // copyTensorsInfo():260 calloc's a Qnn_Tensor_t[] and deepCopyQnnTensorInfo()'s each.
  qnn_wrapper_api::GraphInfo_t** graphsInfo{nullptr};
  uint32_t graphsCount{0};
  sample_app::copyMetadataToGraphsInfo(binaryInfo, graphsInfo, graphsCount);

  // The system context handle is no longer needed once metadata is copied out:
  qnnSystemInterface.systemContextFree(sysCtxHandle);                      // QnnSampleApp.cpp:488
  sysCtxHandle = nullptr;

  // ==========================================================================
  // (d) CREATE THE EXECUTABLE CONTEXT FROM THE BINARY (HTP deserialization)
  //     createFromBinary():506-513
  // ==========================================================================
  Qnn_ContextHandle_t context{nullptr};
  const QnnContext_Config_t** contextConfig{nullptr};
  qnnInterface.contextCreateFromBinary(
      backendHandle,
      deviceHandle,
      contextConfig,
      static_cast<void*>(buffer.data()),
      bufferSize,
      &context,                 // OUT
      /*Qnn_ProfileHandle_t*/ nullptr);                                    // QNN_SUCCESS == 0

  // ==========================================================================
  // (e) RETRIEVE EACH GRAPH HANDLE BY NAME INTO graphsInfo[g]->graph
  //     createFromBinary():528-540
  // ==========================================================================
  for (uint32_t g = 0; g < graphsCount; g++) {
    qnnInterface.graphRetrieve(
        context,
        (*graphsInfo)[g].graphName,        // name parsed from metadata in step (c)
        &((*graphsInfo)[g].graph));        // OUT: Qnn_GraphHandle_t
  }

  // ==========================================================================
  // (f) ALLOCATE CLIENT BUFFERS AND BIND THEM TO INPUT/OUTPUT TENSORS
  //     IOTensor.cpp setupTensors():362-422 / setupInputAndOutputTensors():425
  // ==========================================================================
  // For graph 0; repeat per graph. The metadata tensors live at
  // graphsInfo[0]->inputTensors / outputTensors and are used as "wrappers"
  // (templates) that you deep-copy and then attach a raw client buffer to.
  auto& gi = (*graphsInfo)[0];

  Qnn_Tensor_t* inputs  = (Qnn_Tensor_t*)calloc(gi.numInputTensors,  sizeof(Qnn_Tensor_t));
  Qnn_Tensor_t* outputs = (Qnn_Tensor_t*)calloc(gi.numOutputTensors, sizeof(Qnn_Tensor_t));

  auto setupOne = [](Qnn_Tensor_t* dst, const Qnn_Tensor_t* wrapper) {
    *dst = QNN_TENSOR_INIT;
    sample_app::deepCopyQnnTensorInfo(dst, wrapper);          // copies name/dims/rank/type/quant
    QNN_TENSOR_SET_MEM_TYPE(dst, QNN_TENSORMEMTYPE_RAW);      // raw (CPU) client buffer mode

    // Compute byte length from dims+dtype, malloc, and bind via SET_CLIENT_BUF.
    // QNN_TENSOR_SET_CLIENT_BUF expands to tensor.v1.clientBuf = clientBuf
    // (QnnTypeMacros.hpp:458); clientBuf is a Qnn_ClientBuffer_t { void* data; uint32_t dataSize; }
    std::vector<size_t> dims;
    for (uint32_t r = 0; r < QNN_TENSOR_GET_RANK(dst); ++r)
      dims.push_back(QNN_TENSOR_GET_DIMENSIONS(dst)[r]);
    size_t length = /* datautil::calculateLength(dims, QNN_TENSOR_GET_DATA_TYPE(dst)) */ 0;

    Qnn_ClientBuffer_t clientBuffer = QNN_CLIENT_BUFFER_INIT;
    clientBuffer.data     = malloc(length);    // YOU own this memory
    clientBuffer.dataSize = (uint32_t)length;
    QNN_TENSOR_SET_CLIENT_BUF(dst, clientBuffer);
  };
  for (uint32_t i = 0; i < gi.numInputTensors;  ++i) setupOne(&inputs[i],  &gi.inputTensors[i]);
  for (uint32_t o = 0; o < gi.numOutputTensors; ++o) setupOne(&outputs[o], &gi.outputTensors[o]);

  // ... fill inputs[i] client buffers with your data here ...
  // (IOTensor.cpp:244/353 write into QNN_TENSOR_GET_CLIENT_BUF(input).data)

  // ==========================================================================
  // (g) EXECUTE THE GRAPH
  //     executeGraphs():857-864
  // ==========================================================================
  qnnInterface.graphExecute(
      gi.graph,
      inputs,  gi.numInputTensors,
      outputs, gi.numOutputTensors,
      /*Qnn_ProfileHandle_t*/ nullptr,
      /*Qnn_SignalHandle_t*/  nullptr);                                    // QNN_GRAPH_NO_ERROR

  // ==========================================================================
  // (h) READ OUTPUTS — results are now in each output tensor's client buffer
  //     IOTensor.cpp writeOutputTensor():787 / convertToFloat():602
  // ==========================================================================
  for (uint32_t o = 0; o < gi.numOutputTensors; ++o) {
    void*    outData = QNN_TENSOR_GET_CLIENT_BUF(outputs[o]).data;        // tensor.v1.clientBuf.data
    uint32_t outSize = QNN_TENSOR_GET_CLIENT_BUF(outputs[o]).dataSize;
    // QNN_TENSOR_GET_DATA_TYPE(outputs[o]) tells you how to interpret outData;
    // if quantized (e.g. QNN_DATATYPE_UFIXED_POINT_8/16) dequantize using
    // QNN_TENSOR_GET_QUANT_PARAMS(&outputs[o]).scaleOffsetEncoding.{scale,offset}.
    (void)outData; (void)outSize;
  }

  // --- teardown (reverse order): free client bufs, contextFree, deviceFree,
  //     backendFree, logFree, dlclose. (terminateBackend():225 / freeContext():327)
  return 0;
}
```

---

## Key facts captured verbatim from the source

**The single function-pointer vtable.** After `QnnInterface_getProviders`, the entire C-API is reached through one struct member `QNN_INTERFACE_VER_TYPE qnnInterface = interfaceProviders[i]->QNN_INTERFACE_VER_NAME;` (DynamicLoadUtil.cpp:79). The methods used are: `qnnInterface.logCreate`, `.backendCreate`, `.deviceCreate`, `.contextCreateFromBinary`, `.graphRetrieve`, `.graphExecute`, `.contextFree`, `.deviceFree`, `.backendFree`, `.logFree`. The System reflection methods come from the parallel `qnnSystemInterface` vtable: `.systemContextCreate`, `.systemContextGetBinaryInfo`, `.systemContextFree`.

**Exact `systemContextGetBinaryInfo` signature** (QnnSampleApp.cpp:472): `(sysCtxHandle, void* buffer, uint64_t bufferSize, const QnnSystemContext_BinaryInfo_t** binaryInfo, Qnn_ContextBinarySize_t* binaryInfoSize)`.

**Exact `contextCreateFromBinary` signature** (QnnSampleApp.cpp:506): `(backendHandle, deviceHandle, const QnnContext_Config_t** config, void* buffer, uint64_t bufferSize, Qnn_ContextHandle_t* context, Qnn_ProfileHandle_t profile)`.

**Exact `graphRetrieve` signature** (QnnSampleApp.cpp:535): `(context, const char* graphName, Qnn_GraphHandle_t* graph)`.

**Exact `graphExecute` signature** (QnnSampleApp.cpp:857): `(graph, Qnn_Tensor_t* inputs, uint32_t numInputs, Qnn_Tensor_t* outputs, uint32_t numOutputs, Qnn_ProfileHandle_t profile, Qnn_SignalHandle_t signal)`.

**`binaryInfo` is a version-tagged union.** `copyMetadataToGraphsInfo` (QnnSampleAppUtils.cpp:391) switches on `binaryInfo->version` ∈ `{QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1, _2, _3}` and reads `binaryInfo->contextBinaryInfoV{1,2,3}.graphs` / `.numGraphs`. Per-graph it switches on `QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_{1,3}` → `.graphInfoV1`/`.graphInfoV3` with fields `graphName`, `graphInputs`/`numGraphInputs`, `graphOutputs`/`numGraphOutputs` (all `Qnn_Tensor_t*`). For the S25/Hexagon V79 + QAIRT 2.45, expect **V3** for both. You must handle the version switch, not assume V1.

**Client buffer binding** (IOTensor.cpp:396-408): create a `Qnn_ClientBuffer_t clientBuffer = QNN_CLIENT_BUFFER_INIT;`, set `clientBuffer.data` (your malloc'd pointer) and `clientBuffer.dataSize` (byte length), set mem type with `QNN_TENSOR_SET_MEM_TYPE(t, QNN_TENSORMEMTYPE_RAW)`, then bind with `QNN_TENSOR_SET_CLIENT_BUF(t, clientBuffer)`. The setter macro expands to `tensor.v1.clientBuf = clientBuf` (QnnTypeMacros.hpp:458) — i.e. the underlying struct field is `Qnn_Tensor_t.v1.clientBuf`.

**Reading outputs** (IOTensor.cpp:787): the result bytes are at `QNN_TENSOR_GET_CLIENT_BUF(output).data` (== `tensor.v1.clientBuf.data`, QnnTypeMacros.hpp:340), with the same buffer you bound pre-execute being filled in place. Dequantize via `QNN_TENSOR_GET_QUANT_PARAMS(tensor).scaleOffsetEncoding.{scale,offset}` (IOTensor.cpp:603-604) when the dtype is fixed-point.

**Orchestration order** (main.cpp:462-511, retrieve-from-binary path): `initialize` (logCreate) → `initializeBackend` (backendCreate) → `isDevicePropertySupported` → `createDevice` → `registerOpPackages` → `createFromBinary` (systemContextCreate → systemContextGetBinaryInfo → copyMetadataToGraphsInfo → systemContextFree → contextCreateFromBinary → graphRetrieve) → (`finalizeGraphs` only if `isFinalizeDeserializedGraphSupported`) → `executeGraphs` → `freeContext` → `freeDevice` → `terminateBackend` → `dlclose`.

**NetRun caveat:** the `examples/QNN/NetRun/` folder ships no source — only `*-qnn-net-run.sh` wrappers around the prebuilt binary. The C-API source of record for this exact flow is `SampleApp`, whose `--retrieve_context <model.bin> --system_library libQnnSystem.so` path is functionally identical to what `qnn-net-run --retrieve_context` does internally.


# ================= ndk_build =================

All confirmed. I have everything needed. One blocking gap: **no Android NDK is installed**, so the toolchain path must be obtained first. The standard `sdkmanager` install produces `~/Library/Android/sdk/ndk/<version>/`. I'll provide the exact answer with the install step and the macOS host tag (`darwin-x86_64`), since the prebuilt clang lives under that.

---

## NDK situation (blocking prerequisite)

**No Android NDK is installed on this machine.** Verified exhaustively:
- `/Users/sumakatabattuni/Library/Android/sdk/ndk` — **does not exist** (no `ndk/` or `ndk-bundle/`).
- No NDK in the executorch repo, the Gradle cache, env vars (`ANDROID_NDK_ROOT`/`HOME`/`NDK_ROOT` all unset), or `/opt`/`/usr/local`.
- The SDK has `platform-tools` (adb at `/Users/sumakatabattuni/Library/Android/sdk/platform-tools/adb`), `build-tools`, `platforms`, `emulator` — but **no `ndk/` and no `cmake/`**.

You must install one first. The SDK root is `/Users/sumakatabattuni/Library/Android/sdk` (from `android/local.properties`: `sdk.dir=...`).

```bash
# Install a recent LTS NDK (r26d) via sdkmanager:
/Users/sumakatabattuni/Library/Android/sdk/cmdline-tools/latest/bin/sdkmanager "ndk;26.3.11579264"
# Result: /Users/sumakatabattuni/Library/Android/sdk/ndk/26.3.11579264
```
(cmdline-tools may also need installing: `sdkmanager "cmake;3.22.1" "ndk;26.3.11579264"`.) Any r25–r27 works; QNN 2.45 targets API 21+ and these are all fine. Below I use `NDK=/Users/sumakatabattuni/Library/Android/sdk/ndk/26.3.11579264`. **Host tag is `darwin-x86_64`** on this Mac (the SDK's bundled run scripts say `linux-x86_64` because they're authored for Linux build hosts — substitute `darwin-x86_64` here; it also covers Apple Silicon via Rosetta-free universal binaries).

---

## (1) Cross-compile invocation — arm64-v8a, minSdk 30

Define:
```bash
QNN=/private/tmp/claude-501/-Users-sumakatabattuni-Documents-Personal-slashh-edge/0573850d-791b-4a19-ab79-95d1c47bf752/scratchpad/qairt245/qairt/2.45.0.260326
NDK=/Users/sumakatabattuni/Library/Android/sdk/ndk/26.3.11579264
TC=$NDK/toolchains/llvm/prebuilt/darwin-x86_64
```

**Direct clang++ (matches the SDK's own flags — c++_static, hidden visibility, `-DQNN_API=...`):**
```bash
$TC/bin/aarch64-linux-android30-clang++ \
  -std=c++17 -O3 -fPIC -fvisibility=hidden \
  -DQNN_API='__attribute__((visibility("default")))' \
  -I"$QNN/include/QNN" \
  -static-libstdc++ \
  myrunner.cpp -o qnn_runner \
  -ldl -lm -llog
```
Notes:
- `aarch64-linux-android30-clang++` is the API-30 wrapper; it sets `--target=aarch64-linux-android30` and the sysroot automatically. (Equivalent: `clang++ --target=aarch64-linux-android30 --sysroot=$TC/sysroot ...`.)
- `-static-libstdc++` bundles libc++ so you don't have to ship `libc++_shared.so` (the SDK's run script otherwise pushes `libc++_shared.so` from `$TC/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so`). If you instead use `-stdlib=libc++` shared, push that file alongside the binary.
- `-llog` only if you call `__android_log_*`; `-lm` optional.

**CMake equivalent:**
```bash
cmake -S . -B build-android \
  -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-30 \
  -DANDROID_STL=c++_static
cmake --build build-android -j
```
In `CMakeLists.txt`:
```cmake
target_include_directories(qnn_runner PRIVATE
  /private/tmp/claude-501/-Users-sumakatabattuni-Documents-Personal-slashh-edge/0573850d-791b-4a19-ab79-95d1c47bf752/scratchpad/qairt245/qairt/2.45.0.260326/include/QNN)
target_link_libraries(qnn_runner PRIVATE ${CMAKE_DL_LIBS})   # i.e. -ldl
```

---

## (2) dlopen vs link-time — **runtime dlopen, NOT link-time**

**Confirmed from three independent sources:**

1. **SampleApp `make/Android.mk`** (the canonical executable build) links **zero QNN libraries** — `LOCAL_LDLIBS := -lGLESv2 -lEGL` only, plus `Application.mk`'s `APP_LDFLAGS := -lc -lm -ldl`. The Linux `Makefile.linux-x86_64` likewise links only `LIBS=-ldl`.
2. **Source code** loads the backend at runtime: `src/Utils/DynamicLoadUtil.cpp:55` does `resolveSymbol<...>(libBackendHandle, "QnnInterface_getProviders")`, and `src/PAL/src/linux/DynamicLoading.cpp:39` is a thin wrapper over `::dlopen`. The backend `.so` path is a runtime CLI arg (`--backend libQnnHtp.so`).
3. **The prebuilt `bin/aarch64-android/qnn-net-run`** has `NEEDED` entries of only `libc.so libm.so libdl.so liblog.so` — no `libQnn*` at all.

**What to pass the linker:** only `-ldl` (and `-lm`/`-llog` if used). **Do NOT** add `-L$QNN/lib/aarch64-android -lQnnHtp` etc. — you `dlopen("libQnnHtp.so", RTLD_NOW)`, then `dlsym`/`QnnInterface_getProviders`, exactly as the sample does. The libs in `lib/aarch64-android` are compile-against-headers-only; you ship them and load them on-device.

---

## (3) Run on-device from `/data/local/tmp` (S25 Ultra, SM8750, Hexagon **V79**)

The S25 Ultra HTP is **V79**, so the stub is `libQnnHtpV79Stub.so` and the skel is `lib/hexagon-v79/unsigned/libQnnHtpV79Skel.so` (both verified present). Push everything into one rig dir and point both env vars at it (the SDK's V79 branch sets `LD_LIBRARY_PATH` **and** `ADSP_LIBRARY_PATH` to `${TARGET_ROOT}`):

```bash
QNN=/private/tmp/claude-501/-Users-sumakatabattuni-Documents-Personal-slashh-edge/0573850d-791b-4a19-ab79-95d1c47bf752/scratchpad/qairt245/qairt/2.45.0.260326
ADB=/Users/sumakatabattuni/Library/Android/sdk/platform-tools/adb
RIG=/data/local/tmp/scamshield

$ADB shell "mkdir -p $RIG"

# your executable
$ADB push qnn_runner $RIG

# QNN runtime libs (CPU-side) — HTP core + V79 stub + prepare + system
$ADB push $QNN/lib/aarch64-android/libQnnHtp.so          $RIG
$ADB push $QNN/lib/aarch64-android/libQnnHtpPrepare.so   $RIG
$ADB push $QNN/lib/aarch64-android/libQnnHtpV79Stub.so   $RIG
$ADB push $QNN/lib/aarch64-android/libQnnSystem.so       $RIG

# Hexagon DSP skel for V79 (runs on the NPU, found via ADSP_LIBRARY_PATH)
$ADB push $QNN/lib/hexagon-v79/unsigned/libQnnHtpV79Skel.so $RIG

# only if you linked libc++ shared instead of -static-libstdc++:
# NDK=/Users/sumakatabattuni/Library/Android/sdk/ndk/26.3.11579264
# $ADB push $NDK/toolchains/llvm/prebuilt/darwin-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so $RIG

# plus your .pte / context-binary / inputs as needed
```

**Launch (V79 pattern — both paths point at the rig dir):**
```bash
$ADB shell "cd $RIG && \
  export LD_LIBRARY_PATH=$RIG && \
  export ADSP_LIBRARY_PATH=$RIG && \
  chmod +x ./qnn_runner && \
  ./qnn_runner --backend libQnnHtp.so <your-args>"
```

Why these two env vars:
- `LD_LIBRARY_PATH=$RIG` — Android's CPU-side `dlopen` resolves `libQnnHtp.so`, `libQnnHtpV79Stub.so`, `libQnnSystem.so` (and `libQnnHtpPrepare.so` when building/preparing a graph) from the rig dir.
- `ADSP_LIBRARY_PATH=$RIG` — the FastRPC/cDSP loader on the Hexagon side resolves the **skel** (`libQnnHtpV79Skel.so`) from here. The SDK's V79 branch (lines 168–171) sets it to exactly `${TARGET_ROOT}`. If FastRPC can't find the skel you'll get a `-1` / `0x...` domain error on graph finalize; if so, fall back to the broader search list the V68/V69/V75 branches use:
  `ADSP_LIBRARY_PATH="$RIG;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp"` and `LD_LIBRARY_PATH="$RIG:/vendor/dsp/cdsp:/vendor/lib64/:$LD_LIBRARY_PATH"`.

---

### Caveats / project-specific notes
- **macOS host tag**: use `darwin-x86_64` in the toolchain path; the SDK scripts hardcode `linux-x86_64` because they assume a Linux build host.
- **Retail-SELinux reality** (per project memory `retail-selinux-npu-path`): on a retail S25 the *in-app* QNN delegate can't reach the cDSP, but this **out-of-process `/data/local/tmp` runner with `ADSP_LIBRARY_PATH` set is exactly the proven path** that does reach the NPU. This binary belongs to that `npu_helper.sh` runner pattern.
- **Min API 30**: I used `aarch64-linux-android30-clang++`. The SDK samples ship `APP_PLATFORM := android-21`; 30 is fine and matches your "min API ~30" ask (must be ≤ the device's API, S25 is 34/35).
- **Files confirmed to exist**: headers `/private/tmp/.../2.45.0.260326/include/QNN/` (QnnInterface.h + HTP/ subdir); libs `/private/tmp/.../lib/aarch64-android/{libQnnHtp,libQnnHtpPrepare,libQnnHtpV79Stub,libQnnSystem}.so`; skel `/private/tmp/.../lib/hexagon-v79/unsigned/libQnnHtpV79Skel.so`; prebuilt reference binary `/private/tmp/.../bin/aarch64-android/qnn-net-run`. SDK build id `v2.45.0.260326154327`.
