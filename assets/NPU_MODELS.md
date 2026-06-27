# NPU Model Set — pull & run on the Snapdragon Hexagon NPU

These model artifacts are stored in **Git LFS** (they are 300 MB–614 MB each).
A plain `git clone` only fetches tiny pointer files; you must pull the real
binaries with `git lfs`.

## 1. Fetch the models

```bash
# one-time, if you don't have git-lfs:
brew install git-lfs        # macOS   (apt-get install git-lfs on Linux)
git lfs install

# get the repo + the large model blobs
git clone https://github.com/Naveen-Sai-Ganadi/slashh-executorch.git
cd slashh-executorch
git checkout ng/dev
git lfs pull                # downloads the real .pte / .bin (~2.1 GB)
```

If you already cloned before `git lfs install`, just run `git lfs pull` in the repo.

Verify you got real files (not 130-byte pointers):

```bash
ls -lh assets/teacher_wavlm_broad_qnn.pte    # should be ~614M, not ~130 bytes
```

## 2. The files

| file | size | precision | role |
|---|---|---|---|
| `teacher_wavlm_broad_qnn.pte`            | 614M | fp16 | **primary** — full WavLM teacher .pte, QNN-delegated for HTP |
| `forward_2.bin`                          | 614M | fp16 | raw QNN context binary (WavLM encoder + head) extracted from the .pte — used for the `qnn-net-run` shell path |
| `teacher_wavlm_broad_npu_int8_qnn.bin`   | 311M | int8 | INT8 NPU context binary |
| `teacher_wavlm_broad_int8_relpos_excl.pte`| 302M | int8 | INT8 .pte (rel-pos excluded from quant) |
| `teacher_wavlm_broad_int8_pt.pte`        | 302M | int8 | INT8 .pte (PT2E) |

The small `forward_0.bin` (44K) / `forward_1.bin` (48K) rel-pos graphs are in
plain git (not LFS) and ship with a normal checkout.

Input: raw waveform `[1, 48000]` (3 s @ 16 kHz) → stress logit `[1, 1]`.

## 3. Device requirements (verified set)

- Samsung Galaxy S25 Ultra `SM-S938U`, SoC `SM8750` (Snapdragon 8 Elite), **Hexagon V79 HTP**
- QAIRT / QNN **2.37.0.250724** runtime libs (`libQnnHtp.so`, `libQnnHtpV79Skel.so`, …)
- The `.pte` was built for `QcomChipset.SM8750` / QNN 2.37 — a different SoC or QNN
  version requires a re-export.

## 4. How to run on the NPU

> **Important (proven 2026-06-27):** on a **retail, non-rooted** S25 the *in-app*
> QNN delegate is blocked by SELinux (the `untrusted_app` domain is denied
> `/dev/fastrpc-cdsp`). The proven on-NPU path is **`qnn-net-run` from the adb
> `shell` user (uid 2000), which does have cDSP access.** See
> `docs/v2/2026-06-27-wavlm-s25-npu-proof.md` for the full evidence.

Shell path (runs the WavLM encoder on the Hexagon V79 cDSP):

```bash
# push the context binary + QNN V79 libs to a shell-writable dir
adb push assets/forward_2.bin /data/local/tmp/qnntest/
# also push qnn-net-run + libQnnHtp.so + libQnnHtpV79*.so (QAIRT 2.37 aarch64-android)

adb shell '
  cd /data/local/tmp/qnntest
  export LD_LIBRARY_PATH=/data/local/tmp/qnntest
  export ADSP_LIBRARY_PATH=/data/local/tmp/qnntest
  ./qnn-net-run --backend libQnnHtp.so --retrieve_context forward_2.bin \
    --input_list inlist2f.txt --output_dir out2f --log_level info
'
```

Confirm it ran on the NPU (not CPU) by grepping logcat / stdout for
`_dom=cdsp ... on domain 3` and `Finished Executing Graphs`.

Expected faithful logit ≈ **-2.465** (prob 0.078), within |Δ| ≈ 0.09 of the host
eager fp32 reference — see the proof doc.

## 5. Re-export (different SoC / QNN version)

The `.pte` files target SM8750 + QNN 2.37 specifically. For another device,
re-run the QNN export (see `scratchpad/export_wavlm_qnn_pte.py` and
`scratchpad/build_wavlm_qnn_pte.sh`) against the matching QAIRT version.
