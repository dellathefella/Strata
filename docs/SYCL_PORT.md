# SYCL / Intel Arc port — engineering plan

Target hardware: halo1 — Intel Arc Pro B60 (BMG-G21 x2, Xe2, 24 GB) in a
**Thunderbolt 3 dock** (JHL7440, PCIe x4 Gen3 tunnel), AMD Strix Halo host
(16c Zen5, 125 GB RAM). Model: Qwen3.8-Flash-Next (qwen4exp), ISTA-DASLab
GSQ-RCO GGUFs.

## Measured ceilings (2026-10-02, halo1)

| Quantity | Value | Method |
|---|---|---|
| H2D pinned, 1..8 streams | 2.8-2.9 GB/s | sycl memcpy, 1-2 GiB, multi-queue |
| D2H pinned | 3.2 GB/s | same |
| Bidirectional aggregate | ~2.8 GB/s total (NOT full duplex) | concurrent H2D+D2H |
| Zero-copy kernel read of host USM | 2.6 GB/s | parallel_for over host ptr |
| Link state | IOMMU off (amd_iommu=off), ASPM disabled all hops, pinned mem | setpci/sysfs |
| TB3 theoretical | 3.94 GB/s (x4 Gen3) | JHL7440 spec |

Conclusion: **~3 GB/s aggregate is the hard ceiling**; multi-stream, MRRS,
ASPM/IOMMU tuning move nothing. 80% of Gen3-x4 theoretical is normal TB3
efficiency. Do not design around streaming experts per chunk
(70 GB/chunk / 3 GB/s = ~450 t/s ceiling, and bidir sharing makes it worse).

Reference points on this machine (llama.cpp master 4ebdf2c, SYCL):
- Flash-Next IQ3_S hybrid `-ngl 99 --cpu-moe`: decode 19.22 t/s raw (MAD 0.08),
  26.93 t/s with separate mtp-Q4_0 draft (`--spec-type draft-mtp`),
  prefill 16k = **29 t/s** (CPU expert GEMMs are the bottleneck).
- Clocks locked (min_freq=max_freq via xe freq0 sysfs) for all measurements.

## Architecture for max prefill under TB3

Keep expert bytes OFF the link:

1. **Expert tiering** (Strata `ExpertCache`): hot experts VRAM-resident
   (~10-12 GB of the 24 GB after trunk+KV), profile-driven admission
   (`profile.bin`, hit rate ~0.64), adaptive heat swaps.
2. **Doorbell overlap**: captured layer graph publishes router ids + miss
   list to mapped pinned memory mid-graph; CPU pool (AVX-512 kernels) computes
   miss experts from RAM **while** GPU runs the hit phase; combine adds.
   Cold experts never cross PCIe.
3. **Link duties**: initial trunk load, background tier refills (zero-copy
   reads fine at 2.6 GB/s off the critical path), activations only.
   Never schedule H2D and D2H concurrently (aggregate-shared tunnel).
4. Expected prefill: balanced split puts CPU cold fraction in GPU's shadow:
   GPU ~21 TFLOPS eff int4 MMQ vs CPU ~2.8 TFLOPS eff IQ3 dot => ~400-600 t/s
   (vs 29 t/s llama.cpp cpu-moe today). Decode rises too (less CPU dependence).

## Port strategy (Option A from portability assessment)

Single CUDA-shaped source set; follow the HIP precedent (270-LOC shim +
CMake language flip). For Intel: icpx/-fsycl (oneAPI 2026.1) with a
`sycl_compat` layer; SYCLomatic (dpct) for bulk kernel translation, hand-fix
the load-bearing ones. Bitwise parity suite in tests/ is the acceptance gate.

Risks (from assessment):
- CUDA Graph capture/replay is load-bearing (decode loop, doorbell, verify
  window). Map to `sycl::ext::oneapi::experimental::command_graph` or queue
  replay; re-measure the "poll driver, not memory" lesson on L0/Linux.
- Device-side spin on mapped pinned memory (doorbell) => L0 host USM; verify
  spin behavior early (milestone 1 spike).
- PTX tensor-core paths have portable fp32-FMA fallbacks (used on AMD) —
  start there; XMX via oneAPI matrix ext later.
- cuBLAS prefill GEMM -> oneMKL; MMQ expert prefill -> ggml-sycl mmq or the
  fused-int8 kernels.

## Milestones

1. **Spike**: `cmake/sycl_backend.cmake` + `include/strata/sycl_compat/`;
   one hand-written SYCL kernel (dequant or elementwise) running on B60 with
   parity vs CUDA reference; doorbell spin-on-host-USM microbench on L0.
2. Device/pinned/graph shim: DeviceArena, PinnedArena, GraphRegistry on L0
   (USM + command graphs); h2d/zero-copy benches above as regression tests.
3. Decode path: MMVQ family + router + hit-phase grouped experts on SYCL;
   parity-gated per kernel family.
4. Doorbell CPU/GPU overlap working end-to-end (single card, TB3): prefill
   and decode benchmarks vs the llama.cpp hybrid numbers above.
5. Prefill path: oneMKL dense GEMM + ggml-sycl MMQ experts; 8k chunking.
6. Verify window (MTP spec) on SYCL; bitwise-identical greedy guarantee.
7. Multi-B60: layer-split pipeline (docs/MULTI_GPU.md) + remote expert tiers.

## Non-goals

- Streaming full expert set per prefill chunk over TB3 (physics, see table).
- Windows/WDDM doorbell behavior (Linux-only target).
