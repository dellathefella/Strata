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

## 1+N GPU topology-aware parallelism (design directive 2026-10-03)

Strata ships pipeline layer-split + expert-only helper tiers. The Arc port
adds **intra-stage tensor parallelism** so the engine scales on 1+N cards:

- **1 GPU**: tiering + doorbell overlap (above). No comms.
- **N GPUs**: partition devices into pipeline stages; any stage holding >=2
  devices runs them **tensor-parallel (row-split)**: each GEMM shard computes
  a column/row slice, one all-reduce of the output per op.
- **Placement optimizer** at startup: measure per-link H2D/D2H (the h2d bench
  in this doc) + device VRAM; form TP groups from the fastest links (all-reduce
  volume = 2 x tokens x hidden x 2B per layer op — needs GB/s-class links),
  assign leftover/slow devices as pipeline stages (activation crossing =
  tokens x hidden x 2B once per stage boundary per ubatch — tolerable even
  on x4 Gen3) or as expert-only tiers (remote_experts).
- **The 3-card rule** (user directive, lagrange: dev1+dev2 x8 Gen3 6.8 GB/s,
  dev0 x4 Gen3 2.5 GB/s): TP pair = {dev1, dev2}, pipeline stage = {dev0}.
  Generalizes: odd card counts peel the slowest link into a pipeline stage
  or expert tier; the remainder forms TP pairs.

Budget check (lagrange, ub8192 prefill): TP all-reduce = 42 MB x 2 x 48 layers
= ~4 GB/ubatch over 6.8 GB/s = 0.6 s vs ~4 s compute = 15% overhead — pays for
2x compute. Pipeline boundary on x4: 84 MB/ubatch over 2.5 GB/s = 34 ms —
free. Decode: TP splits weight streaming (54.8 GB / 2 cards in parallel);
pipeline stage adds its share serially — placement puts the smallest layer
range on the slowest card.

Contrast with llama.cpp `-sm row/layer` (measured 2026-10-03, RESULTS.md):
layer = pipeline but activations cross at EVERY layer boundary across the
slow link (comms-bound prefill, 150-216 t/s); row = TP across ALL cards
including the x4 (crashes + comms-bound). The 1+N scheme is neither: TP
inside fast islands, pipeline between islands.

## Milestones (updated)

1. Spike: cmake seam + doorbell spin microbench on L0 (this session).
2. Device/pinned/graph shim on L0.
3. Decode kernel family + parity.
4. Doorbell overlap single-card.
5. Prefill path (oneMKL + ggml-sycl MMQ).
6. Verify window (MTP).
7. Pipeline stages (Strata layer-split) on SYCL.
8. **TP groups (2-dev row-split + host-bounce all-reduce), placement
   optimizer, 3-card lagrange config = the 1+N target.**

## VRAM-resident prompt cache tier (design directive 2026-10-03)

Memory hierarchy for KV / prompt-cache regions, top to bottom:
1. **VRAM pool, per pipeline stage**: each stage keeps the KV rows for its
   layer range of every admitted session, in its own spare VRAM (hybrid
   configs leave ~15 GB/card idle; a 128k Flash-Next session is ~3.8 GB of
   KV at ~29.5 KiB/position). Resume = zero transfer, local to the stage —
   works even on slow-link cards. TP pairs row-split the pool.
2. **Host RAM**: LRU-evicted regions (halogen's current pool location).
3. **NVMe**: on-disk cache for cold sessions.

Eviction: per-region LRU within a stage; a region demotes VRAM->RAM->NVMe
whole (rows are contiguous per region, so demotion is one sequential copy).
Admission mirrors expert-cache accounting: reserve prompt+budget at admit.

Stock-llama.cpp approximation (deployed on lagrange 2026-10-03):
`--parallel 4 -c 131072` keeps four sessions' KV in the unified VRAM pool;
slot prefix-resume gives halogen-style instant follow-ups without any port
work. Measured below.
