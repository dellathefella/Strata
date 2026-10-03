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

1. Spike: cmake seam + doorbell spin microbench on L0 — **DONE 2026-10-03**
   (results below).
2. Device/pinned/graph shim on L0.
3. Decode kernel family + parity — **Q4/Q8 first** (see kernel priority below).
4. Doorbell overlap single-card.
5. Prefill path (oneMKL + ggml-sycl MMQ).
6. Verify window (MTP).
7. Pipeline stages (Strata layer-split) on SYCL.
8. **TP groups (2-dev row-split + host-bounce all-reduce), placement
   optimizer, 3-card lagrange config = the 1+N target.**

## Milestone-1 results (tools/l0_spin_bench, B60 dev1 x8, 2026-10-03)

| Test | Result | Verdict |
|---|---|---|
| A host→device doorbell release→exit | 379.5 µs | PASS — system-scope atomic_ref on host-USM works |
| B device→host store visibility | 1.04 ms | PASS — fine for mid-graph signaling |
| C ~3 s resident spin kernel | survived, no GT reset | PASS — miss-phase waits viable |
| D same-device GPU‖GPU (spin q2 + compute q) | **DEADLOCK** (150 s timeout) | xe/L0 does not timeslice these contexts |

**Design constraint from D**: never rely on concurrent kernels on one device;
phase order is in-stream. Strata's overlap is CPU‖GPU (CPU pool computes miss
experts while GPU runs hits + spins) — unaffected. Multi-device concurrency
(3-card lagrange: spin on dev0, compute on dev1/2) presumed OK, to verify in
milestone 8. Doorbell latency budget: A+B ≈ 1.4 ms round trip per layer
signal — negligible vs ~25-100 ms CPU miss-phase.

## Milestone-3 spike results — Q4/Q8 kernels (tools/q4q8_kernel_bench, B60 dev1, 2026-10-03)

Device capability query (tools/mx_caps): int8 DPAS combos = **M8 x N16 x K32,
s8/u8 x s8/u8 -> s32** (fp16/bf16 = 16x16x16). ggml-sycl uses NEITHER DPAS nor
a real dp4a (dpct::dp4a is unpack+scalar-mul emulation) — confirmed headroom.

| Test | Time | Throughput | Parity |
|---|---|---|---|
| T1 Q8_0 GEMV float-dequant (N=K=4096) | 0.478 ms | 37.3 GB/s | 1.1e-6 |
| T2 Q8_0 GEMV packed int8 dot | 0.175 ms | 102.1 GB/s | 1.2e-7 |
| T3 Q4_0 GEMV float-dequant | 0.186 ms | 50.8 GB/s | 1.4e-6 |
| T4 joint_matrix int8 DPAS microtest | — | PASS | exact |
| T5 DPAS int8 GEMM 1024^3 (untiled v1) | 0.099 ms | **21.6 TOPS** | exact (128 samples) |

Lessons:
- int8 dot path = 2.7x the float-dequant path on GEMV; decode kernels must be int.
- GEMV at 102 GB/s is occupancy-bound (4096 WIs, serial 128-block loop):
  K-split + reduce should approach GDDR6 peak (~3-4x headroom).
- DPAS GEMM untiled, global-mem loads, 16 WIs/tile: 21.6 TOPS with ~10x
  theoretical headroom — local-memory tiling + multi-tile workgroups next.
- GOTCHAS: first kernel launch pays SPIR-V JIT (~350 ms) — always warm up
  before timing; joint_matrix needs `sycl::detail::dynamic_address_cast`
  USM->multi_ptr (gmp helper); int8 accumulator 16x16 is NOT supported
  (query `info::device::matrix_combinations` per device).
- A same-device GPU fault required a host reboot mid-spike; all 3 B60s
  re-enumerated clean afterward (xe minor 0/1/2). Treat long bench loops as
  reboot-safe workloads.

## Kernel port order — Q4/Q8 focus (design directive 2026-10-03)

Xe2/BMG DPAS natively accelerates INT8/INT4 dot products, so Q4_0/Q8_0/IQ4_NL
matmuls can ride the matrix hardware; IQ2/IQ3/K-quant superblocks are
dequant+FP-only. Port priority:

1. Q4_0/Q8_0 dequant + gemv families (`s_gemv`, `iq4nl_s_gemv`, `bf16_gemv`)
2. **DPAS INT8/INT4 matmul kernels** for Q4_0xQ8_0 / Q8_0xQ8_0 (the hardware win;
   `sycl::ext::oneapi::experimental::joint_matrix` or ESIMD dpas)
3. KV core (`s_kv_attn`, `s_kv_update`)
4. IQ2/IQ3/K-quant superblock kernels LAST

Model focus on Arc: Flash-Next IQ4_NL (38.9 GB, fits the dev1+dev2 TP pair
resident) and Q8_0 (70.8 GB, hybrid/tiered). IQ3_S stays only as the current
stock-endpoint baseline.

## Deployed stock baseline (lagrange, 2026-10-03)

Endpoint `qwen38-flashnext-sycl` (port 8183, dev1 `level_zero:1`): Flash-Next
IQ3_S hybrid `-ngl 99 --cpu-moe -c 32768 -ub 8192` + mtp-Q4_0 draft —
prefill 293.1 t/s @16k, decode 12.4 t/s, 45 s warm start. NOTE:
`--load-mode none` wedges at -c >= 32768 (xe ioctl, clean dmesg, survives
reboot) — mmap mode is the healthy path; upstream-bug candidate.

## Prompt cache in the big system-RAM tier (design directive 2026-10-03)

Memory hierarchy for KV / prompt-cache regions, top to bottom:
1. **VRAM**: the active slot's KV only (per pipeline stage; TP pairs
   row-split it). A 128k Flash-Next session is ~3.8 GB at ~29.5 KiB/position.
2. **System RAM — the prompt cache** (corrected directive: the clever tier
   is the huge host RAM, 251 GB on lagrange vs halogen's 128 GB unified):
   every admitted session's regions live here whole; resume = one
   sequential RAM->VRAM copy per stage (6.8 GB/s on x8 links: a 128k
   session restores in ~0.6 s). Capacity: ~60 full 128k sessions.
3. **NVMe**: cold overflow only; rarely touched at this RAM size.

Eviction: per-region LRU within a stage; a region demotes VRAM->RAM->NVMe
whole (rows are contiguous per region, so demotion = one sequential copy).
Admission mirrors expert-cache accounting: reserve prompt+budget at admit.
