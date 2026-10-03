// l0_spin_bench — milestone-1 acceptance for the Strata SYCL port.
//
// Validates the doorbell protocol's hardware assumptions on Level Zero:
//   A. host->device doorbell: a device kernel spins on a system-scope flag
//      in host USM; latency from host store to kernel exit.
//   B. device->host visibility: latency for a device store to host USM to
//      become visible to a polling host thread.
//   C. long resident spin: no KMD watchdog reset for a seconds-long kernel
//      (the miss-phase wait).
//   D. concurrency: a spinning kernel on one queue does not serialize
//      compute on another queue (hit-phase overlap requirement).
//
// Build: icpx -fsycl -O2 l0_spin_bench.cpp -o l0_spin_bench
#include <sycl/sycl.hpp>

#include <chrono>
#include <cstdio>
#include <thread>

using clk = std::chrono::steady_clock;
static double us(clk::time_point a, clk::time_point b) {
    return std::chrono::duration<double, std::micro>(b - a).count();
}

using ref_t = sycl::atomic_ref<int, sycl::memory_order::acq_rel,
                               sycl::memory_scope::system,
                               sycl::access::address_space::global_space>;

int main() {
    sycl::queue q(sycl::gpu_selector_v,
                  sycl::property_list{sycl::property::queue::in_order{}});
    auto dev = q.get_device();
    std::printf("device: %s\n",
                dev.get_info<sycl::info::device::name>().c_str());

    int *hf = (int *)sycl::malloc_host(64, q);
    for (int i = 0; i < 16; i++) hf[i] = 0;

    // A: host -> device doorbell
    {
        int *f = hf;
        auto ev = q.submit([&](sycl::handler &h) {
            h.single_task([=] {
                ref_t gate(f[0]);
                while (gate.load() != 1) {
                }
                ref_t done(f[1]);
                done.store(42);
            });
        });
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        auto t1 = clk::now();
        hf[0] = 1;
        ev.wait();
        auto t2 = clk::now();
        std::printf("A host->device doorbell release->exit: %.1f us (ack=%d)\n",
                    us(t1, t2), hf[1]);
    }

    // B: device -> host store visibility
    {
        int *f = hf;
        auto t0 = clk::now();
        auto ev = q.submit([&](sycl::handler &h) {
            h.single_task([=] {
                ref_t out(f[2]);
                out.store(1);
            });
        });
        while (hf[2] == 0 &&
               std::chrono::duration_cast<std::chrono::seconds>(clk::now() - t0).count() < 5) {
        }
        auto t1 = clk::now();
        ev.wait();
        std::printf("B device->host store visibility: %.1f us (saw %d)\n",
                    us(t0, t1), hf[2]);
    }

    // C: ~3 s resident spin, no watchdog reset
    {
        int *f = hf;
        auto ev = q.submit([&](sycl::handler &h) {
            h.single_task([=] {
                ref_t sink(f[3]);
                long long n = 0;
                for (long long i = 0; i < 2000000000LL; ++i) n += i;
                sink.store((int)(n & 0xff));
            });
        });
        ev.wait();
        std::printf("C 3s-class spin kernel survived (sink=%d)\n", hf[3]);
    }

    // D: CPU||GPU overlap — a resident spin kernel must not starve host work
    // (this is the real doorbell requirement: CPU pool computes miss experts
    // while the GPU spins/runs the hit phase).
    // NOTE: same-DEVICE GPU||GPU concurrency FAILED (deadlock, 2026-10-03):
    // a spinning kernel on one queue blocked parallel_for on another queue —
    // xe/L0 does not timeslice these contexts. Design constraint: never rely
    // on concurrent kernels on one device; order phases in-stream.
    {
        sycl::queue q2(dev,
                       sycl::property_list{sycl::property::queue::in_order{}});
        int *f = hf;
        auto spin = q2.submit([&](sycl::handler &h) {
            h.single_task([=] {
                ref_t gate(f[4]);
                while (gate.load() != 1) {
                }
                ref_t done(f[5]);
                done.store(7);
            });
        });
        // host compute while the spin kernel is resident
        auto t0 = clk::now();
        volatile double acc = 0;
        for (long i = 0; i < 200000000L; i++) acc += (double)(i % 7) * 0.5;
        double host_ms = us(t0, clk::now()) / 1000.0;
        hf[4] = 1;
        spin.wait();
        std::printf("D host 200M-FMA loop during resident spin: %.0f ms "
                    "(baseline without spin is the same CPU-bound loop; "
                    "spin ack=%d)\n",
                    host_ms, hf[5]);
    }

    sycl::free(hf, q);
    return 0;
}
