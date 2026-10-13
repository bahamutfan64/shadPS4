// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <array>
#include <chrono>
#include "common/logging/log.h"
#include "common/types.h"

// DIAGNOSTIC: reports every game frame that took long to arrive, with what the GPU thread spent
// its time on, to find the cause of stutter. Counters are reset at each reported or normal frame.
namespace Common::FrameProbe {

struct Counter {
    std::atomic<u32> count{};
    std::atomic<u64> micros{};
    std::atomic<u64> bytes{};
};

inline Counter shader_compile;
inline Counter pipeline_compile;
inline Counter texture_refresh;
inline Counter make_resident;
inline std::atomic<u32> page_faults{};
// Accumulated over a second (not reset per frame).
inline Counter sec_dma_sync;    // per-draw resync of DMA memory; bytes = resident ranges visited
inline Counter sec_dma_skipped; // resyncs skipped by the experimental switch
inline Counter sec_dma_upload;  // CPU-modified data re-uploaded; bytes = bytes
inline Counter sec_write_fault; // guest write faults handled (any thread)

class Scope {
public:
    explicit Scope(Counter& counter_, u64 bytes_ = 0)
        : counter{counter_}, start{std::chrono::steady_clock::now()} {
        counter.bytes += bytes_;
    }
    ~Scope() {
        ++counter.count;
        counter.micros += std::chrono::duration_cast<std::chrono::microseconds>(
                              std::chrono::steady_clock::now() - start)
                              .count();
    }

private:
    Counter& counter;
    std::chrono::steady_clock::time_point start;
};

inline void OnGameFrame() {
    static std::chrono::steady_clock::time_point last{};
    static u64 frame_index = 0;
    const auto now = std::chrono::steady_clock::now();
    const auto take = [](Counter& c) {
        struct {
            u32 count;
            u64 ms;
            u64 kb;
        } out{c.count.exchange(0), c.micros.exchange(0) / 1000, c.bytes.exchange(0) / 1024};
        return out;
    };
    const auto sh = take(shader_compile);
    const auto pl = take(pipeline_compile);
    const auto tx = take(texture_refresh);
    const auto rs = take(make_resident);
    const u32 faults = page_faults.exchange(0);
    ++frame_index;
    // Time between successive game frames: under 12 ms, 12-21, 21-30, 30 or more. A frame that
    // takes more than 21 ms misses a 60 Hz refresh, so the previous picture is shown twice.
    static std::array<u32, 4> frame_interval{};
    if (last.time_since_epoch().count() != 0) {
        const f64 ms = std::chrono::duration<f64, std::milli>(now - last).count();
        ++frame_interval[ms < 12.0 ? 0 : ms < 21.0 ? 1 : ms < 30.0 ? 2 : 3];
    }
    {
        // Once a second: how many game frames arrived.
        static std::chrono::steady_clock::time_point second_start{};
        static u32 frames_this_second = 0;
        ++frames_this_second;
        if (second_start.time_since_epoch().count() == 0) {
            second_start = now;
        } else if (now - second_start >= std::chrono::seconds(1)) {
            const f64 secs = std::chrono::duration<f64>(now - second_start).count();
            LOG_INFO(Render_Vulkan,
                     "Frame rate: {:.1f} fps (frame #{}) | ms between frames: under 12: {}, "
                     "12-21: {}, 21-30: {}, 30 or more: {}",
                     frames_this_second / secs, frame_index, frame_interval[0], frame_interval[1],
                     frame_interval[2], frame_interval[3]);
            frame_interval = {};
            const u64 ranges = sec_dma_sync.bytes.exchange(0);
            const auto ds = take(sec_dma_sync);
            const auto dk = take(sec_dma_skipped);
            const auto du = take(sec_dma_upload);
            const auto wf = take(sec_write_fault);
            LOG_INFO(Render_Vulkan,
                     "DMA per second: resyncs {} ({} ms, {} ranges each) skipped {} | re-uploads {} "
                     "({} KB) | write faults {} ({} ms)",
                     ds.count, ds.ms, ds.count ? ranges / ds.count : 0, dk.count, du.count,
                     du.kb, wf.count, wf.ms);
            second_start = now;
            frames_this_second = 0;
        }
    }
    if (last.time_since_epoch().count() != 0) {
        const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - last).count();
        if (ms >= 50) {
            LOG_INFO(Render_Vulkan,
                     "Slow frame #{}: {} ms | shaders {} ({} ms) | pipelines {} ({} ms) | texture "
                     "uploads {} ({} ms, {} KB) | resident {} ({} ms, {} KB) | page faults {}",
                     frame_index, ms, sh.count, sh.ms, pl.count, pl.ms, tx.count, tx.ms, tx.kb,
                     rs.count, rs.ms, rs.kb, faults);
        }
    }
    last = now;
}

} // namespace Common::FrameProbe
