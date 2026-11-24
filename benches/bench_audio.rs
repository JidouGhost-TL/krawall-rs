use std::hint::black_box;
use std::time::Instant;

use krawall::{
    AccuracySynth, KrawallLoopMode, KrawallMixMode, KrawallSampleView, KrawallVersion,
    KrawallVoice, StudioSynth, KRAWALL_DEFAULT_CLAMP_SHIFT, KRAWALL_MIX_RATE,
};

fn main() {
    println!("=== krawall-rs Audio Performance Benchmarks ===");

    let pcm_u8: Vec<u8> = (0i32..4096)
        .map(|i| (128 + ((i % 32) - 16) * 6) as u8)
        .collect();
    let sample = KrawallSampleView::from_unsigned_pcm(0, KRAWALL_MIX_RATE, &pcm_u8);
    let mut acc_l = vec![0u16; 1024];
    let mut acc_r = vec![0u16; 1024];
    let mut dma_l = vec![0i8; 1024];
    let mut dma_r = vec![0i8; 1024];

    let iters = 100_000usize;

    // 1. AccuracySynth::mix_nn_stereo (1:1 pitch)
    for _ in 0..1000 {
        let mut pos = 0usize;
        let mut frac = 0u16;
        AccuracySynth::mix_nn_stereo(
            &pcm_u8,
            &mut pos,
            &mut frac,
            0x0001_0000,
            64,
            48,
            &mut acc_l,
            &mut acc_r,
        );
    }
    let start = Instant::now();
    for _ in 0..iters {
        let mut pos = 0usize;
        let mut frac = 0u16;
        AccuracySynth::mix_nn_stereo(
            black_box(&pcm_u8),
            &mut pos,
            &mut frac,
            black_box(0x0001_0000),
            64,
            48,
            &mut acc_l,
            &mut acc_r,
        );
        black_box((&acc_l, &acc_r));
    }
    let elapsed = start.elapsed();
    println!(
        "AccuracySynth::mix_nn_stereo (1:1, 1024f):          {:8.2} ns/iter ({:8.2} MSamples/sec)",
        elapsed.as_nanos() as f64 / iters as f64,
        ((iters * 1024) as f64 / elapsed.as_secs_f64()) / 1_000_000.0
    );

    // 2. AccuracySynth::mix_nn_stereo (resampled 0x15654)
    for _ in 0..1000 {
        let mut pos = 0usize;
        let mut frac = 0x1234u16;
        AccuracySynth::mix_nn_stereo(
            &pcm_u8,
            &mut pos,
            &mut frac,
            0x0001_5654,
            64,
            48,
            &mut acc_l,
            &mut acc_r,
        );
    }
    let start = Instant::now();
    for _ in 0..iters {
        let mut pos = 0usize;
        let mut frac = 0x1234u16;
        AccuracySynth::mix_nn_stereo(
            black_box(&pcm_u8),
            &mut pos,
            &mut frac,
            black_box(0x0001_5654),
            64,
            48,
            &mut acc_l,
            &mut acc_r,
        );
        black_box((&acc_l, &acc_r));
    }
    let elapsed = start.elapsed();
    println!(
        "AccuracySynth::mix_nn_stereo (resampled, 1024f):    {:8.2} ns/iter ({:8.2} MSamples/sec)",
        elapsed.as_nanos() as f64 / iters as f64,
        ((iters * 1024) as f64 / elapsed.as_secs_f64()) / 1_000_000.0
    );

    // 3. AccuracySynth::mix_hq_stereo (resampled 0x15654)
    for _ in 0..1000 {
        let mut pos = 0usize;
        let mut frac = 0x1234u16;
        AccuracySynth::mix_hq_stereo(
            &pcm_u8,
            &mut pos,
            &mut frac,
            0x0001_5654,
            64,
            48,
            &mut acc_l,
            &mut acc_r,
        );
    }
    let start = Instant::now();
    for _ in 0..iters {
        let mut pos = 0usize;
        let mut frac = 0x1234u16;
        AccuracySynth::mix_hq_stereo(
            black_box(&pcm_u8),
            &mut pos,
            &mut frac,
            black_box(0x0001_5654),
            64,
            48,
            &mut acc_l,
            &mut acc_r,
        );
        black_box((&acc_l, &acc_r));
    }
    let elapsed = start.elapsed();
    println!(
        "AccuracySynth::mix_hq_stereo (resampled, 1024f):    {:8.2} ns/iter ({:8.2} MSamples/sec)",
        elapsed.as_nanos() as f64 / iters as f64,
        ((iters * 1024) as f64 / elapsed.as_secs_f64()) / 1_000_000.0
    );

    // 4. AccuracySynth::mix_hq_stereo_versioned (EarlyLsr2)
    for _ in 0..1000 {
        let mut pos = 0usize;
        let mut frac = 0x1234u16;
        AccuracySynth::mix_hq_stereo_versioned(
            KrawallVersion::EarlyLsr2,
            &pcm_u8,
            &mut pos,
            &mut frac,
            0x0001_5654,
            40,
            20,
            &mut acc_l,
            &mut acc_r,
        );
    }
    let start = Instant::now();
    for _ in 0..iters {
        let mut pos = 0usize;
        let mut frac = 0x1234u16;
        AccuracySynth::mix_hq_stereo_versioned(
            black_box(KrawallVersion::EarlyLsr2),
            black_box(&pcm_u8),
            &mut pos,
            &mut frac,
            black_box(0x0001_5654),
            40,
            20,
            &mut acc_l,
            &mut acc_r,
        );
        black_box((&acc_l, &acc_r));
    }
    let elapsed = start.elapsed();
    println!(
        "AccuracySynth::mix_hq_versioned (EarlyLsr2, 1024f): {:8.2} ns/iter ({:8.2} MSamples/sec)",
        elapsed.as_nanos() as f64 / iters as f64,
        ((iters * 1024) as f64 / elapsed.as_secs_f64()) / 1_000_000.0
    );

    // 5. AccuracySynth::clamp_u16_to_dma8 (Standard)
    for _ in 0..1000 {
        AccuracySynth::clamp_u16_to_dma8(
            &acc_l,
            &acc_r,
            &mut dma_l,
            &mut dma_r,
            64,
            48,
            KRAWALL_DEFAULT_CLAMP_SHIFT,
        );
    }
    let start = Instant::now();
    for _ in 0..iters {
        AccuracySynth::clamp_u16_to_dma8(
            black_box(&acc_l),
            black_box(&acc_r),
            &mut dma_l,
            &mut dma_r,
            64,
            48,
            KRAWALL_DEFAULT_CLAMP_SHIFT,
        );
        black_box((&dma_l, &dma_r));
    }
    let elapsed = start.elapsed();
    println!(
        "AccuracySynth::clamp_u16_to_dma8 (Standard, 1024f): {:8.2} ns/iter ({:8.2} MSamples/sec)",
        elapsed.as_nanos() as f64 / iters as f64,
        ((iters * 1024) as f64 / elapsed.as_secs_f64()) / 1_000_000.0
    );

    // 6. AccuracySynth::clamp_u16_to_dma8_versioned (MidSignedUnpack)
    for _ in 0..1000 {
        AccuracySynth::clamp_u16_to_dma8_versioned(
            KrawallVersion::MidSignedUnpack,
            &acc_l,
            &acc_r,
            &mut dma_l,
            &mut dma_r,
            64,
            48,
            KRAWALL_DEFAULT_CLAMP_SHIFT,
        );
    }
    let start = Instant::now();
    for _ in 0..iters {
        AccuracySynth::clamp_u16_to_dma8_versioned(
            black_box(KrawallVersion::MidSignedUnpack),
            black_box(&acc_l),
            black_box(&acc_r),
            &mut dma_l,
            &mut dma_r,
            64,
            48,
            KRAWALL_DEFAULT_CLAMP_SHIFT,
        );
        black_box((&dma_l, &dma_r));
    }
    let elapsed = start.elapsed();
    println!(
        "AccuracySynth::clamp_versioned (MidSigned, 1024f):  {:8.2} ns/iter ({:8.2} MSamples/sec)",
        elapsed.as_nanos() as f64 / iters as f64,
        ((iters * 1024) as f64 / elapsed.as_secs_f64()) / 1_000_000.0
    );

    // 7. AccuracySynth::mix_voice_view (1:1 non-looped)
    for _ in 0..1000 {
        let mut voice = KrawallVoice::with_raw_params(
            0,
            0,
            0,
            0x0001_0000,
            64,
            48,
            KrawallLoopMode::None,
            KrawallMixMode::NearestNeighbor,
        );
        AccuracySynth::mix_voice_view(&sample, &mut voice, &mut acc_l, &mut acc_r);
    }
    let start = Instant::now();
    for _ in 0..iters {
        let mut voice = KrawallVoice::with_raw_params(
            0,
            0,
            0,
            0x0001_0000,
            64,
            48,
            KrawallLoopMode::None,
            KrawallMixMode::NearestNeighbor,
        );
        AccuracySynth::mix_voice_view(
            black_box(&sample),
            &mut voice,
            &mut acc_l,
            &mut acc_r,
        );
        black_box((&acc_l, &acc_r));
    }
    let elapsed = start.elapsed();
    println!(
        "AccuracySynth::mix_voice_view (NN 1:1, 1024f):      {:8.2} ns/iter ({:8.2} MSamples/sec)",
        elapsed.as_nanos() as f64 / iters as f64,
        ((iters * 1024) as f64 / elapsed.as_secs_f64()) / 1_000_000.0
    );

    // 8. StudioSynth::render_voice_f32 (Lanczos-3 48 kHz)
    let studio = StudioSynth::new(48_000);
    let mut f_l = vec![0.0f32; 1024];
    let mut f_r = vec![0.0f32; 1024];
    let studio_iters = 5_000usize;
    for _ in 0..100 {
        let mut pos = 0.0f64;
        f_l.fill(0.0);
        f_r.fill(0.0);
        studio.render_voice_f32(&sample, &mut pos, 16_384.0, 64, 0, &mut f_l, &mut f_r);
    }
    let start = Instant::now();
    for _ in 0..studio_iters {
        let mut pos = 0.0f64;
        f_l.fill(0.0);
        f_r.fill(0.0);
        studio.render_voice_f32(
            black_box(&sample),
            &mut pos,
            black_box(16_384.0),
            64,
            0,
            &mut f_l,
            &mut f_r,
        );
        black_box((&f_l, &f_r));
    }
    let elapsed = start.elapsed();
    println!(
        "StudioSynth::render_voice_f32 (Lanczos 48k, 1024f): {:8.2} ns/iter ({:8.2} MSamples/sec)",
        elapsed.as_nanos() as f64 / studio_iters as f64,
        ((studio_iters * 1024) as f64 / elapsed.as_secs_f64()) / 1_000_000.0
    );
}
