use std::time::Instant;
use krawall::{AccuracySynth, KRAWALL_DEFAULT_CLAMP_SHIFT};

fn main() {
    println!("=== krawall-rs Audio Performance Benchmarks ===");

    let pcm_u8: Vec<u8> = (0i32..4096)
        .map(|i| (128 + ((i % 32) - 16) * 6) as u8)
        .collect();
    let mut acc_l = vec![0u16; 1024];
    let mut acc_r = vec![0u16; 1024];
    let mut dma_l = vec![0i8; 1024];
    let mut dma_r = vec![0i8; 1024];

    let iters = 100_000usize;

    // 1. AccuracySynth::mix_nn_stereo (1:1 pitch)
    for _ in 0..1000 {
        let mut pos = 0usize;
        let mut frac = 0u16;
        AccuracySynth::mix_nn_stereo(&pcm_u8, &mut pos, &mut frac, 0x0001_0000, 64, 48, &mut acc_l, &mut acc_r);
    }
    let start = Instant::now();
    for _ in 0..iters {
        let mut pos = 0usize;
        let mut frac = 0u16;
        AccuracySynth::mix_nn_stereo(&pcm_u8, &mut pos, &mut frac, 0x0001_0000, 64, 48, &mut acc_l, &mut acc_r);
    }
    let elapsed = start.elapsed();
    println!(
        "AccuracySynth::mix_nn_stereo (1:1 pitch, 1024 frames): {:.2} ns/iter ({:.2} MSamples/sec)",
        elapsed.as_nanos() as f64 / iters as f64,
        ((iters * 1024) as f64 / elapsed.as_secs_f64()) / 1_000_000.0
    );

    // 2. AccuracySynth::mix_hq_stereo (resampled 0x15654)
    for _ in 0..1000 {
        let mut pos = 0usize;
        let mut frac = 0x1234u16;
        AccuracySynth::mix_hq_stereo(&pcm_u8, &mut pos, &mut frac, 0x0001_5654, 64, 48, &mut acc_l, &mut acc_r);
    }
    let start = Instant::now();
    for _ in 0..iters {
        let mut pos = 0usize;
        let mut frac = 0x1234u16;
        AccuracySynth::mix_hq_stereo(&pcm_u8, &mut pos, &mut frac, 0x0001_5654, 64, 48, &mut acc_l, &mut acc_r);
    }
    let elapsed = start.elapsed();
    println!(
        "AccuracySynth::mix_hq_stereo (resampled, 1024 frames): {:.2} ns/iter ({:.2} MSamples/sec)",
        elapsed.as_nanos() as f64 / iters as f64,
        ((iters * 1024) as f64 / elapsed.as_secs_f64()) / 1_000_000.0
    );

    // 3. AccuracySynth::clamp_u16_to_dma8
    for _ in 0..1000 {
        AccuracySynth::clamp_u16_to_dma8(&acc_l, &acc_r, &mut dma_l, &mut dma_r, 64, 48, KRAWALL_DEFAULT_CLAMP_SHIFT);
    }
    let start = Instant::now();
    for _ in 0..iters {
        AccuracySynth::clamp_u16_to_dma8(&acc_l, &acc_r, &mut dma_l, &mut dma_r, 64, 48, KRAWALL_DEFAULT_CLAMP_SHIFT);
    }
    let elapsed = start.elapsed();
    println!(
        "AccuracySynth::clamp_u16_to_dma8 (1024 stereo frames): {:.2} ns/iter ({:.2} MSamples/sec)",
        elapsed.as_nanos() as f64 / iters as f64,
        ((iters * 1024) as f64 / elapsed.as_secs_f64()) / 1_000_000.0
    );
}
