//! Byte-exact mGBA ARM Golden Integration Tests for Krawall GBA Tracker Sound Engine (`krawall-rs`).
//!
//! Verifies:
//! - **Tier 1**: Native 18-byte Krawall `Sample` header parsing, unsigned `0x80`-centered PCM
//!   waveform continuity, and 64-byte `mconv` unrolled loop / silence tail validation across
//!   *Franklin the Turtle* and *Hugo - Bukkazoom!*.
//! - **Tier 2**: Krawall 16,384 Hz hardware timer (`0xFC00`), Q16.16 `freq << 2` phase step,
//!   and exact ARM volume/panning scaling math.
//! - **Tier 3**: Strict byte-for-byte `assert_eq!` verification (2,048 / 2,048 bytes) of
//!   `AccuracySynth` against real headless `libmgba` ARM IWRAM mixer dumps:
//!   1. *Franklin the Turtle* Nearest-Neighbor Stereo 1:1 pitch (`0x03000838` + `0x03000C48`)
//!   2. *Franklin the Turtle* Nearest-Neighbor Stereo resampled (`0x03000838` + `0x03000C48`)
//!   3. *Franklin the Turtle* HQ Linear-Interpolated Stereo (`0x03000AC0` + `0x03000C48`)
//!   4. *Franklin the Turtle* Looped sample wrap across 64-byte unrolled loop tail
//!   5. *Franklin the Turtle* 6-voice saturated clamping (`+127` / `-128` LUT clamp)
//!   6. *Franklin the Turtle* One-shot end-of-sample silence-fill (`0x03000BC4` + `0x03000C48`)
//!   7. *Hugo - Bukkazoom!* Nearest-Neighbor Stereo (`0x03000CA8` + `0x030010B8`)
//!   8. *Hugo - Bukkazoom!* HQ Linear-Interpolated Stereo (`0x03000F30` + `0x030010B8`)

use krawall::{
    compute_channel_volumes, freq_to_inc_q16, rate_from_timer_reload, AccuracySynth,
    KrawallLoopMode, KrawallMixMode, KrawallSampleView, KrawallVoice, StudioSynth,
    KRAWALL_DEFAULT_CLAMP_SHIFT, KRAWALL_MIX_RATE, KRAWALL_TIMER0_RELOAD,
};
#[cfg(feature = "alloc")]
use krawall::KrawallSample;

const MANIFEST_JSON: &str = include_str!("golden/manifest.json");

const FRANKLIN_S7_BLOB: &[u8] = include_bytes!("golden/franklin_sample7_oneshot.krw");
const FRANKLIN_S75_BLOB: &[u8] = include_bytes!("golden/franklin_sample75_looped.krw");
const HUGO_S26_BLOB: &[u8] = include_bytes!("golden/hugo_sample26_oneshot.krw");

const GOLDEN_FRANKLIN_NN_1TO1: &[u8] = include_bytes!("golden/franklin_nn_1to1.bin");
const GOLDEN_FRANKLIN_NN_RESAMPLED: &[u8] = include_bytes!("golden/franklin_nn_resampled.bin");
const GOLDEN_FRANKLIN_HQ_RESAMPLED: &[u8] = include_bytes!("golden/franklin_hq_resampled.bin");
const GOLDEN_FRANKLIN_NN_LOOPED: &[u8] = include_bytes!("golden/franklin_nn_looped.bin");
const GOLDEN_FRANKLIN_MULTI_CLAMPED: &[u8] = include_bytes!("golden/franklin_multi_clamped.bin");
const GOLDEN_FRANKLIN_ONESHOT_SILENCE: &[u8] =
    include_bytes!("golden/franklin_oneshot_silence_tail.bin");
const GOLDEN_HUGO_NN_STEREO: &[u8] = include_bytes!("golden/hugo_nn_stereo.bin");
const GOLDEN_HUGO_HQ_STEREO: &[u8] = include_bytes!("golden/hugo_hq_stereo.bin");

fn adjacent_mad_signed(data: &[u8]) -> f64 {
    let mut sum: u64 = 0;
    for w in data.windows(2) {
        sum += ((w[1] as i8 as i32) - (w[0] as i8 as i32)).unsigned_abs() as u64;
    }
    sum as f64 / (data.len() - 1) as f64
}

fn adjacent_mad_unsigned(data: &[u8]) -> f64 {
    let mut sum: u64 = 0;
    for w in data.windows(2) {
        let a = (w[0] ^ 0x80) as i8 as i32;
        let b = (w[1] ^ 0x80) as i8 as i32;
        sum += (b - a).unsigned_abs() as u64;
    }
    sum as f64 / (data.len() - 1) as f64
}

fn assert_stereo_golden_exact(dma_l: &[i8], dma_r: &[i8], expected: &[u8], label: &str) {
    assert_eq!(expected.len(), dma_l.len() + dma_r.len());
    let n = dma_l.len();
    for i in 0..n {
        assert_eq!(
            dma_l[i], expected[i] as i8,
            "{label} LEFT mismatch at frame {i}"
        );
    }
    for i in 0..n {
        assert_eq!(
            dma_r[i],
            expected[n + i] as i8,
            "{label} RIGHT mismatch at frame {i}"
        );
    }
}

#[test]
fn test_manifest_integrity() {
    assert!(!MANIFEST_JSON.is_empty());
    for expected in [
        "franklin_sample7_oneshot.krw",
        "franklin_sample75_looped.krw",
        "hugo_sample26_oneshot.krw",
        "franklin_nn_1to1.bin",
        "franklin_nn_resampled.bin",
        "franklin_hq_resampled.bin",
        "franklin_nn_looped.bin",
        "franklin_multi_clamped.bin",
        "franklin_oneshot_silence_tail.bin",
        "hugo_nn_stereo.bin",
        "hugo_hq_stereo.bin",
    ] {
        assert!(MANIFEST_JSON.contains(expected), "manifest missing {expected}");
    }
}

// ============================================================================
// Tier 1: Native Krawall 18-byte Sample Header, Unsigned PCM & 64-byte Tail
// ============================================================================

#[test]
fn test_tier1_franklin_oneshot_sample_parse_and_continuity() {
    let view = KrawallSampleView::parse_blob(7, FRANKLIN_S7_BLOB, true).unwrap();
    assert_eq!(view.len(), 1064);
    assert_eq!(view.c2_freq, 8_363);
    assert_eq!(view.fine_tune, 0);
    assert_eq!(view.vol_default, 64);
    assert_eq!(view.loop_mode, KrawallLoopMode::None);
    assert_eq!(view.loop_length, 0);
    assert_eq!(view.loop_start(), None);
    assert_eq!(view.tail_u8.len(), 64);
    assert!(view.has_valid_tail());
    assert!(!view.is_silent());

    let mad_u = adjacent_mad_unsigned(view.pcm_u8);
    let mad_s = adjacent_mad_signed(view.pcm_u8);
    assert!(
        mad_u < mad_s,
        "Krawall unsigned 0x80 PCM must be smoother than signed (unsigned={mad_u:.2}, signed={mad_s:.2})"
    );
}

#[test]
fn test_tier1_franklin_looped_sample_parse_and_unrolled_tail() {
    let view = KrawallSampleView::parse_blob(75, FRANKLIN_S75_BLOB, true).unwrap();
    assert_eq!(view.len(), 1040);
    assert_eq!(view.c2_freq, 8_000);
    assert_eq!(view.fine_tune, 156);
    assert_eq!(view.vol_default, 64);
    assert_eq!(view.loop_mode, KrawallLoopMode::Forward);
    assert_eq!(view.loop_length, 1040);
    assert_eq!(view.loop_start(), Some(0));
    assert_eq!(view.loop_end(), Some(1040));
    assert_eq!(view.tail_u8.len(), 64);
    assert!(view.has_valid_tail());
    // Verify the 64-byte unrolled tail matches the start of the loop byte-for-byte
    assert_eq!(view.tail_u8, &view.pcm_u8[..64]);
}

#[test]
fn test_tier1_hugo_oneshot_sample_parse() {
    let view = KrawallSampleView::parse_blob(26, HUGO_S26_BLOB, true).unwrap();
    assert_eq!(view.len(), 2962);
    assert_eq!(view.c2_freq, 21_909);
    assert_eq!(view.fine_tune, 4_184);
    assert_eq!(view.vol_default, 64);
    assert_eq!(view.loop_mode, KrawallLoopMode::None);
    assert!(view.has_valid_tail());

    let mad_u = adjacent_mad_unsigned(view.pcm_u8);
    let mad_s = adjacent_mad_signed(view.pcm_u8);
    assert!(mad_u < mad_s);
}

#[cfg(feature = "alloc")]
#[test]
fn test_tier1_owned_sample_roundtrip() {
    let view = KrawallSampleView::parse_blob(75, FRANKLIN_S75_BLOB, true).unwrap();
    let owned: KrawallSample = view.to_owned_sample();
    assert_eq!(owned.len(), view.len());
    assert_eq!(owned.pcm_u8(), view.pcm_u8);
    assert!(owned.as_view().has_valid_tail());
}

// ============================================================================
// Tier 2: Hardware Timer, Q16.16 Phase Step & Exact ARM Volume Math
// ============================================================================

#[test]
fn test_tier2_hardware_timer_and_volume_math() {
    assert_eq!(rate_from_timer_reload(KRAWALL_TIMER0_RELOAD), KRAWALL_MIX_RATE);
    assert_eq!(KRAWALL_MIX_RATE, 16_384);
    assert_eq!(freq_to_inc_q16(KRAWALL_MIX_RATE), 0x0001_0000);
    assert_eq!(compute_channel_volumes(64, 128, 0), (64, 64));
    assert_eq!(compute_channel_volumes(64, 128, -64), (128, 0));
    assert_eq!(compute_channel_volumes(64, 128, 64), (0, 128));
}

// ============================================================================
// Tier 3: Strict Byte-for-Byte mGBA ARM Golden Mixer Verification (2,048 bytes)
// ============================================================================

#[test]
fn test_tier3_franklin_arm_golden_nn_1to1() {
    let view = KrawallSampleView::parse_blob(7, FRANKLIN_S7_BLOB, true).unwrap();
    let mut acc_l = [0u16; 128];
    let mut acc_r = [0u16; 128];
    let mut dma_l = [0i8; 128];
    let mut dma_r = [0i8; 128];
    let mut pos = 0usize;
    let mut frac = 0u16;

    AccuracySynth::mix_nn_stereo(
        view.pcm_with_tail(),
        &mut pos,
        &mut frac,
        0x0001_0000,
        64,
        32,
        &mut acc_l,
        &mut acc_r,
    );
    AccuracySynth::clamp_u16_to_dma8(
        &acc_l,
        &acc_r,
        &mut dma_l,
        &mut dma_r,
        64,
        32,
        KRAWALL_DEFAULT_CLAMP_SHIFT,
    );

    assert_stereo_golden_exact(&dma_l, &dma_r, GOLDEN_FRANKLIN_NN_1TO1, "franklin_nn_1to1");
}

#[test]
fn test_tier3_franklin_arm_golden_nn_resampled() {
    let view = KrawallSampleView::parse_blob(7, FRANKLIN_S7_BLOB, true).unwrap();
    let mut acc_l = [0u16; 128];
    let mut acc_r = [0u16; 128];
    let mut dma_l = [0i8; 128];
    let mut dma_r = [0i8; 128];
    let mut pos = 0usize;
    let mut frac = 0x4000u16;

    AccuracySynth::mix_nn_stereo(
        view.pcm_with_tail(),
        &mut pos,
        &mut frac,
        0x0001_6A09,
        52,
        28,
        &mut acc_l,
        &mut acc_r,
    );
    AccuracySynth::clamp_u16_to_dma8(
        &acc_l,
        &acc_r,
        &mut dma_l,
        &mut dma_r,
        52,
        28,
        KRAWALL_DEFAULT_CLAMP_SHIFT,
    );

    assert_stereo_golden_exact(
        &dma_l,
        &dma_r,
        GOLDEN_FRANKLIN_NN_RESAMPLED,
        "franklin_nn_resampled",
    );
}

#[test]
fn test_tier3_franklin_arm_golden_hq_resampled() {
    let view = KrawallSampleView::parse_blob(7, FRANKLIN_S7_BLOB, true).unwrap();
    let mut acc_l = [0u16; 128];
    let mut acc_r = [0u16; 128];
    let mut dma_l = [0i8; 128];
    let mut dma_r = [0i8; 128];
    let mut pos = 0usize;
    let mut frac = 0x2000u16;

    AccuracySynth::mix_hq_stereo(
        view.pcm_with_tail(),
        &mut pos,
        &mut frac,
        0x0000_B504,
        60,
        40,
        &mut acc_l,
        &mut acc_r,
    );
    AccuracySynth::clamp_u16_to_dma8(
        &acc_l,
        &acc_r,
        &mut dma_l,
        &mut dma_r,
        60,
        40,
        KRAWALL_DEFAULT_CLAMP_SHIFT,
    );

    assert_stereo_golden_exact(
        &dma_l,
        &dma_r,
        GOLDEN_FRANKLIN_HQ_RESAMPLED,
        "franklin_hq_resampled",
    );
}

#[test]
fn test_tier3_franklin_arm_golden_nn_looped() {
    let view = KrawallSampleView::parse_blob(75, FRANKLIN_S75_BLOB, true).unwrap();
    let mut voice = KrawallVoice::with_raw_params(
        75,
        1000,
        0,
        0x0001_0000,
        56,
        48,
        KrawallLoopMode::Forward,
        KrawallMixMode::NearestNeighbor,
    );
    let mut acc_l = [0u16; 128];
    let mut acc_r = [0u16; 128];
    let mut dma_l = [0i8; 128];
    let mut dma_r = [0i8; 128];

    let (tot_l, tot_r) = AccuracySynth::mix_voice_view(&view, &mut voice, &mut acc_l, &mut acc_r);
    AccuracySynth::clamp_u16_to_dma8(
        &acc_l,
        &acc_r,
        &mut dma_l,
        &mut dma_r,
        tot_l,
        tot_r,
        KRAWALL_DEFAULT_CLAMP_SHIFT,
    );

    assert!(voice.is_active);
    assert_eq!(voice.pos, 88);
    assert_stereo_golden_exact(
        &dma_l,
        &dma_r,
        GOLDEN_FRANKLIN_NN_LOOPED,
        "franklin_nn_looped",
    );
}

#[test]
fn test_tier3_franklin_arm_golden_multi_clamped() {
    let view = KrawallSampleView::parse_blob(7, FRANKLIN_S7_BLOB, true).unwrap();
    let mut acc_l = [0u16; 128];
    let mut acc_r = [0u16; 128];
    let mut dma_l = [0i8; 128];
    let mut dma_r = [0i8; 128];

    let specs: [(usize, i32, u8, u8); 6] = [
        (0, 0x0001_0000, 64, 64),
        (0, 0x0001_0000, 64, 64),
        (0, 0x0001_0000, 64, 64),
        (0, 0x0001_0000, 64, 64),
        (2, 0x0001_0000, 64, 64),
        (4, 0x0001_0000, 64, 64),
    ];

    let mut tot_l = 0u32;
    let mut tot_r = 0u32;
    for (start_off, inc, vl, vr) in specs {
        let mut pos = start_off;
        let mut frac = 0u16;
        AccuracySynth::mix_nn_stereo(
            view.pcm_with_tail(),
            &mut pos,
            &mut frac,
            inc,
            vl,
            vr,
            &mut acc_l,
            &mut acc_r,
        );
        tot_l += vl as u32;
        tot_r += vr as u32;
    }

    AccuracySynth::clamp_u16_to_dma8(
        &acc_l,
        &acc_r,
        &mut dma_l,
        &mut dma_r,
        tot_l,
        tot_r,
        KRAWALL_DEFAULT_CLAMP_SHIFT,
    );

    assert_stereo_golden_exact(
        &dma_l,
        &dma_r,
        GOLDEN_FRANKLIN_MULTI_CLAMPED,
        "franklin_multi_clamped",
    );
    let sat_pos = dma_l.iter().chain(dma_r.iter()).filter(|&&v| v == 127).count();
    let sat_neg = dma_l.iter().chain(dma_r.iter()).filter(|&&v| v == -128).count();
    assert!(sat_pos > 100 && sat_neg > 0, "expected both +127 and -128 saturation");
}

#[test]
fn test_tier3_franklin_arm_golden_oneshot_silence_tail() {
    let view = KrawallSampleView::parse_blob(7, FRANKLIN_S7_BLOB, true).unwrap();
    let mut voice = KrawallVoice::with_raw_params(
        7,
        1004,
        0,
        0x0001_0000,
        64,
        40,
        KrawallLoopMode::None,
        KrawallMixMode::NearestNeighbor,
    );
    let mut acc_l = [0u16; 128];
    let mut acc_r = [0u16; 128];
    let mut dma_l = [0i8; 128];
    let mut dma_r = [0i8; 128];

    let (tot_l, tot_r) = AccuracySynth::mix_voice_view(&view, &mut voice, &mut acc_l, &mut acc_r);
    AccuracySynth::clamp_u16_to_dma8(
        &acc_l,
        &acc_r,
        &mut dma_l,
        &mut dma_r,
        tot_l,
        tot_r,
        KRAWALL_DEFAULT_CLAMP_SHIFT,
    );

    assert!(!voice.is_active);
    assert_eq!(&dma_l[60..], &[0i8; 68]);
    assert_eq!(&dma_r[60..], &[0i8; 68]);
    assert_stereo_golden_exact(
        &dma_l,
        &dma_r,
        GOLDEN_FRANKLIN_ONESHOT_SILENCE,
        "franklin_oneshot_silence_tail",
    );
}

#[test]
fn test_tier3_hugo_arm_golden_nn_stereo() {
    let view = KrawallSampleView::parse_blob(26, HUGO_S26_BLOB, true).unwrap();
    let mut acc_l = [0u16; 128];
    let mut acc_r = [0u16; 128];
    let mut dma_l = [0i8; 128];
    let mut dma_r = [0i8; 128];
    let mut pos = 128usize;
    let mut frac = 0x1234u16;

    AccuracySynth::mix_nn_stereo(
        view.pcm_with_tail(),
        &mut pos,
        &mut frac,
        0x0001_5654,
        64,
        48,
        &mut acc_l,
        &mut acc_r,
    );
    AccuracySynth::clamp_u16_to_dma8(
        &acc_l,
        &acc_r,
        &mut dma_l,
        &mut dma_r,
        64,
        48,
        KRAWALL_DEFAULT_CLAMP_SHIFT,
    );

    assert_stereo_golden_exact(&dma_l, &dma_r, GOLDEN_HUGO_NN_STEREO, "hugo_nn_stereo");
}

#[test]
fn test_tier3_hugo_arm_golden_hq_stereo() {
    let view = KrawallSampleView::parse_blob(26, HUGO_S26_BLOB, true).unwrap();
    let mut acc_l = [0u16; 128];
    let mut acc_r = [0u16; 128];
    let mut dma_l = [0i8; 128];
    let mut dma_r = [0i8; 128];
    let mut pos = 128usize;
    let mut frac = 0x1234u16;

    AccuracySynth::mix_hq_stereo(
        view.pcm_with_tail(),
        &mut pos,
        &mut frac,
        0x0001_5654,
        64,
        48,
        &mut acc_l,
        &mut acc_r,
    );
    AccuracySynth::clamp_u16_to_dma8(
        &acc_l,
        &acc_r,
        &mut dma_l,
        &mut dma_r,
        64,
        48,
        KRAWALL_DEFAULT_CLAMP_SHIFT,
    );

    assert_stereo_golden_exact(&dma_l, &dma_r, GOLDEN_HUGO_HQ_STEREO, "hugo_hq_stereo");
}

// ============================================================================
// Tier 4: StudioSynth 48 kHz Band-Limited Rendering
// ============================================================================

#[test]
fn test_tier4_studio_synth_real_krawall_sample() {
    let view = KrawallSampleView::parse_blob(7, FRANKLIN_S7_BLOB, true).unwrap();
    let studio = StudioSynth::new(48_000);
    let mut left = [0.0f32; 512];
    let mut right = [0.0f32; 512];
    let mut pos = 0.0f64;

    studio.render_voice_f32(&view, &mut pos, 8_363.0, 64, -32, &mut left, &mut right);

    let energy_l: f32 = left.iter().map(|x| x * x).sum();
    let energy_r: f32 = right.iter().map(|x| x * x).sum();
    assert!(energy_l > energy_r, "left pan (-32) must have more energy in left channel");
    assert!(energy_r > 0.01, "expected non-zero stereo bleed in right channel (got {energy_r})");
}
