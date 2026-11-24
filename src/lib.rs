//! # krawall-rs
//!
//! Pure Rust, zero-dependency parser, software synthesizer, and byte-exact golden-tested
//! emulator for the **Krawall GBA Tracker Sound Engine** (Sebastian Kienzl) on the
//! Game Boy Advance.
//!
//! Krawall powered over 100 commercial GBA titles including *The Sims: Bustin' Out*,
//! *The Sims 2*, *LEGO Star Wars: The Video Game*, *Franklin the Turtle*, and
//! *Hugo - Bukkazoom!*.
//!
//! ## Features
//! - **Zero dependencies** in `[dependencies]`, `#![no_std]` compatible, `#![deny(unsafe_code)]`.
//! - **Dual synthesis pipelines**:
//!   - [`AccuracySynth`]: Bit-exact emulation of Krawall's 16,384 Hz ARM IWRAM mixer
//!     (nearest-neighbor & HQ linear interpolation, DC-bias tracking, silence-fill,
//!     and DirectSound 8-bit LUT clamping).
//!   - [`StudioSynth`]: 48,000 Hz `f32` high-fidelity renderer with band-limited Lanczos
//!     interpolation and equal-power stereo panning.
//! - **Golden tested**: 3,072 / 3,072 bytes verified strictly byte-for-byte against
//!   headless `libmgba` ARM execution across all 3 commercial Krawall mixer revisions
//!   ([`KrawallVersion::EarlyLsr2`], [`KrawallVersion::MidSignedUnpack`], and
//!   [`KrawallVersion::Standard`]).
//!
//! ## Quickstart
//!
//! ```rust
//! use krawall::{
//!     AccuracySynth, KrawallSampleView, KrawallVoice,
//!     KRAWALL_DEFAULT_CLAMP_SHIFT, KRAWALL_MIX_RATE,
//! };
//!
//! let pcm_u8: &[u8] = &[0x80, 0xA0, 0xC0, 0xE0, 0xC0, 0xA0, 0x80, 0x60];
//! let sample = KrawallSampleView::from_unsigned_pcm(0, KRAWALL_MIX_RATE, pcm_u8);
//! let mut voice = KrawallVoice::new(0, KRAWALL_MIX_RATE, 64, 0, 128);
//! let mut acc_l = [0u16; 8];
//! let mut acc_r = [0u16; 8];
//! let mut dma_l = [0i8; 8];
//! let mut dma_r = [0i8; 8];
//!
//! let (tot_l, tot_r) = AccuracySynth::mix_voice_view(&sample, &mut voice, &mut acc_l, &mut acc_r);
//! AccuracySynth::clamp_u16_to_dma8(
//!     &acc_l, &acc_r, &mut dma_l, &mut dma_r, tot_l, tot_r, KRAWALL_DEFAULT_CLAMP_SHIFT,
//! );
//! assert_eq!(dma_l, dma_r);
//! ```

#![cfg_attr(not(feature = "std"), no_std)]
#![deny(unsafe_code)]

#[cfg(feature = "alloc")]
extern crate alloc;

pub mod error;
pub mod freq;
pub mod sample;
pub mod sequence;
pub mod synth;
pub mod version;

pub use error::{Error, Result};
pub use freq::{
    apply_fine_tune, compute_channel_volumes, freq_to_inc_q16, freq_to_inc_q16_at_rate,
    inc_q16_to_freq, note_to_freq_hz, period_to_freq_hz, rate_from_timer_reload, GBA_CLOCK_HZ,
    KRAWALL_DEFAULT_C2_FREQ, KRAWALL_MIX_RATE, KRAWALL_PERIOD_CLOCK, KRAWALL_TIMER0_RELOAD,
};
#[cfg(feature = "alloc")]
pub use sample::KrawallSample;
pub use sample::{
    KrawallLoopMode, KrawallSampleHeader, KrawallSampleView, GBA_ROM_BUS_BASE,
    KRAWALL_SAMPLE_HEADER_SIZE, KRAWALL_SILENCE_BYTE, KRAWALL_TAIL_SIZE,
};
pub use sequence::{KrawallEvent, KrawallMixMode, KrawallModule, KrawallPattern, KrawallVoice};
pub use synth::{sqrt_f64, AccuracySynth, StudioSynth, KRAWALL_DEFAULT_CLAMP_SHIFT};
pub use version::KrawallVersion;
