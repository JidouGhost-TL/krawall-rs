//! Krawall module, voice, and tracker event structures.

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

use crate::freq::{compute_channel_volumes, freq_to_inc_q16};
use crate::sample::KrawallLoopMode;

/// Interpolation quality mode supported by Krawall's ARM IWRAM mixer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub enum KrawallMixMode {
    /// Nearest-neighbor resampling (`hq_flag == 0`, e.g. `0x03000838` in *Franklin the Turtle*).
    #[default]
    NearestNeighbor,
    /// High-quality linear interpolation (`hq_flag == 8`, e.g. `0x03000AC0` in *Franklin the Turtle*).
    HqLinear,
}

/// Active playback channel / voice state mirroring Krawall's 44-byte (`0x2C`) ARM channel struct.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct KrawallVoice {
    /// Index of the sample being played.
    pub sample_id: u32,
    /// Current integer sample index within the sample buffer.
    pub pos: usize,
    /// Unsigned 16-bit fractional phase (`0..=65535`, Q16.16 low halfword).
    pub frac: u16,
    /// Signed Q16.16 phase increment per output frame (`freq_hz << 2` at 16,384 Hz).
    /// Negative values play backwards during `PingPong` loops.
    pub inc_q16: i32,
    /// Channel volume (`0..=64`).
    pub volume: u8,
    /// Signed stereo panning (`-64` = hard left, `0` = center, `+64` = hard right).
    pub pan: i8,
    /// Precomputed left mixer gain (`0..=128`, stored at `ch+0x1E`).
    pub vol_left: u8,
    /// Precomputed right mixer gain (`0..=128`, stored at `ch+0x1F`).
    pub vol_right: u8,
    /// Loop mode (`None`, `Forward`, or `PingPong`).
    pub loop_mode: KrawallLoopMode,
    /// Mixer interpolation mode (`NearestNeighbor` or `HqLinear`).
    pub mix_mode: KrawallMixMode,
    /// Whether this voice is currently active.
    pub is_active: bool,
}

impl KrawallVoice {
    /// Create a new active Krawall voice from sample ID, playback frequency in Hz,
    /// channel volume (`0..=64`), panning (`-64..=64`), and master volume (`0..=128`).
    #[must_use]
    pub const fn new(
        sample_id: u32,
        freq_hz: u32,
        volume: u8,
        pan: i8,
        master_vol: u8,
    ) -> Self {
        let vol_clamped = if volume > 64 { 64 } else { volume };
        let (vol_left, vol_right) = compute_channel_volumes(vol_clamped, master_vol, pan);
        Self {
            sample_id,
            pos: 0,
            frac: 0,
            inc_q16: freq_to_inc_q16(freq_hz),
            volume: vol_clamped,
            pan,
            vol_left,
            vol_right,
            loop_mode: KrawallLoopMode::None,
            mix_mode: KrawallMixMode::NearestNeighbor,
            is_active: true,
        }
    }

    /// Create a voice with explicit Q16.16 phase increment and direct left/right mixer gains.
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    pub const fn with_raw_params(
        sample_id: u32,
        pos: usize,
        frac: u16,
        inc_q16: i32,
        vol_left: u8,
        vol_right: u8,
        loop_mode: KrawallLoopMode,
        mix_mode: KrawallMixMode,
    ) -> Self {
        Self {
            sample_id,
            pos,
            frac,
            inc_q16,
            volume: 64,
            pan: 0,
            vol_left,
            vol_right,
            loop_mode,
            mix_mode,
            is_active: true,
        }
    }

    /// Update channel volume and panning and recompute `vol_left` / `vol_right`.
    #[inline]
    pub fn set_volume_pan(&mut self, volume: u8, pan: i8, master_vol: u8) {
        self.volume = volume.min(64);
        self.pan = pan.clamp(-64, 64);
        let (vl, vr) = compute_channel_volumes(self.volume, master_vol, self.pan);
        self.vol_left = vl;
        self.vol_right = vr;
    }

    /// Update playback frequency in Hz (at 16,384 Hz mixing rate), preserving playback direction.
    #[inline]
    pub fn set_frequency_hz(&mut self, freq_hz: u32) {
        let mag = freq_to_inc_q16(freq_hz);
        self.inc_q16 = if self.inc_q16 < 0 {
            mag.wrapping_neg()
        } else {
            mag
        };
    }
}

/// Tracker sequencer event for Krawall modules.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KrawallEvent {
    /// Trigger a sample on `channel` at `freq_hz`, `volume` (`0..=64`), and `pan` (`-64..=64`).
    NoteOn {
        channel: u8,
        sample_id: u32,
        freq_hz: u32,
        volume: u8,
        pan: i8,
    },
    /// Stop / release the active voice on `channel`.
    NoteOff { channel: u8 },
    /// Change channel volume (`0..=64`).
    SetVolume { channel: u8, volume: u8 },
    /// Change channel stereo pan (`-64..=64`).
    SetPan { channel: u8, pan: i8 },
    /// Change channel playback frequency in Hz (portamento / vibrato).
    SetFrequency { channel: u8, freq_hz: u32 },
    /// Set tracker ticks-per-row speed.
    SetSpeed(u8),
    /// Set tracker BPM tempo.
    SetTempo(u8),
    /// Wait `u16` tracker ticks before dispatching the next event.
    WaitTicks(u16),
    /// End of pattern or module.
    End,
}

/// Krawall pattern container.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct KrawallPattern {
    pub id: u32,
    pub rows: u16,
    #[cfg(feature = "alloc")]
    pub events: Vec<KrawallEvent>,
}

/// Krawall tracker module metadata and pattern sequence.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct KrawallModule {
    pub channels: u8,
    pub initial_speed: u8,
    pub initial_tempo: u8,
    #[cfg(feature = "alloc")]
    pub order_list: Vec<u8>,
    #[cfg(feature = "alloc")]
    pub patterns: Vec<KrawallPattern>,
}
