//! Krawall hardware timer, period, `c2_freq`, `fine_tune`, and fixed-point phase math.
//!
//! Krawall's ARM IWRAM mixer runs at an exact power-of-two sample rate of **16,384 Hz**
//! (`REG_TM0CNT_L = 0xFC00`, i.e. `16,777,216 / 1024 = 16,384 Hz`). Because `65,536 / 16,384 == 4`,
//! converting any playback frequency in Hz to a Q16.16 fixed-point phase increment is a single
//! 2-bit left shift (`freq_hz << 2`).

/// GBA master CPU clock frequency in Hz (`2^24`).
pub const GBA_CLOCK_HZ: u32 = 16_777_216;

/// Krawall native DirectSound mixing rate in Hz (`16,777,216 / 1024`).
pub const KRAWALL_MIX_RATE: u32 = 16_384;

/// Krawall GBA Timer 0 reload value (`65536 - 1024 = 0xFC00`) producing 16,384 Hz.
pub const KRAWALL_TIMER0_RELOAD: u16 = 0xFC00;

/// Standard ProTracker / Scream Tracker 3 C2 playback rate in Hz.
pub const KRAWALL_DEFAULT_C2_FREQ: u32 = 8_363;

/// S3M / Krawall Amiga period clock numerator (`8363 * 1712 = 14,317,456`).
pub const KRAWALL_PERIOD_CLOCK: u32 = 14_317_456;

/// Convert a playback frequency in Hz to Krawall's Q16.16 fixed-point phase increment
/// at the native 16,384 Hz mixing rate.
///
/// Matches the ARM driver instruction `lsl r2, r3, #2` (`freq_hz << 2`), where
/// `16,384 Hz` maps to `0x0001_0000` (1.0 sample per output frame).
#[inline]
pub const fn freq_to_inc_q16(freq_hz: u32) -> i32 {
    (freq_hz << 2) as i32
}

/// Convert a playback frequency in Hz to a Q16.16 fixed-point phase increment
/// for an arbitrary output sample rate.
#[inline]
pub const fn freq_to_inc_q16_at_rate(freq_hz: u32, output_rate: u32) -> i32 {
    if output_rate == 0 {
        return 0;
    }
    if output_rate == KRAWALL_MIX_RATE {
        return freq_to_inc_q16(freq_hz);
    }
    (((freq_hz as u64) << 16) / (output_rate as u64)) as i32
}

/// Convert a Krawall Q16.16 fixed-point phase increment back to Hz (at 16,384 Hz mix rate).
#[inline]
pub const fn inc_q16_to_freq(inc_q16: i32) -> u32 {
    inc_q16.unsigned_abs() >> 2
}

/// Convert a GBA Timer 0 reload value to the corresponding DirectSound sample rate in Hz.
#[inline]
pub const fn rate_from_timer_reload(reload: u16) -> u32 {
    let cycles = 65_536u32.saturating_sub(reload as u32);
    if cycles == 0 {
        return 0;
    }
    (GBA_CLOCK_HZ + cycles / 2) / cycles
}

/// Convert a tracker period (S3M/MOD style) to a frequency in Hz.
#[inline]
pub const fn period_to_freq_hz(period: u32) -> u32 {
    if period == 0 {
        return 0;
    }
    KRAWALL_PERIOD_CLOCK / period
}

/// Compute exact Krawall stereo channel volumes `(vol_left, vol_right)` from channel volume,
/// master volume, and signed panning.
///
/// Matches Krawall's ARM channel setup routine (`0x030002e8..0x0300031c` in *Franklin the Turtle*):
/// ```text
/// combined  = master_vol * vol
/// vol_left  = (combined * (64 - pan)) >> 13
/// vol_right = (combined * (64 + pan)) >> 13
/// ```
/// where `vol` is `0..=64`, `master_vol` is `0..=128`, and `pan` is `-64..=64` (`0` = center,
/// `-64` = hard left / mono mode, `+64` = hard right).
#[inline]
pub const fn compute_channel_volumes(vol: u8, master_vol: u8, pan: i8) -> (u8, u8) {
    let p = if pan < -64 {
        -64
    } else if pan > 64 {
        64
    } else {
        pan
    } as i32;
    let combined = (master_vol as u32) * (vol as u32);
    let left_scale = (64 - p) as u32;
    let right_scale = (64 + p) as u32;
    let vol_left = ((combined * left_scale) >> 13) as u8;
    let vol_right = ((combined * right_scale) >> 13) as u8;
    (vol_left, vol_right)
}

/// 12-tone equal-temperament semitone multipliers in Q16.16 fixed-point (`2^(n/12) * 65536`).
const SEMITONE_Q16: [u32; 12] = [
    65_536,  // 0: 1.000000
    69_433,  // 1: 1.059463
    73_562,  // 2: 1.122462
    77_936,  // 3: 1.189207
    82_570,  // 4: 1.259921
    87_480,  // 5: 1.334840
    92_682,  // 6: 1.414214
    98_193,  // 7: 1.498307
    104_032, // 8: 1.587401
    110_218, // 9: 1.681793
    116_772, // 10: 1.781797
    123_715, // 11: 1.887749
];

/// Compute playback frequency in Hz for a sample with base rate `c2_freq` shifted by
/// `semitone_offset` semitones relative to C2 (where `0` = C2, `+12` = C3, `-12` = C1).
///
/// Completely `#![no_std]` and integer-only.
#[inline]
pub const fn note_to_freq_hz(c2_freq: u32, semitone_offset: i32) -> u32 {
    if c2_freq == 0 {
        return 0;
    }
    let octave = semitone_offset.div_euclid(12);
    let semi = semitone_offset.rem_euclid(12) as usize;
    let scaled = ((c2_freq as u64) * (SEMITONE_Q16[semi] as u64) + 32_768) >> 16;
    if octave >= 0 {
        let shift = if octave > 16 { 16 } else { octave as u32 };
        let val = scaled << shift;
        if val > u32::MAX as u64 {
            u32::MAX
        } else {
            val as u32
        }
    } else {
        let shift = if -octave > 31 { 31 } else { (-octave) as u32 };
        (scaled >> shift) as u32
    }
}

/// Convert Krawall's `mconv` precomputed `fine_tune` table offset (in 1/64-semitone units
/// modulo 65536, signed `i16`) into a fine-tuned playback frequency in Hz.
#[inline]
pub const fn apply_fine_tune(base_freq: u32, fine_tune: u16) -> u32 {
    if fine_tune == 0 || base_freq == 0 {
        return base_freq;
    }
    let signed_ft = fine_tune as i16 as i32;
    // Krawall's `mconv` stores log2(c2_freq / 8363) * 768 * 4 (or table offset in 1/3072-octave steps).
    // Approximate small fine-tune deltas in integer arithmetic: semitones = signed_ft / 256.
    let semi = signed_ft / 256;
    note_to_freq_hz(base_freq, semi)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_krawall_timer_and_q16_inc() {
        assert_eq!(rate_from_timer_reload(KRAWALL_TIMER0_RELOAD), 16_384);
        assert_eq!(freq_to_inc_q16(16_384), 0x0001_0000);
        assert_eq!(freq_to_inc_q16(8_192), 0x0000_8000);
        assert_eq!(freq_to_inc_q16(32_768), 0x0002_0000);
        assert_eq!(inc_q16_to_freq(0x0001_0000), 16_384);
        assert_eq!(inc_q16_to_freq(-0x0001_0000), 16_384);
    }

    #[test]
    fn test_compute_channel_volumes_exact_arm() {
        // Center pan (0), full vol (64), full master (128):
        // (128 * 64 * 64) >> 13 = 524288 >> 13 = 64
        assert_eq!(compute_channel_volumes(64, 128, 0), (64, 64));
        // Hard left (-64), full vol (64), full master (128):
        // left = (128 * 64 * 128) >> 13 = 128, right = 0
        assert_eq!(compute_channel_volumes(64, 128, -64), (128, 0));
        // Hard right (+64):
        assert_eq!(compute_channel_volumes(64, 128, 64), (0, 128));
        // Half master (64), full vol (64), center pan (0):
        assert_eq!(compute_channel_volumes(64, 64, 0), (32, 32));
    }

    #[test]
    fn test_note_to_freq_hz() {
        assert_eq!(note_to_freq_hz(8_363, 0), 8_363);
        assert_eq!(note_to_freq_hz(8_363, 12), 16_726);
        assert_eq!(note_to_freq_hz(16_000, -12), 8_000);
        assert_eq!(note_to_freq_hz(16_384, 7), 24_548); // Perfect fifth ~1.4983x
    }
}
