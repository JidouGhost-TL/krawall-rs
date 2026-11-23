//! Software synthesis engines for the Krawall GBA Tracker Sound Engine.
//!
//! Provides two synthesis pipelines:
//! - [`AccuracySynth`]: Bit-exact emulation of Krawall's ARM IWRAM mixer (verified
//!   byte-for-byte against headless `libmgba` ARM execution across commercial GBA titles).
//! - [`StudioSynth`]: High-fidelity 48,000 Hz `f32` renderer with windowed-sinc
//!   (Lanczos-3) band-limited interpolation and true equal-power stereo panning.

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

use crate::freq::KRAWALL_MIX_RATE;
use crate::sample::{KrawallLoopMode, KrawallSampleView};
use crate::sequence::{KrawallMixMode, KrawallVoice};

/// Default Krawall clamp right-shift (`asr #5`), mapping the 16-bit unsigned accumulator
/// minus DC bias (`total_vol << 4`) into signed 8-bit DirectSound DMA FIFO samples.
pub const KRAWALL_DEFAULT_CLAMP_SHIFT: u32 = 5;

/// Compute square root of an `f64` value (`#![no_std]` compatible).
#[inline]
pub fn sqrt_f64(val: f64) -> f64 {
    #[cfg(feature = "std")]
    {
        val.sqrt()
    }
    #[cfg(not(feature = "std"))]
    {
        if val < 0.0 {
            return f64::NAN;
        }
        if val == 0.0 || val.is_infinite() || val.is_nan() {
            return val;
        }
        let bits = val.to_bits();
        let mut x = f64::from_bits((bits >> 1) + (1023u64 << 51));
        if x <= 0.0 || x.is_nan() {
            x = val;
        }
        for _ in 0..6 {
            x = 0.5 * (x + val / x);
        }
        x
    }
}

/// Bit-exact software mixer emulating Krawall's ARM IWRAM mixing, DC-bias tracking,
/// silence-fill, and LUT saturation clamping routines.
#[derive(Debug, Clone)]
pub struct AccuracySynth {
    pub sample_rate: u32,
    pub master_volume: u8,
    pub clamp_shift: u32,
    #[cfg(feature = "alloc")]
    pub voices: Vec<KrawallVoice>,
}

impl Default for AccuracySynth {
    fn default() -> Self {
        Self::new(KRAWALL_MIX_RATE)
    }
}

impl AccuracySynth {
    /// Create a new `AccuracySynth` configured for Krawall's native 16,384 Hz rate.
    pub fn new(sample_rate: u32) -> Self {
        Self {
            sample_rate,
            master_volume: 128,
            clamp_shift: KRAWALL_DEFAULT_CLAMP_SHIFT,
            #[cfg(feature = "alloc")]
            voices: Vec::new(),
        }
    }

    /// Add an active voice to the synthesizer.
    #[cfg(feature = "alloc")]
    pub fn add_voice(&mut self, voice: KrawallVoice) {
        self.voices.push(voice);
    }

    /// Clear all active voices.
    #[cfg(feature = "alloc")]
    pub fn clear_voices(&mut self) {
        self.voices.clear();
    }

    /// Bit-exact Krawall ARM nearest-neighbor stereo voice mixer
    /// (`0x03000838` in *Franklin the Turtle* / `0x03000CA8` in *Hugo - Bukkazoom!*).
    ///
    /// - Fetches unsigned 8-bit PCM (`ldrb`, `0x80` center) at `pos + (r7 >> 16)` (signed Q16.16).
    /// - Scales via `(s * vol) >> 3` (matching ARM 2x16-bit packed `mul` + `bic #0x70000` + `lsr #3`).
    /// - Accumulates into 16-bit unsigned stereo buffers `acc_left` and `acc_right`.
    /// - Updates `*pos` and `*frac` identically to the ARM epilogue (`add r5, r5, r7, asr #16; strh r7`).
    #[allow(clippy::too_many_arguments)]
    pub fn mix_nn_stereo(
        pcm_u8: &[u8],
        pos: &mut usize,
        frac: &mut u16,
        inc_q16: i32,
        vol_left: u8,
        vol_right: u8,
        acc_left: &mut [u16],
        acc_right: &mut [u16],
    ) {
        let len = acc_left.len().min(acc_right.len());
        if len == 0 || pcm_u8.is_empty() {
            return;
        }
        let vl = vol_left as u32;
        let vr = vol_right as u32;
        let base_pos = *pos as isize;
        let mut r7 = *frac as i32;

        for i in 0..len {
            let idx = (base_pos + (r7 >> 16) as isize) as usize;
            let s = pcm_u8[idx] as u32;
            acc_left[i] = acc_left[i].wrapping_add(((s * vl) >> 3) as u16);
            acc_right[i] = acc_right[i].wrapping_add(((s * vr) >> 3) as u16);
            r7 = r7.wrapping_add(inc_q16);
        }

        *pos = (base_pos + (r7 >> 16) as isize) as usize;
        *frac = (r7 & 0xFFFF) as u16;
    }

    /// Bit-exact Krawall ARM HQ linear-interpolated stereo voice mixer
    /// (`0x03000AC0` in *Franklin the Turtle* / `0x03000F30` in *Hugo - Bukkazoom!*).
    ///
    /// Reproduces the exact 32-bit packed ARM instruction sequence:
    /// ```text
    /// sub  sl, s1, s0
    /// mul  sl, frac, sl
    /// add  r8, s0, sl, lsr #16
    /// ...
    /// add  ip, r8, sb, lsl #16
    /// mul  lr, ip, vol
    /// bic  lr, lr, #0x70000
    /// add  acc, acc, lr, lsr #3
    /// ```
    /// including the exact logical-shift-right (`lsr #16`) carry into the upper halfword
    /// when `s1 < s0` and `frac > 0`.
    #[allow(clippy::too_many_arguments)]
    pub fn mix_hq_stereo(
        pcm_u8: &[u8],
        pos: &mut usize,
        frac: &mut u16,
        inc_q16: i32,
        vol_left: u8,
        vol_right: u8,
        acc_left: &mut [u16],
        acc_right: &mut [u16],
    ) {
        let len = acc_left.len().min(acc_right.len());
        if len == 0 || pcm_u8.len() < 2 {
            return;
        }
        let vl = vol_left as u32;
        let vr = vol_right as u32;
        let base_pos = *pos as isize;
        let mut r7 = *frac as i32;

        // Process pairs of frames identically to Krawall's 2x16-bit packed SIMD inner loop
        let pairs = len / 2;
        for p in 0..pairs {
            let idx0 = (base_pos + (r7 >> 16) as isize) as usize;
            let s0 = pcm_u8[idx0] as u32;
            let s1 = pcm_u8[idx0 + 1] as u32;
            let diff0 = s1.wrapping_sub(s0);
            let f0 = (r7 as u32) & 0xFFFF;
            let prod0 = f0.wrapping_mul(diff0);
            let r8 = s0.wrapping_add(prod0 >> 16);
            r7 = r7.wrapping_add(inc_q16);

            let idx1 = (base_pos + (r7 >> 16) as isize) as usize;
            let t0 = pcm_u8[idx1] as u32;
            let t1 = pcm_u8[idx1 + 1] as u32;
            let diff1 = t1.wrapping_sub(t0);
            let f1 = (r7 as u32) & 0xFFFF;
            let prod1 = f1.wrapping_mul(diff1);
            let sb = t0.wrapping_add(prod1 >> 16);
            r7 = r7.wrapping_add(inc_q16);

            let packed = r8.wrapping_add(sb << 16);

            let lr_l = (packed.wrapping_mul(vl) & !0x0007_0000) >> 3;
            let lr_r = (packed.wrapping_mul(vr) & !0x0007_0000) >> 3;

            let i0 = p * 2;
            let w_l = (acc_left[i0] as u32) | ((acc_left[i0 + 1] as u32) << 16);
            let w_l = w_l.wrapping_add(lr_l);
            acc_left[i0] = (w_l & 0xFFFF) as u16;
            acc_left[i0 + 1] = (w_l >> 16) as u16;

            let w_r = (acc_right[i0] as u32) | ((acc_right[i0 + 1] as u32) << 16);
            let w_r = w_r.wrapping_add(lr_r);
            acc_right[i0] = (w_r & 0xFFFF) as u16;
            acc_right[i0 + 1] = (w_r >> 16) as u16;
        }

        // Handle trailing odd frame if caller passes an odd-length slice
        if !len.is_multiple_of(2) {
            let idx = (base_pos + (r7 >> 16) as isize) as usize;
            let s0 = pcm_u8[idx] as u32;
            let s1 = pcm_u8[idx + 1] as u32;
            let diff = s1.wrapping_sub(s0);
            let f = (r7 as u32) & 0xFFFF;
            let r8 = s0.wrapping_add(f.wrapping_mul(diff) >> 16);
            r7 = r7.wrapping_add(inc_q16);
            let last = len - 1;
            acc_left[last] = acc_left[last].wrapping_add(((r8.wrapping_mul(vl)) >> 3) as u16);
            acc_right[last] = acc_right[last].wrapping_add(((r8.wrapping_mul(vr)) >> 3) as u16);
        }

        *pos = (base_pos + (r7 >> 16) as isize) as usize;
        *frac = (r7 & 0xFFFF) as u16;
    }

    /// Bit-exact Krawall ARM silence-fill routine (`0x03000BC4` in *Franklin the Turtle*).
    ///
    /// When a one-shot voice terminates mid-buffer, Krawall adds `(vol << 7) >> 3 == vol << 4`
    /// (the unsigned `0x80` silence bias) to the remaining accumulator slots so the final
    /// buffer-wide DC-bias subtraction cancels cleanly to `0`.
    pub fn fill_silence_bias(
        acc_left: &mut [u16],
        acc_right: &mut [u16],
        vol_left: u8,
        vol_right: u8,
    ) {
        let len = acc_left.len().min(acc_right.len());
        let bias_l = ((vol_left as u32) << 4) as u16;
        let bias_r = ((vol_right as u32) << 4) as u16;
        for i in 0..len {
            acc_left[i] = acc_left[i].wrapping_add(bias_l);
            acc_right[i] = acc_right[i].wrapping_add(bias_r);
        }
    }

    /// Bit-exact Krawall ARM DC-bias subtraction & DirectSound 8-bit LUT clamp routine
    /// (`0x03000C48` in *Franklin the Turtle* / `0x030010B8` in *Hugo - Bukkazoom!*).
    ///
    /// Subtracts `total_vol << 4`, arithmetically shifts right by `shift` (default `5`),
    /// and clamps to signed DirectSound 8-bit PCM `[-128, 127]`.
    pub fn clamp_u16_to_dma8(
        acc_left: &[u16],
        acc_right: &[u16],
        out_left: &mut [i8],
        out_right: &mut [i8],
        total_vol_left: u32,
        total_vol_right: u32,
        shift: u32,
    ) {
        let len = acc_left
            .len()
            .min(acc_right.len())
            .min(out_left.len())
            .min(out_right.len());
        let bias_l = (total_vol_left << 4) as i32;
        let bias_r = (total_vol_right << 4) as i32;
        for i in 0..len {
            let vl = ((acc_left[i] as i32) - bias_l) >> shift;
            let vr = ((acc_right[i] as i32) - bias_r) >> shift;
            out_left[i] = vl.clamp(-128, 127) as i8;
            out_right[i] = vr.clamp(-128, 127) as i8;
        }
    }

    /// High-level loop-aware & one-shot-aware Krawall voice mixer operating on a
    /// [`KrawallSampleView`] and [`KrawallVoice`].
    ///
    /// Mirrors Krawall's `kmix` loop-chunking dispatcher (`0x03000D5C`):
    /// - Splits output into 4-sample-aligned chunks before hitting `end` (or `loop_start`
    ///   for reverse ping-pong).
    /// - Wraps forward loops by `loop_length`, reverses `inc_q16` for ping-pong loops,
    ///   and calls [`Self::fill_silence_bias`] when a one-shot sample finishes mid-buffer.
    ///
    /// Returns `(vol_left_contrib, vol_right_contrib)` to add to the buffer's total DC-bias
    /// volumes passed to [`Self::clamp_u16_to_dma8`].
    pub fn mix_voice_view(
        sample: &KrawallSampleView<'_>,
        voice: &mut KrawallVoice,
        acc_left: &mut [u16],
        acc_right: &mut [u16],
    ) -> (u32, u32) {
        if !voice.is_active || sample.is_empty() {
            return (0, 0);
        }
        let total_frames = acc_left.len().min(acc_right.len());
        if total_frames == 0 {
            return (0, 0);
        }

        let buf = sample.pcm_with_tail();
        let pcm_len = sample.len();
        let loop_len = sample.loop_length as usize;
        let loop_start = pcm_len.saturating_sub(loop_len);
        let vl = voice.vol_left;
        let vr = voice.vol_right;

        let mut offset = 0usize;
        while offset < total_frames && voice.is_active {
            let remaining = total_frames - offset;
            if voice.inc_q16 >= 0 {
                let inc = voice.inc_q16 as u64;
                if voice.pos >= pcm_len {
                    match voice.loop_mode {
                        KrawallLoopMode::Forward if loop_len > 0 => {
                            voice.pos -= loop_len;
                            continue;
                        }
                        KrawallLoopMode::PingPong if loop_len > 0 => {
                            voice.pos = pcm_len.saturating_sub(1);
                            voice.inc_q16 = -voice.inc_q16;
                            continue;
                        }
                        _ => {
                            voice.is_active = false;
                            Self::fill_silence_bias(
                                &mut acc_left[offset..total_frames],
                                &mut acc_right[offset..total_frames],
                                vl,
                                vr,
                            );
                            break;
                        }
                    }
                }

                // Compute frames until reaching pcm_len
                let dist_q16 = (((pcm_len - voice.pos) as u64) << 16).saturating_sub(voice.frac as u64);
                let frames_to_end = if inc == 0 {
                    remaining
                } else {
                    (dist_q16.div_ceil(inc) as usize).max(1)
                };

                // Ensure we never read past buf.len() (accounting for +1 in HQ mode)
                let max_safe_idx = buf.len().saturating_sub(if voice.mix_mode == KrawallMixMode::HqLinear { 1 } else { 0 });
                let dist_buf_q16 = (((max_safe_idx.saturating_sub(voice.pos)) as u64) << 16)
                    .saturating_sub(voice.frac as u64);
                let max_safe_frames = dist_buf_q16
                    .checked_div(inc)
                    .map_or(remaining, |q| (q as usize).max(1));

                let chunk = remaining.min(frames_to_end).min(max_safe_frames);
                match voice.mix_mode {
                    KrawallMixMode::NearestNeighbor => Self::mix_nn_stereo(
                        buf,
                        &mut voice.pos,
                        &mut voice.frac,
                        voice.inc_q16,
                        vl,
                        vr,
                        &mut acc_left[offset..offset + chunk],
                        &mut acc_right[offset..offset + chunk],
                    ),
                    KrawallMixMode::HqLinear => Self::mix_hq_stereo(
                        buf,
                        &mut voice.pos,
                        &mut voice.frac,
                        voice.inc_q16,
                        vl,
                        vr,
                        &mut acc_left[offset..offset + chunk],
                        &mut acc_right[offset..offset + chunk],
                    ),
                }
                offset += chunk;

                if voice.pos >= pcm_len {
                    match voice.loop_mode {
                        KrawallLoopMode::Forward if loop_len > 0 => {
                            while voice.pos >= pcm_len {
                                voice.pos -= loop_len;
                            }
                        }
                        KrawallLoopMode::PingPong if loop_len > 0 => {
                            voice.pos = pcm_len.saturating_sub(1);
                            voice.inc_q16 = -voice.inc_q16;
                        }
                        _ => {
                            voice.is_active = false;
                            if offset < total_frames {
                                Self::fill_silence_bias(
                                    &mut acc_left[offset..total_frames],
                                    &mut acc_right[offset..total_frames],
                                    vl,
                                    vr,
                                );
                            }
                            break;
                        }
                    }
                }
            } else {
                // Reverse playback (PingPong backward leg)
                let inc_abs = voice.inc_q16.unsigned_abs() as u64;
                let dist_q16 = (((voice.pos.saturating_sub(loop_start)) as u64) << 16)
                    .saturating_add(voice.frac as u64);
                let frames_to_start = dist_q16
                    .checked_div(inc_abs)
                    .map_or(remaining, |q| (q as usize).max(1));
                let chunk = remaining.min(frames_to_start);
                match voice.mix_mode {
                    KrawallMixMode::NearestNeighbor => Self::mix_nn_stereo(
                        buf,
                        &mut voice.pos,
                        &mut voice.frac,
                        voice.inc_q16,
                        vl,
                        vr,
                        &mut acc_left[offset..offset + chunk],
                        &mut acc_right[offset..offset + chunk],
                    ),
                    KrawallMixMode::HqLinear => Self::mix_hq_stereo(
                        buf,
                        &mut voice.pos,
                        &mut voice.frac,
                        voice.inc_q16,
                        vl,
                        vr,
                        &mut acc_left[offset..offset + chunk],
                        &mut acc_right[offset..offset + chunk],
                    ),
                }
                offset += chunk;

                if voice.pos <= loop_start {
                    voice.pos = loop_start;
                    voice.inc_q16 = -voice.inc_q16;
                }
            }
        }

        (vl as u32, vr as u32)
    }

    /// Compute Pearson correlation coefficient between two signed 8-bit DMA streams.
    pub fn correlation(a: &[i8], b: &[i8]) -> f64 {
        let n = a.len().min(b.len());
        if n == 0 {
            return 0.0;
        }
        let mut sum_a = 0.0;
        let mut sum_b = 0.0;
        for i in 0..n {
            sum_a += a[i] as f64;
            sum_b += b[i] as f64;
        }
        let mean_a = sum_a / n as f64;
        let mean_b = sum_b / n as f64;

        let mut num = 0.0;
        let mut den_a = 0.0;
        let mut den_b = 0.0;
        for i in 0..n {
            let da = a[i] as f64 - mean_a;
            let db = b[i] as f64 - mean_b;
            num += da * db;
            den_a += da * da;
            den_b += db * db;
        }

        let denom = sqrt_f64(den_a * den_b);
        if denom < 1e-12 {
            if den_a < 1e-12 && den_b < 1e-12 {
                1.0
            } else {
                0.0
            }
        } else {
            num / denom
        }
    }

    /// Compute Root Mean Square Error (RMSE) between two signed 8-bit DMA streams.
    pub fn rmse(a: &[i8], b: &[i8]) -> f64 {
        let n = a.len().min(b.len());
        if n == 0 {
            return 0.0;
        }
        let mut sum_sq = 0.0;
        for i in 0..n {
            let diff = a[i] as f64 - b[i] as f64;
            sum_sq += diff * diff;
        }
        sqrt_f64(sum_sq / n as f64)
    }
}

/// High-fidelity 48,000 Hz `f32` software synthesizer with band-limited windowed-sinc
/// (Lanczos) interpolation and true equal-power stereo panning.
#[derive(Debug, Clone)]
pub struct StudioSynth {
    pub sample_rate: u32,
    /// Number of Lanczos kernel lobes (default `3`, giving a 6-tap band-limited kernel).
    pub sinc_lobes: usize,
}

impl Default for StudioSynth {
    fn default() -> Self {
        Self::new(48_000)
    }
}

impl StudioSynth {
    /// Create a new `StudioSynth` at the specified output sample rate (typically `48_000` Hz).
    pub fn new(sample_rate: u32) -> Self {
        Self {
            sample_rate,
            sinc_lobes: 3,
        }
    }

    /// Normalized `sinc(x) = sin(pi * x) / (pi * x)` computed in pure `#![no_std]` Rust.
    #[inline]
    fn sinc_f64(x: f64) -> f64 {
        let ax = if x < 0.0 { -x } else { x };
        if ax < 1e-9 {
            return 1.0;
        }
        let px = core::f64::consts::PI * x;
        sin_f64(px) / px
    }

    /// Lanczos-`a` band-limited interpolation weight.
    #[inline]
    fn lanczos(x: f64, a: f64) -> f64 {
        let ax = if x < 0.0 { -x } else { x };
        if ax >= a {
            0.0
        } else {
            Self::sinc_f64(x) * Self::sinc_f64(x / a)
        }
    }

    /// Render a Krawall voice into stereo `f32` buffers (accumulating) using band-limited
    /// Lanczos interpolation and equal-power stereo panning.
    ///
    /// `freq_hz` is the desired sample playback rate in Hz, `volume` is `0..=64`, and
    /// `pan` is `-64..=64` (`0` = center).
    #[allow(clippy::too_many_arguments)]
    pub fn render_voice_f32(
        &self,
        sample: &KrawallSampleView<'_>,
        pos_f64: &mut f64,
        freq_hz: f64,
        volume: u8,
        pan: i8,
        out_left: &mut [f32],
        out_right: &mut [f32],
    ) {
        let len = out_left.len().min(out_right.len());
        if len == 0 || sample.is_empty() || self.sample_rate == 0 {
            return;
        }

        let step = freq_hz / (self.sample_rate as f64);
        let gain = (volume.min(64) as f64) / 64.0;
        // Equal-power stereo pan law: pan in [-64, +64] -> theta in [0, pi/2]
        let pan_norm = ((pan.clamp(-64, 64) as f64) + 64.0) / 128.0;
        let angle = pan_norm * core::f64::consts::FRAC_PI_2;
        let gain_l = (gain * cos_f64(angle)) as f32;
        let gain_r = (gain * sin_f64(angle)) as f32;

        let pcm_len = sample.len();
        let loop_len = sample.loop_length as usize;
        let loop_start = pcm_len.saturating_sub(loop_len);
        let a = self.sinc_lobes.max(1) as isize;
        let a_f64 = a as f64;

        for i in 0..len {
            let p = *pos_f64;
            if p >= pcm_len as f64 {
                if sample.loop_mode == KrawallLoopMode::Forward && loop_len > 0 {
                    *pos_f64 = loop_start as f64 + ((p - loop_start as f64) % (loop_len as f64));
                } else {
                    break;
                }
            }
            let p = *pos_f64;
            let center = p as isize;
            let mut acc = 0.0f64;
            let mut wsum = 0.0f64;

            for k in (center - a + 1)..=(center + a) {
                let s_val = if k < 0 {
                    0.0
                } else if (k as usize) < pcm_len {
                    (sample.sample_i8(k as usize) as f64) / 128.0
                } else if sample.loop_mode == KrawallLoopMode::Forward && loop_len > 0 {
                    let wrapped = loop_start + (((k as usize) - loop_start) % loop_len);
                    (sample.sample_i8(wrapped) as f64) / 128.0
                } else {
                    0.0
                };
                let w = Self::lanczos(p - (k as f64), a_f64);
                acc += s_val * w;
                wsum += w;
            }

            let val = if wsum.abs() > 1e-9 {
                (acc / wsum) as f32
            } else {
                0.0
            };

            out_left[i] += val * gain_l;
            out_right[i] += val * gain_r;
            *pos_f64 += step;
        }
    }
}

/// `#![no_std]` sine helper (delegates to `f64::sin` under `std`, Taylor series under `no_std`).
#[inline]
fn sin_f64(x: f64) -> f64 {
    #[cfg(feature = "std")]
    {
        x.sin()
    }
    #[cfg(not(feature = "std"))]
    {
        // Range-reduce x into [-pi, pi]
        let tau = core::f64::consts::TAU;
        let mut r = x % tau;
        if r > core::f64::consts::PI {
            r -= tau;
        } else if r < -core::f64::consts::PI {
            r += tau;
        }
        // Reflect into [-pi/2, pi/2]
        if r > core::f64::consts::FRAC_PI_2 {
            r = core::f64::consts::PI - r;
        } else if r < -core::f64::consts::FRAC_PI_2 {
            r = -core::f64::consts::PI - r;
        }
        let r2 = r * r;
        // 7-term minimax/Taylor polynomial accurate to < 1e-11 on [-pi/2, pi/2]
        r * (1.0
            - r2 * (1.0 / 6.0
                - r2 * (1.0 / 120.0
                    - r2 * (1.0 / 5040.0
                        - r2 * (1.0 / 362_880.0
                            - r2 * (1.0 / 39_916_800.0 - r2 / 6_227_020_800.0))))))
    }
}

/// `#![no_std]` cosine helper.
#[inline]
fn cos_f64(x: f64) -> f64 {
    #[cfg(feature = "std")]
    {
        x.cos()
    }
    #[cfg(not(feature = "std"))]
    {
        sin_f64(x + core::f64::consts::FRAC_PI_2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sqrt_f64_accuracy() {
        assert_eq!(sqrt_f64(0.0), 0.0);
        assert!(sqrt_f64(-1.0).is_nan());
        assert!((sqrt_f64(4.0) - 2.0).abs() < 1e-12);
        assert!((sqrt_f64(2.0) - core::f64::consts::SQRT_2).abs() < 1e-12);
    }

    #[test]
    fn test_studio_synth_renders_stereo_f32() {
        let pcm = [0x80u8, 0xA0, 0xC0, 0xE0, 0xC0, 0xA0, 0x80, 0x60, 0x40, 0x20, 0x40, 0x60];
        let view = KrawallSampleView::from_unsigned_pcm(0, 16_384, &pcm);
        let synth = StudioSynth::new(48_000);
        let mut left = [0.0f32; 32];
        let mut right = [0.0f32; 32];
        let mut pos = 0.0;
        synth.render_voice_f32(&view, &mut pos, 16_384.0, 64, 0, &mut left, &mut right);

        // Center pan -> left == right, non-zero energy
        let mut energy = 0.0f32;
        for i in 0..32 {
            assert!((left[i] - right[i]).abs() < 1e-5);
            energy += left[i] * left[i];
        }
        assert!(energy > 0.1, "expected audible energy, got {energy}");
    }
}
