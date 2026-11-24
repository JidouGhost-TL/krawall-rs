//! Software synthesis engines for the Krawall GBA Tracker Sound Engine.
//!
//! Provides two synthesis pipelines:
//! - [`AccuracySynth`]: Bit-exact emulation of Krawall's ARM IWRAM mixer (verified
//!   byte-for-byte against headless `libmgba` ARM execution across all 3 commercial
//!   Krawall mixer revisions).
//! - [`StudioSynth`]: High-fidelity 48,000 Hz `f32` renderer with precomputed polyphase
//!   windowed-sinc (Lanczos-3) band-limited interpolation and true equal-power stereo panning.

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

use crate::freq::KRAWALL_MIX_RATE;
use crate::sample::{KrawallLoopMode, KrawallSampleView, KRAWALL_SILENCE_BYTE};
use crate::sequence::{KrawallMixMode, KrawallVoice};
use crate::version::KrawallVersion;

/// Default Krawall clamp right-shift (`asr #5`), mapping the 16-bit unsigned accumulator
/// minus DC bias (`total_vol << 4`) into signed 8-bit DirectSound DMA FIFO samples.
pub const KRAWALL_DEFAULT_CLAMP_SHIFT: u32 = 5;

/// Number of fractional polyphase steps in [`StudioSynth`]'s compile-time Lanczos-3 LUT.
const LANCZOS_PHASES: usize = 256;

/// Compile-time normalized 6-tap Lanczos-3 polyphase lookup table (`257 * 6 * 4 = 6,168` bytes).
const LANCZOS3_LUT: [[f32; 6]; LANCZOS_PHASES + 1] = build_lanczos3_lut();

/// Build the normalized 6-tap Lanczos-3 polyphase lookup table at compile time.
const fn build_lanczos3_lut() -> [[f32; 6]; LANCZOS_PHASES + 1] {
    let mut table = [[0.0f32; 6]; LANCZOS_PHASES + 1];
    let mut p = 0usize;
    while p <= LANCZOS_PHASES {
        let frac = (p as f64) / (LANCZOS_PHASES as f64);
        let mut weights = [0.0f64; 6];
        let mut sum = 0.0f64;
        let mut idx = 0usize;
        while idx < 6 {
            let k = (idx as i32) - 2; // taps -2..=+3
            let w = lanczos_kernel(frac - (k as f64), 3.0);
            weights[idx] = w;
            sum += w;
            idx += 1;
        }
        let inv_sum = if sum > 1e-12 { 1.0 / sum } else { 0.0 };
        let mut j = 0usize;
        while j < 6 {
            table[p][j] = (weights[j] * inv_sum) as f32;
            j += 1;
        }
        p += 1;
    }
    table
}

/// Compute square root of an `f64` value (`#![no_std]` compatible).
#[inline]
#[must_use]
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
/// silence-fill, and LUT saturation clamping routines across all 3 commercial revisions.
#[derive(Debug, Clone)]
pub struct AccuracySynth {
    pub sample_rate: u32,
    pub master_volume: u8,
    pub clamp_shift: u32,
    pub version: KrawallVersion,
    #[cfg(feature = "alloc")]
    pub voices: Vec<KrawallVoice>,
}

impl Default for AccuracySynth {
    fn default() -> Self {
        Self::new(KRAWALL_MIX_RATE)
    }
}

impl AccuracySynth {
    /// Create a new `AccuracySynth` configured for Krawall's native 16,384 Hz rate
    /// and [`KrawallVersion::Standard`].
    #[must_use]
    pub fn new(sample_rate: u32) -> Self {
        Self::with_version(sample_rate, KrawallVersion::Standard)
    }

    /// Create a new `AccuracySynth` configured for a specific [`KrawallVersion`] revision.
    #[must_use]
    pub fn with_version(sample_rate: u32, version: KrawallVersion) -> Self {
        Self {
            sample_rate,
            master_volume: 128,
            clamp_shift: KRAWALL_DEFAULT_CLAMP_SHIFT,
            version,
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

    /// Const-generic bit-exact nearest-neighbor stereo mixer kernel.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn mix_nn_stereo_impl<const SHIFT: u32>(
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
        let al = &mut acc_left[..len];
        let ar = &mut acc_right[..len];

        // 1:1 pitch fast path: contiguous slice iteration enables full SIMD auto-vectorization
        // with zero per-sample bounds checks. Guarded on `len <= 32767` so `r7` cannot wrap i32.
        if inc_q16 == 0x0001_0000 && len <= 32_767 {
            let start = *pos;
            if let Some(src) = pcm_u8.get(start..start + len) {
                for ((&b, l), r) in src.iter().zip(al.iter_mut()).zip(ar.iter_mut()) {
                    let s = b as u32;
                    *l = l.wrapping_add(((s * vl) >> SHIFT) as u16);
                    *r = r.wrapping_add(((s * vr) >> SHIFT) as u16);
                }
                *pos = start + len;
                return;
            }
        }

        let base_pos = *pos as isize;
        let mut r7 = *frac as i32;

        for (l, r) in al.iter_mut().zip(ar.iter_mut()) {
            let idx = base_pos.wrapping_add((r7 >> 16) as isize) as usize;
            let Some(&b) = pcm_u8.get(idx) else {
                break;
            };
            let s = b as u32;
            *l = l.wrapping_add(((s * vl) >> SHIFT) as u16);
            *r = r.wrapping_add(((s * vr) >> SHIFT) as u16);
            r7 = r7.wrapping_add(inc_q16);
        }

        *pos = base_pos.wrapping_add((r7 >> 16) as isize) as usize;
        *frac = (r7 & 0xFFFF) as u16;
    }

    /// Const-generic bit-exact HQ linear-interpolated stereo mixer kernel.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn mix_hq_stereo_impl<const SHIFT: u32, const BIC_MASK: u32>(
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
        let pairs_len = len & !1;
        let (l_pairs, _) = acc_left[..pairs_len].as_chunks_mut::<2>();
        let (r_pairs, _) = acc_right[..pairs_len].as_chunks_mut::<2>();
        for (l_pair, r_pair) in l_pairs.iter_mut().zip(r_pairs.iter_mut()) {
            let idx0 = base_pos.wrapping_add((r7 >> 16) as isize) as usize;
            let Some(p0) = idx0.checked_add(2).and_then(|e| pcm_u8.get(idx0..e)) else {
                break;
            };
            let s0 = p0[0] as u32;
            let s1 = p0[1] as u32;
            let diff0 = s1.wrapping_sub(s0);
            let f0 = (r7 as u32) & 0xFFFF;
            let r8 = s0.wrapping_add(f0.wrapping_mul(diff0) >> 16);
            r7 = r7.wrapping_add(inc_q16);

            let idx1 = base_pos.wrapping_add((r7 >> 16) as isize) as usize;
            let Some(p1) = idx1.checked_add(2).and_then(|e| pcm_u8.get(idx1..e)) else {
                break;
            };
            let t0 = p1[0] as u32;
            let t1 = p1[1] as u32;
            let diff1 = t1.wrapping_sub(t0);
            let f1 = (r7 as u32) & 0xFFFF;
            let sb = t0.wrapping_add(f1.wrapping_mul(diff1) >> 16);
            r7 = r7.wrapping_add(inc_q16);

            let packed = r8.wrapping_add(sb << 16);
            let lr_l = (packed.wrapping_mul(vl) & !BIC_MASK) >> SHIFT;
            let lr_r = (packed.wrapping_mul(vr) & !BIC_MASK) >> SHIFT;

            let w_l = ((l_pair[0] as u32) | ((l_pair[1] as u32) << 16)).wrapping_add(lr_l);
            l_pair[0] = (w_l & 0xFFFF) as u16;
            l_pair[1] = (w_l >> 16) as u16;

            let w_r = ((r_pair[0] as u32) | ((r_pair[1] as u32) << 16)).wrapping_add(lr_r);
            r_pair[0] = (w_r & 0xFFFF) as u16;
            r_pair[1] = (w_r >> 16) as u16;
        }

        // Handle trailing odd frame if caller passes an odd-length slice
        if !len.is_multiple_of(2) {
            let idx = base_pos.wrapping_add((r7 >> 16) as isize) as usize;
            if let Some(p0) = idx.checked_add(2).and_then(|e| pcm_u8.get(idx..e)) {
                let s0 = p0[0] as u32;
                let s1 = p0[1] as u32;
                let r8 = s0
                    .wrapping_add(((r7 as u32) & 0xFFFF).wrapping_mul(s1.wrapping_sub(s0)) >> 16);
                r7 = r7.wrapping_add(inc_q16);
                let last = len - 1;
                acc_left[last] =
                    acc_left[last].wrapping_add(((r8.wrapping_mul(vl)) >> SHIFT) as u16);
                acc_right[last] =
                    acc_right[last].wrapping_add(((r8.wrapping_mul(vr)) >> SHIFT) as u16);
            }
        }

        *pos = base_pos.wrapping_add((r7 >> 16) as isize) as usize;
        *frac = (r7 & 0xFFFF) as u16;
    }

    /// Bit-exact Krawall ARM nearest-neighbor stereo voice mixer
    /// (`0x03000838` in *Franklin the Turtle* / `0x03000CA8` in *Hugo - Bukkazoom!*).
    ///
    /// - Fetches unsigned 8-bit PCM (`ldrb`, `0x80` center) at `pos + (r7 >> 16)` (signed Q16.16).
    /// - Scales via `(s * vol) >> 3` (matching ARM 2x16-bit packed `mul` + `bic #0x70000` + `lsr #3`).
    /// - Accumulates into 16-bit unsigned stereo buffers `acc_left` and `acc_right`.
    /// - Updates `*pos` and `*frac` identically to the ARM epilogue (`add r5, r5, r7, asr #16; strh r7`).
    #[allow(clippy::too_many_arguments)]
    #[inline]
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
        Self::mix_nn_stereo_impl::<3>(
            pcm_u8, pos, frac, inc_q16, vol_left, vol_right, acc_left, acc_right,
        );
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
    #[inline]
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
        Self::mix_hq_stereo_impl::<3, 0x0007_0000>(
            pcm_u8, pos, frac, inc_q16, vol_left, vol_right, acc_left, acc_right,
        );
    }

    /// Bit-exact Krawall ARM silence-fill routine (`0x03000BC4` in *Franklin the Turtle*).
    ///
    /// When a one-shot voice terminates mid-buffer, Krawall adds `(vol << 7) >> 3 == vol << 4`
    /// (the unsigned `0x80` silence bias) to the remaining accumulator slots so the final
    /// buffer-wide DC-bias subtraction cancels cleanly to `0`.
    #[inline]
    pub fn fill_silence_bias(
        acc_left: &mut [u16],
        acc_right: &mut [u16],
        vol_left: u8,
        vol_right: u8,
    ) {
        Self::fill_silence_bias_versioned(
            KrawallVersion::Standard,
            acc_left,
            acc_right,
            vol_left,
            vol_right,
        );
    }

    /// Bit-exact Krawall ARM silence-fill routine parameterized by [`KrawallVersion`].
    ///
    /// Adds `vol << version.bias_shift()` (`vol << 5` for [`KrawallVersion::EarlyLsr2`],
    /// `vol << 4` for [`KrawallVersion::MidSignedUnpack`] and [`KrawallVersion::Standard`]).
    #[inline]
    pub fn fill_silence_bias_versioned(
        version: KrawallVersion,
        acc_left: &mut [u16],
        acc_right: &mut [u16],
        vol_left: u8,
        vol_right: u8,
    ) {
        let len = acc_left.len().min(acc_right.len());
        let shift = version.bias_shift();
        let bias_l = ((vol_left as u32) << shift) as u16;
        let bias_r = ((vol_right as u32) << shift) as u16;
        for x in &mut acc_left[..len] {
            *x = x.wrapping_add(bias_l);
        }
        for x in &mut acc_right[..len] {
            *x = x.wrapping_add(bias_r);
        }
    }

    /// Bit-exact Krawall ARM nearest-neighbor stereo voice mixer for any [`KrawallVersion`].
    ///
    /// - [`KrawallVersion::Standard`] & [`KrawallVersion::MidSignedUnpack`]: scales via `(s * vol) >> 3`.
    /// - [`KrawallVersion::EarlyLsr2`]: scales via `(s * vol) >> 2` (2x higher precision/gain).
    #[allow(clippy::too_many_arguments)]
    #[inline]
    pub fn mix_nn_stereo_versioned(
        version: KrawallVersion,
        pcm_u8: &[u8],
        pos: &mut usize,
        frac: &mut u16,
        inc_q16: i32,
        vol_left: u8,
        vol_right: u8,
        acc_left: &mut [u16],
        acc_right: &mut [u16],
    ) {
        match version {
            KrawallVersion::EarlyLsr2 => Self::mix_nn_stereo_impl::<2>(
                pcm_u8, pos, frac, inc_q16, vol_left, vol_right, acc_left, acc_right,
            ),
            KrawallVersion::MidSignedUnpack | KrawallVersion::Standard => {
                Self::mix_nn_stereo_impl::<3>(
                    pcm_u8, pos, frac, inc_q16, vol_left, vol_right, acc_left, acc_right,
                )
            }
        }
    }

    /// Bit-exact Krawall ARM HQ linear-interpolated stereo voice mixer for any [`KrawallVersion`].
    ///
    /// Applies the exact packed `bic` mask (`0x30000` for `EarlyLsr2`, `0x70000` for
    /// `MidSignedUnpack` / `Standard`) and right shift (`2` vs `3`).
    #[allow(clippy::too_many_arguments)]
    #[inline]
    pub fn mix_hq_stereo_versioned(
        version: KrawallVersion,
        pcm_u8: &[u8],
        pos: &mut usize,
        frac: &mut u16,
        inc_q16: i32,
        vol_left: u8,
        vol_right: u8,
        acc_left: &mut [u16],
        acc_right: &mut [u16],
    ) {
        match version {
            KrawallVersion::EarlyLsr2 => Self::mix_hq_stereo_impl::<2, 0x0003_0000>(
                pcm_u8, pos, frac, inc_q16, vol_left, vol_right, acc_left, acc_right,
            ),
            KrawallVersion::MidSignedUnpack | KrawallVersion::Standard => {
                Self::mix_hq_stereo_impl::<3, 0x0007_0000>(
                    pcm_u8, pos, frac, inc_q16, vol_left, vol_right, acc_left, acc_right,
                )
            }
        }
    }

    /// Bit-exact Krawall ARM DC-bias subtraction & DirectSound 8-bit LUT clamp routine
    /// parameterized by [`KrawallVersion`].
    ///
    /// - [`KrawallVersion::Standard`]: unsigned `u16` unpack (`lsr #16`), bias `total_vol << 4`.
    /// - [`KrawallVersion::MidSignedUnpack`]: signed `i16` unpack (`asr #16`), bias `total_vol << 4`.
    /// - [`KrawallVersion::EarlyLsr2`]: signed `i16` unpack (`asr #16`), bias `total_vol << 5`.
    #[allow(clippy::too_many_arguments)]
    #[inline]
    pub fn clamp_u16_to_dma8_versioned(
        version: KrawallVersion,
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
        let al = &acc_left[..len];
        let ar = &acc_right[..len];
        let ol = &mut out_left[..len];
        let or = &mut out_right[..len];
        let shift = shift.min(31);
        let bias_shift = version.bias_shift();
        let signed = version.uses_signed_clamp_unpack();
        // Algebraic identity: `(a as i16 as i32) - bias == ((a ^ 0x8000) as i32) - (bias + 32768)`.
        // Unifies signed and unsigned unpack into one branchless auto-vectorizable SIMD loop.
        let xor_mask: u16 = if signed { 0x8000 } else { 0 };
        let bias_adj: i32 = if signed { 32_768 } else { 0 };
        let bias_l = (total_vol_left.wrapping_shl(bias_shift) as i32).wrapping_add(bias_adj);
        let bias_r = (total_vol_right.wrapping_shl(bias_shift) as i32).wrapping_add(bias_adj);

        for i in 0..len {
            ol[i] = (((al[i] ^ xor_mask) as i32).wrapping_sub(bias_l) >> shift).clamp(-128, 127)
                as i8;
            or[i] = (((ar[i] ^ xor_mask) as i32).wrapping_sub(bias_r) >> shift).clamp(-128, 127)
                as i8;
        }
    }

    /// Bit-exact Krawall ARM DC-bias subtraction & DirectSound 8-bit LUT clamp routine
    /// (`0x03000C48` in *Franklin the Turtle* / `0x030010B8` in *Hugo - Bukkazoom!*).
    ///
    /// Subtracts `total_vol << 4`, arithmetically shifts right by `shift` (default `5`),
    /// and clamps to signed DirectSound 8-bit PCM `[-128, 127]`.
    #[inline]
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
        let al = &acc_left[..len];
        let ar = &acc_right[..len];
        let ol = &mut out_left[..len];
        let or = &mut out_right[..len];
        let shift = shift.min(31);
        let bias_l = total_vol_left.wrapping_shl(4) as i32;
        let bias_r = total_vol_right.wrapping_shl(4) as i32;
        for i in 0..len {
            ol[i] = ((al[i] as i32).wrapping_sub(bias_l) >> shift).clamp(-128, 127) as i8;
            or[i] = ((ar[i] as i32).wrapping_sub(bias_r) >> shift).clamp(-128, 127) as i8;
        }
    }

    /// High-level loop-aware & one-shot-aware Krawall voice mixer operating on a
    /// [`KrawallSampleView`] and [`KrawallVoice`] using [`KrawallVersion::Standard`].
    ///
    /// Mirrors Krawall's `kmix` loop-chunking dispatcher (`0x03000D5C`):
    /// - Splits output into chunks before hitting `end` (or `loop_start` for reverse ping-pong).
    /// - Wraps forward loops by `loop_length`, reverses `inc_q16` for ping-pong loops,
    ///   and calls [`Self::fill_silence_bias`] when a one-shot sample finishes mid-buffer.
    ///
    /// Returns `(vol_left_contrib, vol_right_contrib)` to add to the buffer's total DC-bias
    /// volumes passed to [`Self::clamp_u16_to_dma8`].
    #[inline]
    pub fn mix_voice_view(
        sample: &KrawallSampleView<'_>,
        voice: &mut KrawallVoice,
        acc_left: &mut [u16],
        acc_right: &mut [u16],
    ) -> (u32, u32) {
        Self::mix_voice_view_versioned(
            KrawallVersion::Standard,
            sample,
            voice,
            acc_left,
            acc_right,
        )
    }

    /// High-level loop-aware & one-shot-aware Krawall voice mixer parameterized by
    /// [`KrawallVersion`].
    ///
    /// Returns `(vol_left_contrib, vol_right_contrib)` to add to the buffer's total DC-bias
    /// volumes passed to [`Self::clamp_u16_to_dma8_versioned`].
    pub fn mix_voice_view_versioned(
        version: KrawallVersion,
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
                            voice.pos = loop_start + ((voice.pos - loop_start) % loop_len);
                            continue;
                        }
                        KrawallLoopMode::PingPong if loop_len > 0 => {
                            voice.pos = pcm_len.saturating_sub(1);
                            voice.inc_q16 = voice.inc_q16.saturating_neg();
                            continue;
                        }
                        _ => {
                            voice.is_active = false;
                            Self::fill_silence_bias_versioned(
                                version,
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
                let dist_q16 =
                    (((pcm_len - voice.pos) as u64) << 16).saturating_sub(voice.frac as u64);
                let frames_to_end = if inc == 0 {
                    remaining
                } else {
                    (dist_q16.div_ceil(inc) as usize).max(1)
                };

                // Ensure we never read past buf.len() (accounting for +1 in HQ mode)
                let max_safe_idx = buf
                    .len()
                    .saturating_sub(usize::from(voice.mix_mode == KrawallMixMode::HqLinear));
                let dist_buf_q16 = (((max_safe_idx.saturating_sub(voice.pos)) as u64) << 16)
                    .saturating_sub(voice.frac as u64);
                let max_safe_frames = dist_buf_q16
                    .checked_div(inc)
                    .map_or(remaining, |q| (q as usize).max(1));

                let chunk = remaining.min(frames_to_end).min(max_safe_frames);
                match voice.mix_mode {
                    KrawallMixMode::NearestNeighbor => Self::mix_nn_stereo_versioned(
                        version,
                        buf,
                        &mut voice.pos,
                        &mut voice.frac,
                        voice.inc_q16,
                        vl,
                        vr,
                        &mut acc_left[offset..offset + chunk],
                        &mut acc_right[offset..offset + chunk],
                    ),
                    KrawallMixMode::HqLinear => Self::mix_hq_stereo_versioned(
                        version,
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
                            voice.pos = loop_start + ((voice.pos - loop_start) % loop_len);
                        }
                        KrawallLoopMode::PingPong if loop_len > 0 => {
                            voice.pos = pcm_len.saturating_sub(1);
                            voice.inc_q16 = voice.inc_q16.saturating_neg();
                        }
                        _ => {
                            voice.is_active = false;
                            if offset < total_frames {
                                Self::fill_silence_bias_versioned(
                                    version,
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
                    KrawallMixMode::NearestNeighbor => Self::mix_nn_stereo_versioned(
                        version,
                        buf,
                        &mut voice.pos,
                        &mut voice.frac,
                        voice.inc_q16,
                        vl,
                        vr,
                        &mut acc_left[offset..offset + chunk],
                        &mut acc_right[offset..offset + chunk],
                    ),
                    KrawallMixMode::HqLinear => Self::mix_hq_stereo_versioned(
                        version,
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
                    voice.inc_q16 = voice.inc_q16.saturating_neg();
                }
            }
        }

        (vl as u32, vr as u32)
    }

    /// Compute Pearson correlation coefficient between two signed 8-bit DMA streams.
    #[must_use]
    pub fn correlation(a: &[i8], b: &[i8]) -> f64 {
        let n = a.len().min(b.len());
        if n == 0 {
            return 0.0;
        }
        let mut sum_a = 0.0;
        let mut sum_b = 0.0;
        for (&va, &vb) in a[..n].iter().zip(&b[..n]) {
            sum_a += va as f64;
            sum_b += vb as f64;
        }
        let mean_a = sum_a / n as f64;
        let mean_b = sum_b / n as f64;

        let mut num = 0.0;
        let mut den_a = 0.0;
        let mut den_b = 0.0;
        for (&va, &vb) in a[..n].iter().zip(&b[..n]) {
            let da = va as f64 - mean_a;
            let db = vb as f64 - mean_b;
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
    #[must_use]
    pub fn rmse(a: &[i8], b: &[i8]) -> f64 {
        let n = a.len().min(b.len());
        if n == 0 {
            return 0.0;
        }
        let mut sum_sq = 0.0;
        for (&va, &vb) in a[..n].iter().zip(&b[..n]) {
            let diff = va as f64 - vb as f64;
            sum_sq += diff * diff;
        }
        sqrt_f64(sum_sq / n as f64)
    }
}

/// High-fidelity 48,000 Hz `f32` software synthesizer with band-limited windowed-sinc
/// (Lanczos) interpolation and true equal-power stereo panning.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct StudioSynth {
    pub sample_rate: u32,
    /// Number of Lanczos kernel lobes (default `3`, using a compile-time 256-phase
    /// 6-tap polyphase LUT for maximum throughput).
    pub sinc_lobes: usize,
}

impl Default for StudioSynth {
    fn default() -> Self {
        Self::new(48_000)
    }
}

impl StudioSynth {
    /// Standard studio output sample rate (`48_000` Hz).
    pub const SAMPLE_RATE: u32 = 48_000;

    /// Create a new `StudioSynth` at the specified output sample rate (typically `48_000` Hz).
    #[must_use]
    pub fn new(sample_rate: u32) -> Self {
        Self {
            sample_rate,
            sinc_lobes: 3,
        }
    }

    /// Render a Krawall voice into stereo `f32` buffers (accumulating) using band-limited
    /// Lanczos interpolation and equal-power stereo panning.
    ///
    /// `freq_hz` is the desired sample playback rate in Hz, `volume` is `0..=64`, and
    /// `pan` is `-64..=64` (`0` = center).
    ///
    /// When `self.sinc_lobes == 3` (the default), uses a compile-time 256-phase 6-tap
    /// Lanczos-3 polyphase lookup table with a Q32.32 fixed-point interior fast path.
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
        let frames = out_left.len().min(out_right.len());
        if frames == 0
            || sample.is_empty()
            || self.sample_rate == 0
            || !freq_hz.is_finite()
            || freq_hz <= 0.0
            || !pos_f64.is_finite()
        {
            return;
        }
        if *pos_f64 < 0.0 {
            *pos_f64 = 0.0;
        }

        let step = freq_hz / (self.sample_rate as f64);
        let gain = (volume.min(64) as f64) / 64.0;
        // Equal-power stereo pan law: pan in [-64, +64] -> theta in [0, pi/2]
        let pan_norm = ((pan.clamp(-64, 64) as f64) + 64.0) / 128.0;
        let angle = pan_norm * core::f64::consts::FRAC_PI_2;
        // Fold `1 / 128` signed-8-bit normalization into stereo gains.
        let scale = gain * (1.0 / 128.0);
        let gain_l = (scale * cos_f64(angle)) as f32;
        let gain_r = (scale * sin_f64(angle)) as f32;

        let pcm = sample.pcm_u8;
        let pcm_len = pcm.len();
        let loop_len = sample.loop_length as usize;
        let looped = sample.loop_mode == KrawallLoopMode::Forward && loop_len > 0;
        let loop_start = pcm_len.saturating_sub(loop_len);
        let loop_start_f = loop_start as f64;
        let loop_len_f = loop_len as f64;
        let pcm_len_f = pcm_len as f64;

        let out_l = &mut out_left[..frames];
        let out_r = &mut out_right[..frames];

        if self.sinc_lobes == 3 {
            let safe_max_base = pcm_len.saturating_sub(4);
            let mut ph = *pos_f64;
            let mut idx = 0usize;

            while idx < frames {
                if ph >= pcm_len_f {
                    if looped {
                        ph -= loop_len_f;
                        if ph >= pcm_len_f {
                            ph = loop_start_f + ((ph - loop_start_f) % loop_len_f);
                        }
                        if ph < loop_start_f {
                            ph = loop_start_f;
                        }
                    } else {
                        break;
                    }
                }

                // Fast interior run: Q32.32 fixed-point phase with zero boundary checks or f64 casts.
                let base = ph as usize;
                if pcm_len >= 6 && base >= 2 && base <= safe_max_base && step < (1u64 << 31) as f64
                {
                    let safe_limit = (safe_max_base as f64) + 0.999_999;
                    let max_interior = (((safe_limit - ph) / step) as usize).saturating_add(1);
                    let run = max_interior.min(frames - idx);
                    if run >= 4 {
                        const Q32_SCALE: f64 = 4_294_967_296.0;
                        const INV_Q24: f32 = 1.0 / 16_777_216.0;
                        let mut ph_q32 = (ph * Q32_SCALE) as u64;
                        let step_q32 = (step * Q32_SCALE) as u64;
                        let last_base = ((ph_q32
                            .saturating_add(step_q32.saturating_mul((run - 1) as u64)))
                            >> 32) as usize;
                        if last_base <= safe_max_base {
                            for (ol, or) in out_l[idx..idx + run]
                                .iter_mut()
                                .zip(&mut out_r[idx..idx + run])
                            {
                                let b = (ph_q32 >> 32) as usize;
                                let p = ((ph_q32 >> 24) & 0xFF) as usize;
                                let t = ((ph_q32 & 0x00FF_FFFF) as f32) * INV_Q24;
                                let w0 = &LANCZOS3_LUT[p];
                                let w1 = &LANCZOS3_LUT[p + 1];
                                let win = &pcm[b - 2..b + 4];
                                let s0 = ((win[0] ^ KRAWALL_SILENCE_BYTE) as i8) as f32;
                                let s1 = ((win[1] ^ KRAWALL_SILENCE_BYTE) as i8) as f32;
                                let s2 = ((win[2] ^ KRAWALL_SILENCE_BYTE) as i8) as f32;
                                let s3 = ((win[3] ^ KRAWALL_SILENCE_BYTE) as i8) as f32;
                                let s4 = ((win[4] ^ KRAWALL_SILENCE_BYTE) as i8) as f32;
                                let s5 = ((win[5] ^ KRAWALL_SILENCE_BYTE) as i8) as f32;
                                let a0 = s0 * w0[0]
                                    + s1 * w0[1]
                                    + s2 * w0[2]
                                    + s3 * w0[3]
                                    + s4 * w0[4]
                                    + s5 * w0[5];
                                let a1 = s0 * w1[0]
                                    + s1 * w1[1]
                                    + s2 * w1[2]
                                    + s3 * w1[3]
                                    + s4 * w1[4]
                                    + s5 * w1[5];
                                let val = a0 + t * (a1 - a0);
                                *ol += val * gain_l;
                                *or += val * gain_r;
                                ph_q32 += step_q32;
                            }
                            ph += (run as f64) * step;
                            idx += run;
                            continue;
                        }
                    }
                }

                // Boundary / loop-edge single-sample step using the precomputed Lanczos-3 LUT
                let frac = ph - (base as f64);
                let idx_f = frac * (LANCZOS_PHASES as f64);
                let p = (idx_f as usize).min(LANCZOS_PHASES - 1);
                let t = (idx_f - (p as f64)) as f32;
                let w0 = &LANCZOS3_LUT[p];
                let w1 = &LANCZOS3_LUT[p + 1];

                let mut s = [0.0f32; 6];
                let base_i = base as isize;
                for (slot, k) in s.iter_mut().zip(-2isize..=3) {
                    let sample_idx = base_i + k;
                    if sample_idx >= 0 {
                        let u = sample_idx as usize;
                        if u < pcm_len {
                            *slot = ((pcm[u] ^ KRAWALL_SILENCE_BYTE) as i8) as f32;
                        } else if looped {
                            let wrapped = loop_start + ((u - loop_start) % loop_len);
                            *slot = ((pcm[wrapped] ^ KRAWALL_SILENCE_BYTE) as i8) as f32;
                        }
                    }
                }

                let a0 = s[0] * w0[0]
                    + s[1] * w0[1]
                    + s[2] * w0[2]
                    + s[3] * w0[3]
                    + s[4] * w0[4]
                    + s[5] * w0[5];
                let a1 = s[0] * w1[0]
                    + s[1] * w1[1]
                    + s[2] * w1[2]
                    + s[3] * w1[3]
                    + s[4] * w1[4]
                    + s[5] * w1[5];
                let val = a0 + t * (a1 - a0);

                out_l[idx] += val * gain_l;
                out_r[idx] += val * gain_r;
                ph += step;
                idx += 1;
            }

            *pos_f64 = ph;
            return;
        }

        // General fallback for custom `sinc_lobes != 3`
        let a = self.sinc_lobes.clamp(1, 64) as isize;
        let a_f64 = a as f64;
        let mut ph = *pos_f64;

        for (ol, or) in out_l.iter_mut().zip(out_r.iter_mut()) {
            if ph >= pcm_len_f {
                if looped {
                    ph = loop_start_f + ((ph - loop_start_f) % loop_len_f);
                } else {
                    break;
                }
            }
            let center = ph as isize;
            let mut acc = 0.0f64;
            let mut wsum = 0.0f64;

            for k in (center - a + 1)..=(center + a) {
                let s_val = if k < 0 {
                    0.0
                } else if (k as usize) < pcm_len {
                    ((pcm[k as usize] ^ KRAWALL_SILENCE_BYTE) as i8) as f64
                } else if looped {
                    let wrapped = loop_start + (((k as usize) - loop_start) % loop_len);
                    ((pcm[wrapped] ^ KRAWALL_SILENCE_BYTE) as i8) as f64
                } else {
                    0.0
                };
                let w = lanczos_kernel(ph - (k as f64), a_f64);
                acc += s_val * w;
                wsum += w;
            }

            let val = if wsum.abs() > 1e-9 {
                (acc / wsum) as f32
            } else {
                0.0
            };

            *ol += val * gain_l;
            *or += val * gain_r;
            ph += step;
        }

        *pos_f64 = ph;
    }
}

/// `#![no_std]` compile-time Lanczos windowed-sinc kernel `sinc(x) * sinc(x / a)`.
#[inline]
const fn lanczos_kernel(x: f64, a: f64) -> f64 {
    let ax = if x < 0.0 { -x } else { x };
    if ax < 1e-12 {
        return 1.0;
    }
    if ax >= a {
        return 0.0;
    }
    let pi_x = core::f64::consts::PI * x;
    let pi_x_a = pi_x / a;
    (sin_f64(pi_x) / pi_x) * (sin_f64(pi_x_a) / pi_x_a)
}

/// Pure `#![no_std]` `const fn` sine helper using range reduction + 13th-order Taylor polynomial.
#[inline]
const fn sin_f64(mut x: f64) -> f64 {
    const TWO_PI: f64 = core::f64::consts::TAU;
    const PI: f64 = core::f64::consts::PI;
    const HALF_PI: f64 = core::f64::consts::FRAC_PI_2;

    // Range-reduce x into [-pi, pi]
    let k = (x / TWO_PI) as i64;
    x -= (k as f64) * TWO_PI;
    if x > PI {
        x -= TWO_PI;
    } else if x < -PI {
        x += TWO_PI;
    }
    // Reflect into [-pi/2, pi/2]
    if x > HALF_PI {
        x = PI - x;
    } else if x < -HALF_PI {
        x = -PI - x;
    }
    let r2 = x * x;
    x * (1.0
        - r2 * (1.0 / 6.0
            - r2 * (1.0 / 120.0
                - r2 * (1.0 / 5_040.0
                    - r2 * (1.0 / 362_880.0
                        - r2 * (1.0 / 39_916_800.0 - r2 / 6_227_020_800.0))))))
}

/// Pure `#![no_std]` `const fn` cosine helper.
#[inline]
const fn cos_f64(x: f64) -> f64 {
    sin_f64(x + core::f64::consts::FRAC_PI_2)
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

    #[test]
    fn test_studio_synth_lut_matches_direct_lanczos3() {
        let pcm: [u8; 64] = core::array::from_fn(|i| {
            let angle = (i as f64) * 0.35;
            ((sin_f64(angle) * 100.0) as i8 as u8) ^ KRAWALL_SILENCE_BYTE
        });
        let view = KrawallSampleView::from_unsigned_pcm(0, 16_384, &pcm);
        let synth_lut = StudioSynth::new(48_000);
        let mut left = [0.0f32; 128];
        let mut right = [0.0f32; 128];
        let mut pos = 5.25;
        synth_lut.render_voice_f32(&view, &mut pos, 12_000.0, 64, 0, &mut left, &mut right);

        // Compare against direct Lanczos-3 reference on interior samples
        let step = 12_000.0 / 48_000.0;
        let gain_l = ((1.0 / 128.0) * cos_f64(core::f64::consts::FRAC_PI_4)) as f32;
        let mut ref_pos = 5.25f64;
        for &actual in &left {
            let center = ref_pos as isize;
            let mut acc = 0.0f64;
            let mut wsum = 0.0f64;
            for k in (center - 2)..=(center + 3) {
                let s = ((pcm[k as usize] ^ KRAWALL_SILENCE_BYTE) as i8) as f64;
                let w = lanczos_kernel(ref_pos - (k as f64), 3.0);
                acc += s * w;
                wsum += w;
            }
            let expected = ((acc / wsum) as f32) * gain_l;
            assert!(
                (actual - expected).abs() < 1e-4,
                "LUT mismatch: actual={actual}, expected={expected}"
            );
            ref_pos += step;
        }
    }

    #[test]
    fn test_early_lsr2_oneshot_silence_tail_cancels_to_zero() {
        let pcm = [0x80u8, 0x90, 0xA0, 0x90];
        let view = KrawallSampleView::from_unsigned_pcm(0, 16_384, &pcm);
        let mut voice = KrawallVoice::with_raw_params(
            0,
            0,
            0,
            0x0001_0000,
            40,
            24,
            KrawallLoopMode::None,
            KrawallMixMode::NearestNeighbor,
        );
        let mut acc_l = [0u16; 16];
        let mut acc_r = [0u16; 16];
        let mut dma_l = [0i8; 16];
        let mut dma_r = [0i8; 16];

        let (tot_l, tot_r) = AccuracySynth::mix_voice_view_versioned(
            KrawallVersion::EarlyLsr2,
            &view,
            &mut voice,
            &mut acc_l,
            &mut acc_r,
        );
        AccuracySynth::clamp_u16_to_dma8_versioned(
            KrawallVersion::EarlyLsr2,
            &acc_l,
            &acc_r,
            &mut dma_l,
            &mut dma_r,
            tot_l,
            tot_r,
            KRAWALL_DEFAULT_CLAMP_SHIFT,
        );
        assert!(!voice.is_active);
        assert_eq!(&dma_l[4..], &[0i8; 12]);
        assert_eq!(&dma_r[4..], &[0i8; 12]);
    }

    #[test]
    fn test_malformed_inputs_do_not_panic() {
        let pcm = [0x80u8, 0x90, 0xA0, 0x90, 0x80, 0x70];
        let mut acc_l = [0u16; 8];
        let mut acc_r = [0u16; 8];
        let mut dma_l = [0i8; 8];
        let mut dma_r = [0i8; 8];

        // OOB start pos in NN & HQ must not panic
        let mut pos = 999usize;
        let mut frac = 0u16;
        AccuracySynth::mix_nn_stereo(&pcm, &mut pos, &mut frac, 0x0001_0000, 64, 64, &mut acc_l, &mut acc_r);
        AccuracySynth::mix_hq_stereo(&pcm, &mut pos, &mut frac, 0x0001_0000, 64, 64, &mut acc_l, &mut acc_r);

        // Excessive shift & total_vol must not panic
        AccuracySynth::clamp_u16_to_dma8(&acc_l, &acc_r, &mut dma_l, &mut dma_r, u32::MAX, u32::MAX, 99);
        AccuracySynth::clamp_u16_to_dma8_versioned(
            KrawallVersion::EarlyLsr2,
            &acc_l,
            &acc_r,
            &mut dma_l,
            &mut dma_r,
            u32::MAX,
            u32::MAX,
            99,
        );

        // Tiny frequency step in StudioSynth must not overflow usize
        let view = KrawallSampleView::from_unsigned_pcm(0, 16_384, &pcm);
        let synth = StudioSynth::new(48_000);
        let mut l = [0.0f32; 8];
        let mut r = [0.0f32; 8];
        let mut pf = 2.0;
        synth.render_voice_f32(&view, &mut pf, 1e-300, 64, 0, &mut l, &mut r);
        assert!(l.iter().all(|x| x.is_finite()));
    }
}
