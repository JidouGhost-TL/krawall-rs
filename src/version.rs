//! Krawall engine revision detection and mixer variant metadata.
//!
//! Auditing all 129 Krawall ROMs across the 3,377-title GBA corpus reveals exactly three
//! distinct ARM IWRAM mixer revisions:
//!
//! 1. [`KrawallVersion::EarlyLsr2`] (2002 / early builds, 4 ROMs):
//!    16 channels, unsigned phase (`lsr #16`), 2-bit right-shift scaling (`(s * vol) >> 2`,
//!    `bic #0x30000`), DC bias `total_vol << 5`, signed 16-bit halfword unpack in clamp.
//! 2. [`KrawallVersion::MidSignedUnpack`] (2002–2003 transitional builds, 5 ROMs, e.g.
//!    *Racing Fever*, *GP-1 Racing*):
//!    signed phase (`asr #16`), 3-bit right-shift scaling (`(s * vol) >> 3`, `bic #0x70000`),
//!    DC bias `total_vol << 4`, signed 16-bit halfword unpack (`asr #16`) in clamp.
//! 3. [`KrawallVersion::Standard`] (2003–2007 standard builds, 120 ROMs, e.g. *Franklin the
//!    Turtle*, *Hugo - Bukkazoom!*, *The Sims: Bustin' Out*, *LEGO Star Wars*, *Crash of the
//!    Titans*):
//!    32 channels, signed phase (`asr #16`), 3-bit right-shift scaling (`(s * vol) >> 3`,
//!    `bic #0x70000`), DC bias `total_vol << 4`, unsigned 16-bit halfword unpack (`lsr #16`)
//!    and dynamic shift patching in clamp.

/// Distinct Krawall ARM IWRAM mixer revisions found across GBA ROMs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KrawallVersion {
    /// Early 2002 Krawall (`lsr #2` sample scaling, `bic #0x30000`, `bias = total_vol << 5`,
    /// signed 16-bit unpack clamp, 16 channels, unsigned `lsr #16` phase).
    EarlyLsr2,
    /// Transitional 2002–2003 Krawall (`lsr #3` sample scaling, `bic #0x70000`,
    /// `bias = total_vol << 4`, signed 16-bit unpack clamp, signed `asr #16` phase).
    MidSignedUnpack,
    /// Standard 2003–2007 Krawall (`lsr #3` sample scaling, `bic #0x70000`,
    /// `bias = total_vol << 4`, unsigned 16-bit unpack clamp, 32 channels, signed `asr #16` phase).
    #[default]
    Standard,
}

impl KrawallVersion {
    /// Per-sample multiplication right-shift (`2` for `EarlyLsr2`, `3` for `MidSignedUnpack` / `Standard`).
    #[inline]
    pub const fn mix_shift(self) -> u32 {
        match self {
            Self::EarlyLsr2 => 2,
            Self::MidSignedUnpack | Self::Standard => 3,
        }
    }

    /// ARM 2x16-bit packed `bic` mask preventing low-bit bleed across halfwords during `lsr`.
    #[inline]
    pub const fn packed_bic_mask(self) -> u32 {
        match self {
            Self::EarlyLsr2 => 0x0003_0000,
            Self::MidSignedUnpack | Self::Standard => 0x0007_0000,
        }
    }

    /// Left-shift applied to `total_vol` to form the unsigned `0x80` DC silence bias
    /// (`(vol << 7) >> mix_shift`, i.e. `5` for `EarlyLsr2`, `4` for `MidSignedUnpack` / `Standard`).
    #[inline]
    pub const fn bias_shift(self) -> u32 {
        7 - self.mix_shift()
    }

    /// Whether the ARM clamp routine unpacks 16-bit accumulator halfwords as signed `i16`
    /// (`asr #16` / `lsl #16; asr #16`) rather than unsigned `u16` (`lsr #16`).
    #[inline]
    pub const fn uses_signed_clamp_unpack(self) -> bool {
        matches!(self, Self::EarlyLsr2 | Self::MidSignedUnpack)
    }

    /// Maximum simultaneous hardware mixer channels for this Krawall build.
    #[inline]
    pub const fn max_channels(self) -> u8 {
        match self {
            Self::EarlyLsr2 => 16,
            Self::MidSignedUnpack | Self::Standard => 32,
        }
    }

    /// Detect the Krawall mixer revision directly from a GBA ROM or IWRAM image by scanning
    /// for the canonical Krawall ARM mixer prologue and inspecting its scaling/clamp opcodes.
    pub fn detect_from_bytes(data: &[u8]) -> Option<Self> {
        // Canonical Krawall ARM mixer prologue:
        //   ldr  r5, [r0, #8]     (0xE5905008)
        //   ldr  r6, [r0, #0x14]  (0xE5906014)
        //   ldrh r7, [r0, #0x1c]  (0xE1D071BC)
        const PROLOGUE: [u8; 12] = [
            0x08, 0x50, 0x90, 0xE5, 0x14, 0x60, 0x90, 0xE5, 0xBC, 0x71, 0xD0, 0xE1,
        ];
        // `bic ..., #0x30000` (EarlyLsr2) vs `bic ..., #0x70000` (Mid/Standard)
        let pos = data.windows(PROLOGUE.len()).position(|w| w == PROLOGUE)?;
        let block_end = (pos + 0x640).min(data.len());
        let block = &data[pos..block_end];

        // Check for `bic rX, rY, #0x30000` (immediate 0x03 rotated right by 16 -> 0x0803)
        let (words, _) = block.as_chunks::<4>();
        for &chunk in words {
            let word = u32::from_le_bytes(chunk);
            if (word & 0xFFF0_0FFF) == 0xE3C0_0803 {
                return Some(Self::EarlyLsr2);
            }
        }

        // Distinguish MidSignedUnpack (`asr sl, r4, #16` = 0xE1A0A844) vs Standard (`lsr sl, r4, #16` = 0xE1A0A824)
        for &chunk in words {
            let word = u32::from_le_bytes(chunk);
            if word == 0xE1A0_A844 {
                return Some(Self::MidSignedUnpack);
            }
            if word == 0xE1A0_A824 {
                return Some(Self::Standard);
            }
        }

        Some(Self::Standard)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version_parameters() {
        assert_eq!(KrawallVersion::EarlyLsr2.mix_shift(), 2);
        assert_eq!(KrawallVersion::EarlyLsr2.bias_shift(), 5);
        assert_eq!(KrawallVersion::EarlyLsr2.packed_bic_mask(), 0x0003_0000);
        assert!(KrawallVersion::EarlyLsr2.uses_signed_clamp_unpack());
        assert_eq!(KrawallVersion::EarlyLsr2.max_channels(), 16);

        assert_eq!(KrawallVersion::MidSignedUnpack.mix_shift(), 3);
        assert_eq!(KrawallVersion::MidSignedUnpack.bias_shift(), 4);
        assert!(KrawallVersion::MidSignedUnpack.uses_signed_clamp_unpack());

        assert_eq!(KrawallVersion::Standard.mix_shift(), 3);
        assert_eq!(KrawallVersion::Standard.bias_shift(), 4);
        assert!(!KrawallVersion::Standard.uses_signed_clamp_unpack());
        assert_eq!(KrawallVersion::Standard.max_channels(), 32);
    }

    #[test]
    fn test_detect_from_opcodes() {
        let mut standard = [0u8; 24];
        standard[0..12].copy_from_slice(&[
            0x08, 0x50, 0x90, 0xE5, 0x14, 0x60, 0x90, 0xE5, 0xBC, 0x71, 0xD0, 0xE1,
        ]);
        standard[16..20].copy_from_slice(&0xE1A0_A824u32.to_le_bytes());
        assert_eq!(
            KrawallVersion::detect_from_bytes(&standard),
            Some(KrawallVersion::Standard)
        );

        let mut mid = standard;
        mid[16..20].copy_from_slice(&0xE1A0_A844u32.to_le_bytes());
        assert_eq!(
            KrawallVersion::detect_from_bytes(&mid),
            Some(KrawallVersion::MidSignedUnpack)
        );

        let mut early = standard;
        early[12..16].copy_from_slice(&0xE3C8_A803u32.to_le_bytes());
        assert_eq!(
            KrawallVersion::detect_from_bytes(&early),
            Some(KrawallVersion::EarlyLsr2)
        );
    }
}
