# krawall-rs

**Zero-dependency, pure Rust parser, software synthesizer, and byte-exact golden-tested emulator for the Krawall GBA Tracker Sound Engine (Game Boy Advance).**

[![Crates.io](https://img.shields.io/crates/v/krawall.svg)](https://crates.io/crates/krawall)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](#license)
[![no_std](https://img.shields.io/badge/no__std-compatible-success.svg)](#features)

The **Krawall Sound Engine** was created by **Sebastian Kienzl** as a high-performance, ARM IWRAM-accelerated tracker music and SFX engine for the Game Boy Advance. Running at a power-of-two **16,384 Hz** hardware DirectSound mixing rate (`REG_TM0CNT_L = 0xFC00`), Krawall powered over 100 commercial GBA titles including *The Sims: Bustin' Out*, *The Sims 2*, *LEGO Star Wars: The Video Game*, *Franklin the Turtle*, and *Hugo - Bukkazoom!*.

`krawall-rs` provides a clean-room, reverse-engineered implementation designed for audio preservation, DAWs, PC source ports, and emulators.

> **Looking for a ready-to-use CLI?** You can scan GBA ROMs, extract WAV samples, and compile SoundFont 2.04 (.sf2) banks directly from the command line using our **[gbakit](https://github.com/JidouGhost-TL/gbakit)** toolkit (`gbakit audio scan`, `gbakit audio export`, `gbakit audio soundfont`).

---

## Features

- **Zero External Dependencies**: Pure Rust with no dependencies in `[dependencies]`, `#![deny(unsafe_code)]`.
- **`#![no_std]` Compatible**: Ready for embedded targets and WASM runtimes (`alloc` optional).
- **Native Krawall Sample Parser**: Zero-allocation `KrawallSampleView` parsing Krawall's 18-byte `Sample` header, inline unsigned 8-bit PCM (`0x80` silence center), and 64-byte `mconv` unrolled loop / silence tail.
- **Dual Synthesis Engines**:
  - **`AccuracySynth`**: Bit-exact emulation of Krawall's 16,384 Hz ARM IWRAM mixer — both **Nearest-Neighbor** (`0x03000838`) and **HQ Linear-Interpolated** (`0x03000AC0`) kernels, Q16.16 `freq << 2` phase stepping, 2x16-bit packed `(s * vol) >> 3` accumulation, mid-buffer silence-bias fill (`0x03000BC4`), and DC-bias subtraction + DirectSound 8-bit LUT clamping (`0x03000C48`).
  - **`StudioSynth`**: Renders at 48,000 Hz, 32-bit floating-point precision with band-limited Lanczos windowed-sinc interpolation and true equal-power stereo panning.
- **Byte-Exact mGBA ARM Golden Tested**: 3,072 / 3,072 output bytes verified strictly byte-for-byte (`assert_eq!`) across all 3 distinct Krawall ARM mixer revisions discovered in the 3,377-ROM GBA corpus (`EarlyLsr2`, `MidSignedUnpack`, and `Standard`).

---

## Supported Krawall Engine Revisions

Auditing all 129 Krawall ROMs across the full 3,377-ROM GBA corpus identified exactly three ARM IWRAM mixer revisions, all supported and byte-exact golden-tested via `KrawallVersion`:

| Revision | Era / Corpus Share | Mix Scaling | DC Bias | Clamp Unpack | Channels |
| --- | --- | --- | --- | --- | --- |
| `KrawallVersion::EarlyLsr2` | Early 2002 (4 ROMs) | `(s * vol) >> 2` (`bic #0x30000`) | `total_vol << 5` | Signed `i16` (`asr #16`) | 16 |
| `KrawallVersion::MidSignedUnpack` | 2002–2003 (5 ROMs, e.g. *Racing Fever*, *GP-1 Racing*) | `(s * vol) >> 3` (`bic #0x70000`) | `total_vol << 4` | Signed `i16` (`asr #16`) | 32 |
| `KrawallVersion::Standard` | 2003–2007 (120 ROMs, e.g. *Franklin the Turtle*, *Hugo - Bukkazoom!*, *The Sims*, *LEGO Star Wars*, *Crash of the Titans*) | `(s * vol) >> 3` (`bic #0x70000`) | `total_vol << 4` | Unsigned `u16` (`lsr #16`) | 32 |

---

## Quickstart

### Zero-Allocation Mixing (`#![no_std]`)

```rust
use krawall::{
    AccuracySynth, KrawallSampleView, KrawallVoice,
    KRAWALL_DEFAULT_CLAMP_SHIFT, KRAWALL_MIX_RATE,
};

// Wrap unsigned 8-bit PCM (0x80 center) directly without heap allocation
let pcm_u8: &[u8] = &[0x80, 0xA0, 0xC0, 0xE0, 0xC0, 0xA0, 0x80, 0x60];
let sample = KrawallSampleView::from_unsigned_pcm(0, KRAWALL_MIX_RATE, pcm_u8);

// 16,384 Hz (1:1 pitch), full volume (64), centered pan (0), full master (128)
let mut voice = KrawallVoice::new(0, KRAWALL_MIX_RATE, 64, 0, 128);
let mut acc_l = [0u16; 64];
let mut acc_r = [0u16; 64];
let mut dma_l = [0i8; 64];
let mut dma_r = [0i8; 64];

let (tot_l, tot_r) = AccuracySynth::mix_voice_view(&sample, &mut voice, &mut acc_l, &mut acc_r);
AccuracySynth::clamp_u16_to_dma8(
    &acc_l,
    &acc_r,
    &mut dma_l,
    &mut dma_r,
    tot_l,
    tot_r,
    KRAWALL_DEFAULT_CLAMP_SHIFT,
);
```

---

## Golden Test Methodology

Every mixing path in `AccuracySynth` is verified byte-for-byte against real ARM execution traces captured in headless `libmgba`:

1. **Tier 1 (Native Sample Format)**: Real Krawall 18-byte headers, unsigned `0x80` PCM continuity, and 64-byte `mconv` unrolled loop/silence tails from *Franklin the Turtle* and *Hugo - Bukkazoom!*.
2. **Tier 2 (Timer & Fixed-Point Math)**: GBA Timer 0 reload (`0xFC00` -> 16,384 Hz), Q16.16 `freq << 2` step, and exact ARM stereo volume/panning scaling.
3. **Tier 3 (3,072 / 3,072 Byte-Exact ARM Mixer Goldens)**: Strict `assert_eq!` across 12 controlled ARM execution vectors covering all 3 Krawall mixer revisions (`Standard`, `MidSignedUnpack`, and `EarlyLsr2`): Nearest-Neighbor 1:1, Nearest-Neighbor resampled, HQ Linear-Interpolated resampled (including ARM's exact `lsr #16` halfword carry), forward loop wrap across the 64-byte unrolled tail, 6-voice `+127`/`-128` saturated LUT clamping, one-shot mid-buffer silence-bias fill, *Racing Fever* signed-unpack clamping, *EarlyLsr2* `lsr #2` NN/HQ mixing, and cross-title verification on *Hugo - Bukkazoom!*.

---

## Benchmarks

```sh
cargo bench --bench bench_audio
```

| Benchmark | Throughput |
| --- | ---: |
| `AccuracySynth::mix_nn_stereo` (1:1, 1,024f) | ~8.15 GSamples/s |
| `AccuracySynth::mix_nn_stereo` (resampled, 1,024f) | ~1.40 GSamples/s |
| `AccuracySynth::mix_hq_stereo` (resampled, 1,024f) | ~1.02 GSamples/s |
| `AccuracySynth::clamp_u16_to_dma8` (`Standard`, 1,024f) | ~5.95 GSamples/s |
| `AccuracySynth::mix_voice_view` (NN 1:1, 1,024f) | ~6.99 GSamples/s |
| `StudioSynth::render_voice_f32` (Lanczos-3 48 kHz, 1,024f) | ~216 MSamples/s |

---

## Community & Support

Come discuss the wonders of GBA RE, modding, etc. — also the best way to report issues or bugs, or ask for help on your own project: https://discord.gg/khwxFSVNrS

---

## License

Licensed under either of:
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))
