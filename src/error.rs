//! Error types for Krawall parsing and synthesis.

use core::fmt;

/// Errors returned when parsing Krawall sample headers, modules, or audio streams.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Error {
    /// Input buffer ended before a complete header or payload could be read.
    UnexpectedEof,
    /// Sample header field failed structural validation.
    InvalidHeader(&'static str),
    /// Loop length or loop bounds exceeded the sample length.
    InvalidLoopRange { loop_length: u32, pcm_length: usize },
    /// Sample index was not found in the active bank.
    InvalidSample(u32),
    /// Pattern or order index was out of range.
    InvalidPattern(u32),
    /// Corrupted or unsupported Krawall stream data.
    CorruptedData(&'static str),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnexpectedEof => write!(f, "Unexpected end of data while parsing Krawall stream"),
            Self::InvalidHeader(msg) => write!(f, "Invalid Krawall sample header: {msg}"),
            Self::InvalidLoopRange {
                loop_length,
                pcm_length,
            } => write!(
                f,
                "Invalid Krawall loop length {loop_length} for {pcm_length}-byte sample"
            ),
            Self::InvalidSample(id) => write!(f, "Invalid Krawall sample index {id}"),
            Self::InvalidPattern(id) => write!(f, "Invalid Krawall pattern index {id}"),
            Self::CorruptedData(msg) => write!(f, "Corrupted Krawall data: {msg}"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for Error {}

/// Convenience result alias for `krawall`.
pub type Result<T> = core::result::Result<T, Error>;
