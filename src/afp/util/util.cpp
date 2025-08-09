#include "afp/util/util.hpp"

namespace afp::util {
std::string_view error_name(UtilError e) noexcept {
  switch (e) {
    case UtilError::None:
      return "None";
    case UtilError::InvalidArgument:
      return "InvalidArgument";
    case UtilError::SizeMismatch:
      return "SizeMismatch";
    case UtilError::AlignmentError:
      return "AlignmentError";
    case UtilError::OutOfMemory:
      return "OutOfMemory";
    case UtilError::IOError:
      return "IOError";
    case UtilError::DecodeError:
      return "DecodeError";
    case UtilError::UnsupportedFormat:
      return "UnsupportedFormat";
    case UtilError::DspError:
      return "DspError";
    case UtilError::IndexCorrupt:
      return "IndexCorrupt";
    case UtilError::NotFound:
      return "NotFound";
    case UtilError::Unavailable:
      return "Unavailable";
    case UtilError::Timeout:
      return "Timeout";
    case UtilError::Internal:
      return "Internal";
    case UtilError::ResourceExhausted:
      return "ResourceExhausted";
  }
  return "Unknown";
}

std::string_view error_description(UtilError e) noexcept {
  switch (e) {
    case UtilError::None:
      return "Success";
    case UtilError::InvalidArgument:
      return "An input argument violated preconditions.";
    case UtilError::SizeMismatch:
      return "Buffer or shape sizes do not match.";
    case UtilError::AlignmentError:
      return "Buffer alignment unsuitable for SIMD/KFR assumptions.";
    case UtilError::OutOfMemory:
      return "Allocation failed due to insufficient memory.";
    case UtilError::IOError:
      return "Underlying I/O operation failed.";
    case UtilError::DecodeError:
      return "Data could not be decoded or parsed.";
    case UtilError::UnsupportedFormat:
      return "Requested format or feature is not supported.";
    case UtilError::DspError:
      return "DSP backend reported a failure.";
    case UtilError::IndexCorrupt:
      return "Index storage appears corrupt or inconsistent.";
    case UtilError::NotFound:
      return "Requested item does not exist.";
    case UtilError::Unavailable:
      return "Subsystem not initialized or temporarily unavailable.";
    case UtilError::Timeout:
      return "Operation exceeded allowed time.";
    case UtilError::Internal:
      return "An internal invariant was violated (bug).";
    case UtilError::ResourceExhausted:
      return "Resource exhausted or unavailable.";
  }
  return "Unknown error";
}
} // namespace afp::util
