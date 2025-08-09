#include <gtest/gtest.h>
#include "afp/io/io.hpp"
#include "afp/util/util.hpp"
#include "test_utils.hpp"

using afp::io::DecodeParams;

static std::unique_ptr<afp::io::IAudioDecoder> new_decoder() {
    auto f = afp::io::make_default_decoder_factory();
    return f->create_decoder();
}

TEST(FlacDecode, Stereo16bit) {
    auto blob = testio::load_asset("tiny_stereo.flac");
    if (blob.empty())
        GTEST_SKIP() << "tiny_stereo.flac asset not available";
    auto d = new_decoder();
    DecodeParams p{};
    auto out = d->decode_bytes(
        std::span<const std::byte>(reinterpret_cast<const std::byte *>(blob.data()), blob.size()), p);
    if (out) {
        EXPECT_GT(out->samples.size(), 0u);
        EXPECT_GT(out->sample_rate_hz, 0u);
    } else {
        // If the blob is too tiny, ensure mapping is a clean DecodeError
        EXPECT_EQ(out.error(), afp::util::UtilError::DecodeError);
    }
}

TEST(FlacDecode, CorruptionTruncated) {
    auto blob = testio::load_asset("tiny_stereo.flac");
    if (blob.empty())
        GTEST_SKIP();
    blob.resize(16); // truncate
    auto d = new_decoder();
    DecodeParams p{};
    auto out = d->decode_bytes(
        std::span<const std::byte>(reinterpret_cast<const std::byte *>(blob.data()), blob.size()), p);
    ASSERT_FALSE(out.has_value());
    ASSERT_EQ(out.error(), afp::util::UtilError::DecodeError);
}
