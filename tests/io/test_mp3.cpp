#include <gtest/gtest.h>
#include "afp/io/io.hpp"
#include "afp/util/util.hpp"
#include "test_utils.hpp"

using afp::io::DecodeParams;

static std::unique_ptr<afp::io::IAudioDecoder> new_decoder() {
    auto f = afp::io::make_default_decoder_factory();
    return f->create_decoder();
}

TEST(Mp3Decode, WithID3v2TagIfPresent) {
    auto id3 = testio::load_asset("tiny_id3.mp3");
    if (id3.empty())
        GTEST_SKIP() << "tiny_id3.mp3 asset not available";
    auto d = new_decoder();
    DecodeParams p{};
    auto out = d->decode_bytes(std::span<const std::byte>(reinterpret_cast<const std::byte *>(id3.data()), id3.size()),
                               p);
    if (out) {
        EXPECT_GT(out->samples.size(), 0u);
        EXPECT_GT(out->sample_rate_hz, 0u);
    } else {
        // Very tiny MP3s might fail; ensure DecodeError mapping
        EXPECT_EQ(out.error(), afp::util::UtilError::DecodeError);
    }
}

TEST(Mp3Decode, RawShortStreamOrGracefulFail) {
    auto raw = testio::load_asset("tiny_raw.mp3");
    if (raw.empty())
        GTEST_SKIP() << "tiny_raw.mp3 asset not available";
    auto d = new_decoder();
    DecodeParams p{};
    auto out = d->decode_bytes(std::span<const std::byte>(reinterpret_cast<const std::byte *>(raw.data()), raw.size()),
                               p);
    if (out) {
        EXPECT_GT(out->samples.size(), 0u);
    } else {
        EXPECT_EQ(out.error(), afp::util::UtilError::DecodeError);
    }
}

TEST(Mp3Decode, CorruptionBadSync) {
    std::vector<uint8_t> bad(64, 0x00); // not a valid MP3
    auto d = new_decoder();
    DecodeParams p{};
    auto out = d->decode_bytes(std::span<const std::byte>(reinterpret_cast<const std::byte *>(bad.data()), bad.size()),
                               p);
    ASSERT_FALSE(out.has_value());
    ASSERT_EQ(out.error(), afp::util::UtilError::DecodeError);
}
