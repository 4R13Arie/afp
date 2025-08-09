#include "afp/index/index.hpp"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include <lmdb.h>
#include "afp/util/util.hpp"

namespace afp::index {
using afp::util::Expected;
using afp::util::UtilError;
using afp::util::FingerprintKey;
using afp::util::Posting;
using afp::util::TrackId;
using afp::util::TimeMs;

// -----------------------------
// Helpers: error mapping, endian, varint
// -----------------------------

namespace {
inline UtilError map_mdb(int rc) noexcept {
  if (rc == 0) return UtilError::None;
  switch (rc) {
    case MDB_NOTFOUND:
      return UtilError::NotFound;
    case MDB_MAP_FULL:
      return UtilError::ResourceExhausted;
    case MDB_PANIC:
    case MDB_CORRUPTED:
      return UtilError::IndexCorrupt;
    case MDB_BAD_VALSIZE:
    case MDB_BAD_DBI:
    case EINVAL:
      return UtilError::InvalidArgument;
    case ENOSPC:
    case ENOMEM:
      return UtilError::OutOfMemory;
    default:
      return UtilError::IOError;
  }
}

// Little-endian put/get
inline void put_u32(std::vector<std::uint8_t>& o, std::uint32_t v) {
  o.push_back((v) & 0xFFu);
  o.push_back((v >> 8) & 0xFFu);
  o.push_back((v >> 16) & 0xFFu);
  o.push_back((v >> 24) & 0xFFu);
}

inline void put_u64(std::vector<std::uint8_t>& o, std::uint64_t v) {
  for (int i = 0; i < 8; ++i) o.push_back((v >> (8 * i)) & 0xFFu);
}

inline std::uint32_t get_u32(const std::uint8_t* p) {
  return (std::uint32_t)p[0] | ((std::uint32_t)p[1] << 8) | (
           (std::uint32_t)p[2] << 16) | (
           (std::uint32_t)p[3] << 24);
}

inline std::uint64_t get_u64(const std::uint8_t* p) {
  std::uint64_t v = 0;
  for (int i = 0; i < 8; ++i) v |= (std::uint64_t)p[i] << (8 * i);
  return v;
}

// Unsigned base-128 varint
inline void put_varu(std::vector<std::uint8_t>& out, std::uint64_t v) {
  while (v >= 0x80) {
    out.push_back((std::uint8_t)((v & 0x7F) | 0x80));
    v >>= 7;
  }
  out.push_back((std::uint8_t)(v & 0x7F));
}

inline const std::uint8_t* get_varu(const std::uint8_t* p,
                                    const std::uint8_t* end, std::uint64_t& v) {
  v = 0;
  int s = 0;
  while (p < end) {
    std::uint8_t b = *p++;
    v |= (std::uint64_t)(b & 0x7F) << s;
    if (!(b & 0x80)) return p;
    s += 7;
    if (s > 63) return nullptr;
  }
  return nullptr;
}

// Value layout flags
constexpr std::uint8_t kFlagCompress = 0x01; // delta+varint used
constexpr std::uint8_t kFlagGrouped = 0x02; // per-track grouping included

// Value format (LE):
// [flags u8][group_count u32]
//   repeat group_count times:
//     [track_id u64][count u32][deltas varint... or absolute u32...]
struct Groups {
  // track -> sorted times
  std::map<TrackId, std::vector<TimeMs>> groups;
  // ordered to keep stable layout
};

inline void encode_groups(const Groups& g, bool compress,
                          std::vector<std::uint8_t>& out) {
  out.clear();
  out.reserve(128);
  std::uint8_t flags = 0;
  if (compress) flags |= kFlagCompress;
  flags |= kFlagGrouped;
  out.push_back(flags);
  put_u32(out, (std::uint32_t)g.groups.size());
  for (const auto& kv : g.groups) {
    put_u64(out, kv.first);
    const auto& times = kv.second;
    put_u32(out, (std::uint32_t)times.size());
    if (compress) {
      std::uint32_t prev = 0;
      for (size_t i = 0; i < times.size(); ++i) {
        const std::uint32_t d = (i == 0) ? times[i] : (times[i] - prev);
        put_varu(out, d);
        prev = times[i];
      }
    } else {
      for (auto t : times) put_u32(out, t);
    }
  }
}

inline Expected<Groups> decode_groups(const std::uint8_t* data, size_t len) {
  if (len < 1 + 4) return tl::unexpected(UtilError::IndexCorrupt);
  Groups g;
  const std::uint8_t* p = data;
  const std::uint8_t* end = data + len;
  const std::uint8_t flags = *p++;
  const bool compress = (flags & kFlagCompress) != 0;
  const bool grouped = (flags & kFlagGrouped) != 0;
  if (!grouped) return tl::unexpected(UtilError::IndexCorrupt);
  if (p + 4 > end) return tl::unexpected(UtilError::IndexCorrupt);
  const std::uint32_t gcnt = get_u32(p);
  p += 4;
  for (std::uint32_t gi = 0; gi < gcnt; ++gi) {
    if (p + 8 + 4 > end) return tl::unexpected(UtilError::IndexCorrupt);
    const TrackId track = get_u64(p);
    p += 8;
    const std::uint32_t count = get_u32(p);
    p += 4;
    std::vector<TimeMs> times;
    times.reserve(count);
    if (compress) {
      std::uint32_t prev = 0;
      for (std::uint32_t i = 0; i < count; ++i) {
        std::uint64_t d = 0;
        const std::uint8_t* np = get_varu(p, end, d);
        if (!np) return tl::unexpected(UtilError::IndexCorrupt);
        p = np;
        prev += (std::uint32_t)d;
        times.push_back(prev);
      }
    } else {
      if (p + 4ull * count > end)
        return
            tl::unexpected(UtilError::IndexCorrupt);
      for (std::uint32_t i = 0; i < count; ++i) {
        const std::uint32_t t = get_u32(p);
        p += 4;
        times.push_back(t);
      }
    }
    // guarantee sorted & unique
    std::sort(times.begin(), times.end());
    times.erase(std::unique(times.begin(), times.end()), times.end());
    g.groups.emplace(track, std::move(times));
  }
  return g;
}

// Merge-in: append times to group[track], enforce cap, keep sorted unique.
inline void merge_times(std::vector<TimeMs>& base, std::span<const TimeMs> add,
                        std::size_t cap_per_key) {
  base.insert(base.end(), add.begin(), add.end());
  std::sort(base.begin(), base.end());
  base.erase(std::unique(base.begin(), base.end()), base.end());
  if (cap_per_key > 0 && base.size() > cap_per_key) {
    // Keep earliest times; drop tail
    base.resize(cap_per_key);
  }
}
} // namespace

// -----------------------------
// LMDB Writer (single-writer)
// -----------------------------

class LmdbIndexWriter final : public IIndexWriter {
public:
  LmdbIndexWriter(MDB_env* env, MDB_dbi dbi, BuildOptions opts)
    : env_(env), dbi_(dbi), opts_(opts) {
  }

  // Purpose: Prepare staging buffers; LMDB env/dbi provided by factory.
  // Thread-safety: NOT thread-safe (single-writer).
  Expected<void>
  open(std::string_view path, const BuildOptions& opts) override {
    (void)path; // DB already opened by factory; we keep options from ctor
    opts_ = opts;
    staging_.clear();
    opened_ = true;
    return {};
  }

  // Purpose: Stage a single posting. Amortized O(1).
  Expected<void>
  insert(TrackId track, FingerprintKey key, TimeMs t_anchor_ms) override {
    if (!opened_) return tl::unexpected(UtilError::Unavailable);
    auto& per_track = staging_[key.value][track];
    per_track.push_back(t_anchor_ms);
    return {};
  }

  // Purpose: Stage many postings in one call. O(N).
  Expected<void> insert_bulk(TrackId track,
                             std::span<const FingerprintKey> keys,
                             std::span<const TimeMs> t_anchor_ms) override {
    if (!opened_) return tl::unexpected(UtilError::Unavailable);
    if (keys.size() != t_anchor_ms.size())
      return tl::unexpected(
          UtilError::SizeMismatch);
    auto itK = keys.begin();
    auto itT = t_anchor_ms.begin();
    for (; itK != keys.end(); ++itK, ++itT) {
      staging_[itK->value][track].push_back(*itT);
    }
    return {};
  }

  // Purpose: Atomically merge staged postings into LMDB with one write txn.
  // Complexity: O(total staged) + LMDB write cost.
  Expected<void> commit() override {
    if (!opened_) return tl::unexpected(UtilError::Unavailable);
    if (staging_.empty()) return {};

    MDB_txn* txn = nullptr;
    int rc = mdb_txn_begin(env_, nullptr, 0, &txn);
    if (rc != 0) return tl::unexpected(map_mdb(rc));

    std::vector<std::uint8_t> valbuf;
    for (auto& kEntry : staging_) {
      const std::uint32_t key_u32 = kEntry.first;

      // Read current value (if any)
      MDB_val k, v;
      k.mv_size = sizeof(key_u32);
      k.mv_data = const_cast<std::uint32_t*>(&key_u32);
      Groups g;

      rc = mdb_get(txn, dbi_, &k, &v);
      if (rc == 0) {
        auto dec = decode_groups(static_cast<const std::uint8_t*>(v.mv_data),
                                 v.mv_size);
        if (!dec) {
          mdb_txn_abort(txn);
          return tl::unexpected(dec.error());
        }
        g = std::move(*dec);
      } else if (rc != MDB_NOTFOUND) {
        mdb_txn_abort(txn);
        return tl::unexpected(map_mdb(rc));
      }

      // Merge staged per-track times into g
      for (auto& trackEntry : kEntry.second) {
        auto& times_add = trackEntry.second;
        std::sort(times_add.begin(), times_add.end());
        times_add.erase(std::unique(times_add.begin(), times_add.end()),
                        times_add.end());
        auto& base = g.groups[trackEntry.first];
        merge_times(
            base, std::span<const TimeMs>(times_add.data(), times_add.size()),
            opts_.max_postings_per_key);
      }

      // Encode and put
      encode_groups(g, opts_.compress_postings, valbuf);

      MDB_val nv;
      nv.mv_size = valbuf.size();
      nv.mv_data = valbuf.data();
      rc = mdb_put(txn, dbi_, &k, &nv, 0);
      if (rc != 0) {
        mdb_txn_abort(txn);
        return tl::unexpected(map_mdb(rc));
      }
    }

    rc = mdb_txn_commit(txn);
    if (rc != 0) return tl::unexpected(map_mdb(rc));

    staging_.clear();
    return {};
  }

  // Purpose: Clear local staging; env/dbi remain valid.
  void close() override {
    staging_.clear();
    opened_ = false;
  }

private:
  // key → (track → times)
  std::unordered_map<std::uint32_t, std::unordered_map<
                       TrackId, std::vector<TimeMs>>> staging_;
  MDB_env* env_{nullptr};
  MDB_dbi dbi_{0};
  BuildOptions opts_{};
  bool opened_{false};
};

// -----------------------------
// LMDB Reader (multi-reader, thread-safe)
// -----------------------------

class LmdbIndexReader final : public IIndexReader {
public:
  LmdbIndexReader(MDB_env* env, MDB_dbi dbi) : env_(env), dbi_(dbi) {
  }

  // Purpose: No-op (env/dbi already open); validate handle.
  Expected<void> open(std::string_view /*path*/) override {
    opened_.store(true, std::memory_order_release);
    return {};
  }

  // Purpose: Lookup postings for a key. Average O(1)+decode.
  Expected<std::vector<Posting>> lookup(FingerprintKey key) const override {
    if (!opened_.load(std::memory_order_acquire))
      return tl::unexpected(
          UtilError::Unavailable);

    MDB_txn* txn = nullptr;
    int rc = mdb_txn_begin(env_, nullptr, MDB_RDONLY, &txn);
    if (rc != 0) return tl::unexpected(map_mdb(rc));

    MDB_val k, v;
    std::uint32_t k32 = key.value;
    k.mv_size = sizeof(k32);
    k.mv_data = &k32;

    rc = mdb_get(txn, dbi_, &k, &v);
    if (rc == MDB_NOTFOUND) {
      mdb_txn_abort(txn);
      return std::vector<Posting>{};
    }
    if (rc != 0) {
      mdb_txn_abort(txn);
      return tl::unexpected(map_mdb(rc));
    }

    auto dec = decode_groups(static_cast<const std::uint8_t*>(v.mv_data),
                             v.mv_size);
    mdb_txn_abort(txn);
    if (!dec) return tl::unexpected(dec.error());

    std::vector<Posting> out;
    // Flatten grouped postings
    for (const auto& kv : dec->groups) {
      const TrackId track = kv.first;
      for (auto t : kv.second) out.push_back(Posting{track, t});
    }
    return out;
  }

  // Purpose: Batch lookups; sum of individual costs.
  Expected<std::vector<std::vector<Posting>>>
  lookup_batch(std::span<const FingerprintKey> keys) const override {
    if (!opened_.load(std::memory_order_acquire))
      return tl::unexpected(
          UtilError::Unavailable);

    std::vector<std::vector<Posting>> out;
    out.reserve(keys.size());

    MDB_txn* txn = nullptr;
    int rc = mdb_txn_begin(env_, nullptr, MDB_RDONLY, &txn);
    if (rc != 0) return tl::unexpected(map_mdb(rc));

    for (auto kIn : keys) {
      MDB_val k, v;
      std::uint32_t k32 = kIn.value;
      k.mv_size = sizeof(k32);
      k.mv_data = &k32;

      rc = mdb_get(txn, dbi_, &k, &v);
      if (rc == MDB_NOTFOUND) {
        out.emplace_back();
        continue;
      }
      if (rc != 0) {
        mdb_txn_abort(txn);
        return tl::unexpected(map_mdb(rc));
      }

      auto dec = decode_groups(static_cast<const std::uint8_t*>(v.mv_data),
                               v.mv_size);
      if (!dec) {
        mdb_txn_abort(txn);
        return tl::unexpected(dec.error());
      }

      std::vector<Posting> postings;
      for (const auto& kv : dec->groups) {
        const TrackId track = kv.first;
        for (auto t : kv.second) postings.push_back(Posting{track, t});
      }
      out.emplace_back(std::move(postings));
    }
    mdb_txn_abort(txn);
    return out;
  }

  void close() override { opened_.store(false, std::memory_order_release); }

private:
  MDB_env* env_{nullptr};
  MDB_dbi dbi_{0};
  std::atomic<bool> opened_{false};
};

// -----------------------------
// Factory: open LMDB env/DBI, hand out writer/reader
// -----------------------------

class LmdbIndexFactory final : public IIndexFactory {
public:
  explicit LmdbIndexFactory(const std::string& path, const LmdbOptions& opts)
    : path_(path), opts_(opts) {
    (void)open_env();
  }

  ~LmdbIndexFactory() override {
    if (env_) {
      // dbi_ closed when env closes; LMDB cleans up readers automatically
      mdb_env_close(env_);
      env_ = nullptr;
    }
  }

  std::unique_ptr<IIndexWriter> create_writer() override {
    return std::make_unique<LmdbIndexWriter>(env_, dbi_, build_opts_);
  }

  std::unique_ptr<IIndexReader> create_reader() override {
    return std::make_unique<LmdbIndexReader>(env_, dbi_);
  }

  // Optional: allow caller to set BuildOptions defaults for writers
  void set_build_options(const BuildOptions& bo) { build_opts_ = bo; }

private:
  UtilError open_env() {
    // Create dir
    std::error_code ec;
    std::filesystem::create_directories(path_, ec);

    int rc = mdb_env_create(&env_);
    if (rc != 0) return map_mdb(rc);

    // Flags
    unsigned int flags = 0;
    if (opts_.use_nosync) flags |= MDB_NOSYNC;
    if (opts_.use_nometasync) flags |= MDB_NOMETASYNC;
    if (opts_.use_writemap) flags |= MDB_WRITEMAP;

    rc = mdb_env_set_mapsize(env_, (size_t)opts_.map_size_bytes);
    if (rc != 0) return map_mdb(rc);
    rc = mdb_env_set_maxdbs(env_, opts_.max_dbs);
    if (rc != 0) return map_mdb(rc);

    rc = mdb_env_open(env_, path_.c_str(), flags, 0644);
    if (rc != 0) return map_mdb(rc);

    MDB_txn* txn;
    rc = mdb_txn_begin(env_, nullptr, 0, &txn);
    if (rc != 0) return map_mdb(rc);

    // Open (or create) the main DBI for key→value
    rc = mdb_dbi_open(txn, nullptr /* unnamed */, 0, &dbi_);
    if (rc != 0) {
      mdb_txn_abort(txn);
      return map_mdb(rc);
    }

    rc = mdb_txn_commit(txn);
    if (rc != 0) return map_mdb(rc);
    return UtilError::None;
  }

  std::string path_;
  LmdbOptions opts_{};
  BuildOptions build_opts_{}; // defaults propagated to writers
  MDB_env* env_{nullptr};
  MDB_dbi dbi_{0};
};

std::unique_ptr<IIndexFactory> make_lmdb_index_factory(
    const LmdbOptions& env_opts) {
  // Caller should provide a directory path via environment variable or a setter.
  // For simplicity, pick "./afp_index.lmdb".
  static const std::string default_path = "afp_index.lmdb";
  return std::make_unique<LmdbIndexFactory>(default_path, env_opts);
}
} // namespace afp::index
