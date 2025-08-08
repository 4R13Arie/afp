#pragma once
#include <cstdint>
#include <memory>
#include <string_view>
#include <vector>

#include "afp/util/util.hpp"

namespace afp::index {

/**
 * Build/ingestion options for the inverted index.
 *  - compress_postings: on-disk delta + varint encoding
 *  - max_postings_per_key: limit per-key postings during ingest (0 = unlimited)
 *  - song_grouping: group postings by track for better locality on disk
 */
struct BuildOptions {
  bool compress_postings{true};
  std::size_t max_postings_per_key{0}; // 0 = unlimited
  bool song_grouping{true};
};

/**
 * Append-only inverted index writer (single-writer).
 * Thread-safety: NOT thread-safe; one instance per building thread.
 */
class IIndexWriter {
public:
  virtual ~IIndexWriter() = default;

  /**
   * Purpose: Initialize a new or existing index at 'path' for write.
   * Preconditions: path is a valid directory or file target.
   * Postconditions: internal structures are ready for inserts.
   * Complexity: O(1) besides filesystem ops.
   * Thread-safety: NO (writer is single-threaded).
   */
  virtual afp::util::Expected<void>
  open(std::string_view path, const BuildOptions& opts) = 0;

  /**
   * Purpose: Insert a single posting (key → (track, t_anchor_ms)).
   * Preconditions:
   *  - key.value is a valid packed fingerprint key
   *  - t_anchor_ms in milliseconds from track start
   * Postconditions:
   *  - The posting is staged for commit; may be capped by max_postings_per_key.
   * Complexity: Amortized O(1).
   * Thread-safety: NO.
   */
  virtual afp::util::Expected<void>
  insert(afp::util::TrackId track,
         afp::util::FingerprintKey key,
         afp::util::TimeMs t_anchor_ms) = 0;

  /**
   * Purpose: Bulk ingest postings for a single track (faster than repeated insert).
   * Preconditions: keys.size() == t_anchor_ms.size().
   * Postconditions: all accepted postings are staged for commit; per-key caps apply.
   * Complexity: O(N) over input length.
   * Thread-safety: NO.
   */
  virtual afp::util::Expected<void>
  insert_bulk(afp::util::TrackId track,
              std::span<const afp::util::FingerprintKey> keys,
              std::span<const afp::util::TimeMs> t_anchor_ms) = 0;

  /**
   * Purpose: Flush staged postings to disk atomically (write temp then rename).
   * Preconditions: open() has been called.
   * Postconditions: on success, durable index is available to readers at path.
   * Complexity: O(total_postings) to serialize + FS write cost.
   * Thread-safety: NO.
   */
  virtual afp::util::Expected<void> commit() = 0;

  /** Close writer, releasing resources. Idempotent. */
  virtual void close() = 0;
};

/**
 * Inverted index reader with scalable, thread-safe lookups.
 * Thread-safety: YES for open/lookup/lookup_batch/close if treated as const after open.
 */
class IIndexReader {
public:
  virtual ~IIndexReader() = default;

  /**
   * Purpose: Open a persisted index for read-only queries.
   * Preconditions: path points to a valid index file; sufficient memory to load.
   * Postconditions: internal read-only structures are built/ready.
   * Complexity: O(total_index_size) to load; O(1) pointer swaps internally.
   * Thread-safety: YES (immutable after open).
   */
  virtual afp::util::Expected<void>
  open(std::string_view path) = 0;

  /**
   * Purpose: Lookup postings for a single fingerprint key.
   * Preconditions: key.value is valid.
   * Postconditions: returns postings (copy) possibly empty if key not present.
   * Complexity: Average O(1) hash probe + O(k) copy where k = postings for key.
   * Thread-safety: YES.
   */
  virtual afp::util::Expected<std::vector<afp::util::Posting>>
  lookup(afp::util::FingerprintKey key) const = 0;

  /**
   * Purpose: Batch lookup for multiple keys to reduce overhead.
   * Preconditions: keys.size() may be 0..N.
   * Postconditions: result vector aligns with input 'keys'.
   * Complexity: Sum of individual lookups.
   * Thread-safety: YES.
   */
  virtual afp::util::Expected<std::vector<std::vector<afp::util::Posting>>>
  lookup_batch(std::span<const afp::util::FingerprintKey> keys) const = 0;

  /** Close reader; safe to call multiple times. */
  virtual void close() = 0;
};

/** Factory for writer/reader. */
class IIndexFactory {
public:
  virtual ~IIndexFactory() = default;
  virtual std::unique_ptr<IIndexWriter> create_writer() = 0;
  virtual std::unique_ptr<IIndexReader> create_reader() = 0;
};

/** Default index factory (in-memory with on-disk snapshot). */
std::unique_ptr<IIndexFactory> make_default_index_factory();

/**
* LMDB environment/options.
* Units:
*  - map_size_bytes: bytes
*  - max_dbs: count
*
* Thread-safety:
*  - The environment pointer is shared; LMDB supports multiple readers and one writer.
*/
struct LmdbOptions {
  std::size_t map_size_bytes{1ull << 30}; // 1 GiB default; tune per catalog
  unsigned int max_dbs{1};                // we use a single DBI for key→value
  bool use_nosync{false};                 // if true, MDB_NOSYNC for faster builds (crash risk)
  bool use_nometasync{false};             // MDB_NOMETASYNC toggle
  bool use_writemap{false};               // MDB_WRITEMAP (can speed up builds)
};

/**
 * Factory that produces LMDB-backed writer/reader.
 *
 * Thread-safety:
 *  - The factory itself is stateless/thread-safe.
 */
std::unique_ptr<IIndexFactory> make_lmdb_index_factory(const LmdbOptions& env_opts);

} // namespace afp::index
