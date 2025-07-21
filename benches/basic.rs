#![feature(test)]

extern crate test;

use atomic_cuckoo_filter::CuckooFilter;
use test::Bencher;

/// Benchmarks basic single-threaded insert and remove performance of the atomic
/// cuckoo filter. This provides baseline performance metrics for the lock-free
/// implementation without any concurrent access.
///
/// Setup: 131k capacity filter with 8-bit fingerprints
/// Test: Continuous insert/remove cycle with a sliding window of 100k items
#[bench]
fn insert_and_remove(b: &mut Bencher) {
    let filter = CuckooFilter::builder()
        .capacity(131072)
        .fingerprint_size(8)
        .build()
        .unwrap();
    let mut i = 0;
    b.iter(|| {
        i += 1;
        let _ = filter.insert(&i);
        filter.remove(&(i - 100000)); // Remove item from 100k iterations ago
    });
}

/// Benchmarks single-threaded insert_unique performance. insert_unique provides
/// atomic test-and-insert semantics, ensuring items are only inserted if they
/// don't already exist in the filter.
///
/// Setup: 131k capacity filter with 8-bit fingerprints, initially empty
/// Test: Continuous insert_unique operations with incrementing u16 values
#[bench]
fn insert_unique(b: &mut Bencher) {
    let filter = CuckooFilter::builder()
        .capacity(131072)
        .fingerprint_size(8)
        .build()
        .unwrap();
    let mut i: u16 = 0;
    b.iter(|| {
        i += 1;
        let _ = filter.insert_unique(&i);
    });
}

/// Benchmarks insert/remove performance when the filter is configured with
/// zero evictions allowed. In this case, the filter can do all operations
/// atomically without any locks.
///
/// Setup: 131k capacity filter with max_evictions=0 and 8-bit fingerprints
/// Test: Insert/remove cycle with sliding window, no eviction attempts allowed
#[bench]
fn insert_and_remove_with_max_evictions_0(b: &mut Bencher) {
    let filter = CuckooFilter::builder()
        .capacity(131072)
        .max_evictions(0) // No evictions allowed - faster failure on collisions
        .fingerprint_size(8)
        .build()
        .unwrap();
    let mut i = 0;
    b.iter(|| {
        i += 1;
        let _ = filter.insert(&i);
        filter.remove(&(i - 100000)) // Remove item from 100k iterations ago
    });
}

/// Benchmarks insert performance as the filter becomes increasingly full.
/// This tests how performance degrades as the load factor increases and
/// hash collisions become more frequent, requiring more eviction attempts.
///
/// Setup: 131k capacity filter with 8-bit fingerprints, initially empty
/// Test: Continuous insertions without any removes until filter reaches capacity
#[bench]
fn insert_into_full_filter(b: &mut Bencher) {
    let filter = CuckooFilter::builder()
        .capacity(131072)
        .fingerprint_size(8)
        .build()
        .unwrap();
    let mut i = 0;
    b.iter(|| {
        i += 1;
        let _ = filter.insert(&i);
    });
}

/// Benchmarks contains() performance when querying for items that exist in the filter.
/// This tests positive lookup performance with a fully-populated filter, measuring
/// the cost of successful hash table lookups.
///
/// Setup: 131k capacity filter pre-populated with all u16 values (0-65535)
/// Test: Cycling through contains() calls for values that definitely exist
#[bench]
fn contains_true(b: &mut Bencher) {
    let filter = CuckooFilter::builder()
        .capacity(131072)
        .fingerprint_size(8)
        .build()
        .unwrap();
    // Pre-populate with all possible u16 values
    for i in 0..=65535u16 {
        filter.insert(&i).unwrap();
    }
    let mut i: u16 = 0;
    b.iter(|| {
        i += 1;
        filter.contains(&i);
    });
}

/// Benchmarks contains() performance when querying for items that don't exist.
/// This tests negative lookup performance with an empty filter, measuring
/// the cost of failed hash table lookups.
///
/// Setup: 131k capacity filter with 8-bit fingerprints, completely empty
/// Test: Continuous contains() calls for items that definitely don't exist
#[bench]
fn contains_false(b: &mut Bencher) {
    let filter = CuckooFilter::builder()
        .capacity(131072)
        .fingerprint_size(8)
        .build()
        .unwrap();
    let mut i: u16 = 0;
    b.iter(|| {
        i += 1;
        filter.contains(&i);
    });
}

/// Benchmarks contains() performance with a filter configured for zero evictions.
/// In this case, the filter can do all operations atomically without any optimistic concurrency control.
///
/// Setup: 131k capacity filter with max_evictions=0, attempt to insert all u16 values
/// Test: Contains() calls for values that may or may not exist (depending on insertion success)
#[bench]
fn contains_with_max_evictions_0(b: &mut Bencher) {
    let filter = CuckooFilter::builder()
        .capacity(131072)
        .max_evictions(0) // No evictions - some insertions may fail due to collisions
        .build()
        .unwrap();
    // Attempt to insert all u16 values (some may fail due to collisions)
    for i in 0..=65535u16 {
        let _ = filter.insert(&i);
    }
    let mut i: u16 = 0;
    b.iter(|| {
        i += 1;
        filter.contains(&i);
    });
}
