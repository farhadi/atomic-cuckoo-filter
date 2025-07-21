#![feature(test)]

extern crate test;

use cuckoofilter::CuckooFilter;
use std::hash::DefaultHasher;
use std::sync::RwLock;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, atomic::Ordering};
use std::thread;
use test::Bencher;

/// Benchmarks basic single-threaded insert and remove performance using the
/// reference cuckoofilter implementation. This provides a baseline for
/// comparing against the atomic implementation.
///
/// Setup: 131k capacity filter using DefaultHasher
/// Test: Continuous insert/remove cycle with a sliding window of 100k items
#[bench]
fn insert_and_remove(b: &mut Bencher) {
    let mut filter = CuckooFilter::<DefaultHasher>::with_capacity(131072);
    let mut i = 0;
    b.iter(|| {
        i += 1;
        let _ = filter.add(&i);
        filter.delete(&(i - 100000));
    });
}

/// Benchmarks single-threaded test_and_add performance (insert_unique equivalent).
/// test_and_add ensures an item is only inserted if it doesn't already exist.
///
/// Setup: 131k capacity filter, initially empty
/// Test: Continuous test_and_add operations with incrementing u16 values
#[bench]
fn insert_unique(b: &mut Bencher) {
    let mut filter = CuckooFilter::<DefaultHasher>::with_capacity(131072);
    let mut i: u16 = 0;
    b.iter(|| {
        i += 1;
        let _ = filter.test_and_add(&i);
    });
}

/// Benchmarks insert performance when the filter becomes increasingly full.
/// This tests how performance degrades as the filter reaches capacity and
/// hash collisions become more frequent.
///
/// Setup: 131k capacity filter, initially empty  
/// Test: Continuous insertions until filter is full (no removes)
#[bench]
fn insert_into_full_filter(b: &mut Bencher) {
    let mut filter = CuckooFilter::<DefaultHasher>::with_capacity(131072);
    let mut i = 0;
    b.iter(|| {
        i += 1;
        let _ = filter.add(&i);
    });
}

/// Benchmarks contains() performance when querying for items that exist in the filter.
/// This tests positive lookup performance with a fully-populated filter.
///
/// Setup: 131k capacity filter pre-populated with all u16 values (0-65535)
/// Test: Cycling through contains() calls for values that definitely exist
#[bench]
fn contains_true(b: &mut Bencher) {
    let mut filter = CuckooFilter::<DefaultHasher>::with_capacity(131072);
    // Pre-populate with all possible u16 values
    for i in 0..=65535u16 {
        filter.add(&i).unwrap();
    }
    let mut i: u16 = 0;
    b.iter(|| {
        i += 1;
        filter.contains(&i);
    });
}

/// Benchmarks contains() performance when querying for items that don't exist.
/// This tests negative lookup performance with an empty filter.
///
/// Setup: 131k capacity filter, completely empty
/// Test: Continuous contains() calls for items that definitely don't exist
#[bench]
fn contains_false(b: &mut Bencher) {
    let filter = CuckooFilter::<DefaultHasher>::with_capacity(131072);
    let mut i: u16 = 0;
    b.iter(|| {
        i += 1;
        filter.contains(&i);
    });
}

/// Benchmarks concurrent read performance using RwLock-protected filter.
/// This tests read scalability compared to the lock-free atomic implementation.
///
/// Setup: 131k capacity filter with 100k pre-inserted items, protected by RwLock
/// Scenario: 10 background threads doing continuous reads while main thread benchmarks reads
/// Note: Uses read locks for all contains() operations
#[bench]
fn concurrent_contains(b: &mut Bencher) {
    let filter = Arc::new(RwLock::new(CuckooFilter::<DefaultHasher>::with_capacity(
        131072,
    )));
    let stop_flag = Arc::new(AtomicBool::new(false));
    let mut handles = vec![];

    // Pre-populate with 100k items (even numbers) using write lock
    {
        let f = filter.clone();
        let mut f = f.write().unwrap();
        for i in 0..100000 {
            f.add(&(i * 2)).unwrap();
        }
    }

    // Start 10 background threads doing continuous contains() with read locks
    for _ in 0..10 {
        let f = filter.clone();
        let stop = stop_flag.clone();
        handles.push(thread::spawn(move || {
            let mut i = 0;
            while !stop.load(Ordering::Relaxed) {
                if i == 200000 {
                    i = 0;
                } else {
                    i += 1;
                }
                f.read().unwrap().contains(&i);
            }
        }));
    }

    // Benchmark contains() performance using read locks
    let mut i = 0;
    b.iter(|| {
        if i == 200000 {
            i = 0;
        } else {
            i += 1;
        }
        filter.read().unwrap().contains(&i);
    });

    // Clean up background threads
    stop_flag.store(true, Ordering::Relaxed);
    for h in handles {
        h.join().unwrap();
    }
}

/// Benchmarks read performance while background threads are writing using RwLock.
/// This tests read/write contention compared to the lock-free atomic implementation.
///
/// Setup: 131k capacity filter protected by RwLock
/// Scenario: 10 background threads doing write operations (insert/remove)
///          while main thread benchmarks read performance
#[bench]
fn concurrent_contains_under_write_contention(b: &mut Bencher) {
    let filter = Arc::new(RwLock::new(CuckooFilter::<DefaultHasher>::with_capacity(
        131072,
    )));
    let stop_flag = Arc::new(AtomicBool::new(false));
    let mut handles = vec![];

    // Start 10 background threads doing write operations (insert/remove)
    for c in 0..10 {
        let f = filter.clone();
        let stop = stop_flag.clone();
        handles.push(thread::spawn(move || {
            let mut i: u16 = c;
            while !stop.load(Ordering::Relaxed) {
                i += 10;
                let _ = f.write().unwrap().add(&i);
                f.write().unwrap().delete(&(i - 10000));
            }
        }));
    }

    // Benchmark read performance under write contention
    let mut i: u16 = 0;
    b.iter(|| {
        i += 1;
        filter.read().unwrap().contains(&i);
    });

    // Clean up background threads
    stop_flag.store(true, Ordering::Relaxed);
    for h in handles {
        h.join().unwrap();
    }
}
