#![feature(test)]

extern crate test;

use atomic_cuckoo_filter::CuckooFilter;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, atomic::Ordering};
use std::thread;
use test::Bencher;

/// Benchmarks concurrent read performance (contains() calls) while multiple background
/// threads are also performing reads. This tests the filter's ability to handle
/// high-concurrency read workloads without contention.
///
/// Setup: 131k capacity filter with 100k pre-inserted items
/// Scenario: 10 background threads continuously calling contains() while main thread benchmarks contains()
#[bench]
fn concurrent_contains(b: &mut Bencher) {
    let filter = Arc::new(
        CuckooFilter::builder()
            .capacity(131072)
            .fingerprint_size(8)
            .build()
            .unwrap(),
    );
    let stop_flag = Arc::new(AtomicBool::new(false));
    let mut handles = vec![];

    // Pre-populate with 100k items (even numbers)
    for i in 0..100000 {
        filter.insert(&(i * 2)).unwrap();
    }

    // Start 10 background threads doing continuous contains() calls
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
                f.contains(&i);
            }
        }))
    }

    // Benchmark contains() calls in main thread
    let mut i = 0;
    b.iter(|| {
        if i == 200000 {
            i = 0;
        } else {
            i += 1;
        }
        filter.contains(&i);
    });

    // Clean up background threads
    stop_flag.store(true, Ordering::Relaxed);
    for h in handles {
        h.join().unwrap();
    }
}

/// Benchmarks contains() performance while background threads are actively
/// inserting and removing items. This tests read performance under write contention.
///
/// Setup: 131k capacity filter, initially empty
/// Scenario: 10 background threads inserting new items and removing old ones,
///          while main thread benchmarks contains() performance
#[bench]
fn concurrent_contains_under_write_contention(b: &mut Bencher) {
    let filter = Arc::new(
        CuckooFilter::builder()
            .capacity(131072)
            .fingerprint_size(8)
            .build()
            .unwrap(),
    );
    let stop_flag = Arc::new(AtomicBool::new(false));
    let mut handles = vec![];

    // Start 10 background threads doing insert/remove operations
    for c in 0..10 {
        let f = filter.clone();
        let stop = stop_flag.clone();
        handles.push(thread::spawn(move || {
            let mut i: u16 = c;
            while !stop.load(Ordering::Relaxed) {
                i += 10;
                let _ = f.insert(&i);
                f.remove(&(i - 10000));
            }
        }))
    }

    // Benchmark contains() calls while background threads are modifying the filter
    let mut i: u16 = 0;
    b.iter(|| {
        i += 1;
        filter.contains(&i);
    });

    // Clean up background threads
    stop_flag.store(true, Ordering::Relaxed);
    for h in handles {
        h.join().unwrap();
    }
}
