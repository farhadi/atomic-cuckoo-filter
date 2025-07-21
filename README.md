# Atomic Cuckoo Filter

A high-performance, lock-free concurrent cuckoo filter implementation in Rust for efficient set membership testing.

[![Crates.io](https://img.shields.io/crates/v/atomic-cuckoo-filter.svg)](https://crates.io/crates/atomic-cuckoo-filter)
[![Documentation](https://docs.rs/atomic-cuckoo-filter/badge.svg)](https://docs.rs/atomic-cuckoo-filter)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Overview

This crate provides a sophisticated implementation of a cuckoo filter - a probabilistic data structure
for fast set membership testing. Unlike traditional implementations, this version uses **lock-free**
atomic operations and is designed for high-concurrency environments.

## Key Features

✨ **Lock-Free Concurrency**: All operations use atomic compare-exchange loops instead of traditional locks  
🚀 **High Performance**: Optimized for multi-threaded environments with minimal blocking  
🔍 **No False Negatives**: Items that were inserted are guaranteed to be found  
🎯 **Controllable False Positives**: Configurable fingerprint size to tune accuracy  
📦 **Space Efficient**: ~20-30% less memory usage than Bloom filters for the same false positive rate  
🗑️ **Deletion Support**: Unlike Bloom filters, items can be safely removed  
⏱️ **Bounded Lookup Time**: Always at most 2 bucket checks maximum  
🔧 **Highly Configurable**: Customizable capacity, fingerprint size, bucket size, and eviction limits  

## Quick Start

Add this to your `Cargo.toml`:

```toml
[dependencies]
atomic-cuckoo-filter = "0.1"
```

### Basic Usage

```rust
use atomic_cuckoo_filter::CuckooFilter;

// Create a filter with default settings
let filter = CuckooFilter::new();

// Insert items
filter.insert(&"hello").unwrap();
filter.insert(&"world").unwrap();
filter.insert(&42).unwrap();

// Check membership
assert!(filter.contains(&"hello"));
assert!(filter.contains(&42));
assert!(!filter.contains(&"rust"));

// Remove items
assert!(filter.remove(&"hello"));
assert!(!filter.contains(&"hello"));

// Count occurrences (not meant to be used as a counting filter, but to detect duplicates or hash collisions)
filter.insert(&"duplicate").unwrap();
filter.insert(&"duplicate").unwrap();
assert_eq!(filter.count(&"duplicate"), 2);

println!("Filter contains {} items", filter.len());

// Unique Insertions (Atomically check and insert items)
// Returns Ok(true) if inserted, Ok(false) if already present
match filter.insert_unique(&"item") {
    Ok(true) => println!("Item was inserted"),
    Ok(false) => println!("Item already existed"),
    Err(e) => println!("Filter is full: {}", e),
}
```

### Custom Configuration

```rust
use atomic_cuckoo_filter::CuckooFilter;

let filter = CuckooFilter::builder()
    .capacity(1_000_000)        // Target capacity
    .fingerprint_size(16)       // Bits per fingerprint (4, 8, 16, or 32)
    .bucket_size(4)             // Fingerprints per bucket
    .max_evictions(500)         // Maximum eviction chain length
    .build()
    .unwrap();
```

### Custom Hash Functions

```rust
use ahash::AHasher;

let filter = CuckooFilterBuilder::<AHasher>::default()
    .capacity(1024)
    .build()
    .unwrap();
```

### Concurrent Usage

The filter is designed for high-concurrency scenarios:

```rust
use atomic_cuckoo_filter::CuckooFilter;
use std::sync::Arc;
use std::thread;

let filter = Arc::new(CuckooFilter::with_capacity(100_000));

// Spawn multiple threads for concurrent operations
let mut handles = vec![];

// Writer threads
for i in 0..4 {
    let filter_clone = Arc::clone(&filter);
    handles.push(thread::spawn(move || {
        for j in 0..1000 {
            let item = format!("item_{}_{}", i, j);
            filter_clone.insert(&item).unwrap();
        }
    }));
}

// Reader threads
for i in 0..4 {
    let filter_clone = Arc::clone(&filter);
    handles.push(thread::spawn(move || {
        for j in 0..1000 {
            let item = format!("item_{}_{}", i, j);
            while !filter_clone.contains(&item) {};
        }
    }));
}

// Wait for all threads to complete
for handle in handles {
    handle.join().unwrap();
}

println!("Final filter size: {}", filter.len());
```

## Configuration Options

| Parameter | Description | Valid Values | Default |
|-----------|-------------|--------------|---------|
| `capacity` | Target number of items | Any positive integer | 1,048,576 |
| `fingerprint_size` | Bits per fingerprint | 4, 8, 16, or 32 | 16 |
| `bucket_size` | Fingerprints per bucket | Any positive integer | 4 |
| `max_evictions` | Max eviction chain length | Any integer ≥ 0 | 500 |

### Choosing Parameters

**Fingerprint Size**: Larger fingerprints = fewer false positives but more memory usage

**Bucket Size**: Larger buckets = Faster inserts (fewer evictions), but slower lookups, and slightly higher FPR

**Max Evictions**: 
- 0 = No evictions (faster but may fail to insert occasionally)
- Higher values = Better space utilization but slower inserts when load factor is high

## Concurrency Model

All operations use atomic compare-exchange loops instead of traditional locks, with optimistic
concurrency control for read operations. The only exception is when inserting with evictions,
where an atomic-based lock is used to ensure consistency.

## Error Handling

The main error type is `Error::NotEnoughSpace`, returned when the filter cannot accommodate more items:

```rust
use atomic_cuckoo_filter::{CuckooFilter, Error};

let small_filter = CuckooFilter::builder()
    .capacity(10)
    .max_evictions(0)  // Disable evictions
    .build()
    .unwrap();

// Fill the filter
for i in 0..20 {
    match small_filter.insert(&i) {
        Ok(()) => println!("Inserted {}", i),
        Err(Error::NotEnoughSpace) => {
            println!("Filter is full at {} items", small_filter.len());
            break;
        }
    }
}
```

## Testing

Run the test suite:

```bash
# Unit tests
cargo test

# Benchmarks
cargo bench
```

## Safety and Guarantees

- **Thread Safety**: All operations are thread-safe and can be called concurrently
- **Memory Safety**: No unsafe code in the public API (uses `parking_lot_core` internally)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
