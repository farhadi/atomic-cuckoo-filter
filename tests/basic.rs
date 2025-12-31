use ahash::AHasher;
use atomic_cuckoo_filter::{CuckooFilter, CuckooFilterBuilder, DeserializeError};
use std::collections::hash_map::DefaultHasher;
// Helper function to create test data
fn test_items(count: usize) -> Vec<String> {
    (0..count).map(|i| format!("test_item_{i}")).collect()
}

#[test]
fn test_new_filter() {
    let filter = CuckooFilter::new();
    assert_eq!(filter.len(), 0);
    assert!(filter.is_empty());
    assert_eq!(filter.capacity(), 1048576); // Default capacity
}

#[test]
fn test_with_capacity() {
    let filter = CuckooFilter::with_capacity(1000);
    assert_eq!(filter.len(), 0);
    assert!(filter.is_empty());
    assert_eq!(filter.capacity(), 1024); // Rounded up to power of 2
}

#[test]
fn test_builder_default() {
    let filter = CuckooFilter::builder().build().unwrap();
    assert_eq!(filter.len(), 0);
    assert!(filter.is_empty());
}

#[test]
fn test_builder_custom_config() {
    let filter = CuckooFilter::builder()
        .capacity(2048)
        .fingerprint_size(8)
        .bucket_size(2)
        .max_evictions(100)
        .build()
        .unwrap();

    assert_eq!(filter.len(), 0);
    assert_eq!(filter.capacity(), 2048);
}

#[test]
fn test_builder_validation_invalid_fingerprint_size() {
    let result = CuckooFilter::builder()
        .fingerprint_size(7) // Invalid: must be 4, 8, 16, or 32
        .build();

    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("Invalid fingerprint_size")
    );
}

#[test]
fn test_builder_validation_zero_bucket_size() {
    let result = CuckooFilter::builder().bucket_size(0).build();

    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("bucket_size must be greater than zero")
    );
}

#[test]
fn test_builder_validation_zero_capacity() {
    let result = CuckooFilter::builder().capacity(0).build();

    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("capacity must be greater than zero")
    );
}

#[test]
fn test_empty_filter_operations() {
    let filter = CuckooFilter::with_capacity(1024);

    // Test operations on empty filter
    assert!(!filter.contains(&"nonexistent"));
    assert_eq!(filter.count(&"nonexistent"), 0);
    assert!(!filter.remove(&"nonexistent"));
    assert_eq!(filter.len(), 0);
    assert!(filter.is_empty());
}

#[test]
fn test_basic_insert_contains() {
    let filter = CuckooFilter::with_capacity(1024);
    let item = "test_item";

    assert!(!filter.contains(&item));
    assert!(filter.insert(&item).is_ok());
    assert!(filter.contains(&item));
    assert_eq!(filter.len(), 1);
    assert!(!filter.is_empty());
}

#[test]
fn test_insert_duplicate_items() {
    let filter = CuckooFilter::with_capacity(1024);
    let item = "duplicate_item";

    // Insert same item multiple times
    assert!(filter.insert(&item).is_ok());
    assert!(filter.insert(&item).is_ok());
    assert!(filter.insert(&item).is_ok());

    assert!(filter.contains(&item));
    assert_eq!(filter.count(&item), 3);
    assert_eq!(filter.len(), 3);
}

#[test]
fn test_insert_unique() {
    let filter = CuckooFilter::with_capacity(1024);
    let item = "unique_item";

    // First insertion should succeed
    assert_eq!(filter.insert_unique(&item), Ok(true));
    assert_eq!(filter.count(&item), 1);

    // Second insertion should return false (already exists)
    assert_eq!(filter.insert_unique(&item), Ok(false));
    assert_eq!(filter.count(&item), 1);
    assert_eq!(filter.len(), 1);
}

#[test]
fn test_remove_existing_item() {
    let filter = CuckooFilter::with_capacity(1024);
    let item = "removable_item";

    // Insert and then remove
    assert!(filter.insert(&item).is_ok());
    assert!(filter.contains(&item));
    assert!(filter.remove(&item));
    assert!(!filter.contains(&item));
    assert_eq!(filter.len(), 0);

    // Trying to remove again should return false
    assert!(!filter.remove(&item));
}

#[test]
fn test_remove_duplicate_items() {
    let filter = CuckooFilter::with_capacity(1024);
    let item = "dup_removable";

    // Insert multiple copies
    assert!(filter.insert(&item).is_ok());
    assert!(filter.insert(&item).is_ok());
    assert!(filter.insert(&item).is_ok());
    assert_eq!(filter.count(&item), 3);

    // Remove one at a time
    assert!(filter.remove(&item));
    assert_eq!(filter.count(&item), 2);
    assert!(filter.remove(&item));
    assert_eq!(filter.count(&item), 1);
    assert!(filter.remove(&item));
    assert_eq!(filter.count(&item), 0);
    assert!(!filter.contains(&item));
}

#[test]
fn test_clear() {
    let filter = CuckooFilter::with_capacity(1024);
    let items = test_items(100);

    // Insert many items
    for item in &items {
        assert!(filter.insert(item).is_ok());
    }
    assert_eq!(filter.len(), 100);

    // Clear all items
    filter.clear();
    assert_eq!(filter.len(), 0);
    assert!(filter.is_empty());

    // Verify all items are gone
    for item in &items {
        assert!(!filter.contains(item));
    }
}

#[test]
fn test_serialize_and_deserialize() {
    let filter = CuckooFilter::builder()
        .capacity(512)
        .fingerprint_size(8)
        .bucket_size(4)
        .max_evictions(50)
        .build()
        .unwrap();

    for item in &test_items(50) {
        assert!(filter.insert(item).is_ok());
    }

    let serialized = filter.to_bytes();
    let restored =
        CuckooFilter::<DefaultHasher>::from_bytes(&serialized).expect("should deserialize");

    assert_eq!(restored.capacity(), filter.capacity());
    assert_eq!(restored.len(), filter.len());

    for item in &test_items(50) {
        assert!(restored.contains(item));
    }
}

#[test]
fn test_serde_json_roundtrip() {
    let filter = CuckooFilter::builder()
        .capacity(128)
        .fingerprint_size(8)
        .bucket_size(4)
        .max_evictions(10)
        .build()
        .unwrap();

    for item in &test_items(20) {
        assert!(filter.insert(item).is_ok());
    }

    let json = serde_json::to_string(&filter).expect("serialize to json");
    let restored: CuckooFilter<DefaultHasher> =
        serde_json::from_str(&json).expect("deserialize from json");

    assert_eq!(restored.capacity(), filter.capacity());
    assert_eq!(restored.len(), filter.len());

    for item in &test_items(20) {
        assert!(restored.contains(item));
    }
}

#[test]
fn test_deserialize_invalid_header() {
    let bytes = vec![0u8; 10];
    let error = CuckooFilter::<DefaultHasher>::from_bytes(&bytes).unwrap_err();
    assert_eq!(error, DeserializeError::InvalidLength);

    let mut serialized = CuckooFilter::new().to_bytes();
    serialized[0] = b'X';
    let error = CuckooFilter::<DefaultHasher>::from_bytes(&serialized).unwrap_err();
    assert_eq!(error, DeserializeError::InvalidHeader);
}

#[test]
fn test_count_functionality() {
    let filter = CuckooFilter::with_capacity(1024);
    let item = "countable_item";

    assert_eq!(filter.count(&item), 0);

    // Add items and verify count increases
    for i in 1..=5 {
        assert!(filter.insert(&item).is_ok());
        assert_eq!(filter.count(&item), i);
    }

    // Remove items and verify count decreases
    for i in (1..=5).rev() {
        assert!(filter.remove(&item));
        assert_eq!(filter.count(&item), i - 1);
    }
}

#[test]
fn test_different_item_types() {
    let filter = CuckooFilter::with_capacity(1024);

    // Test with different types that implement Hash
    assert!(filter.insert(&42i32).is_ok());
    assert!(filter.insert(&"string").is_ok());
    assert!(filter.insert(&vec![1, 2, 3]).is_ok());
    assert!(filter.insert(&(1, 2, 3)).is_ok());

    assert!(filter.contains(&42i32));
    assert!(filter.contains(&"string"));
    assert!(filter.contains(&vec![1, 2, 3]));
    assert!(filter.contains(&(1, 2, 3)));

    assert_eq!(filter.len(), 4);
}

#[test]
fn test_false_positives() {
    let filter = CuckooFilter::builder()
        .capacity(1024)
        .fingerprint_size(8) // Smaller fingerprint = higher false positive rate
        .build()
        .unwrap();

    // Insert known items
    let known_items: Vec<i32> = (0..500).collect();
    for item in &known_items {
        assert!(filter.insert(item).is_ok());
    }

    // Test with unknown items
    let unknown_items: Vec<i32> = (1000..2000).collect();
    let false_positives = unknown_items
        .iter()
        .filter(|item| filter.contains(item))
        .count();

    // Should have some false positives but not too many
    assert!(false_positives > 0);
    assert!(false_positives < 50); // Less than 10% false positive rate
}

#[test]
fn test_no_false_negatives() {
    let filter = CuckooFilter::with_capacity(1024);
    let items = test_items(1024);

    // Insert items and filter out the ones that failed to insert
    let inserted_items = items
        .into_iter()
        .filter(|item| filter.insert(item).is_ok())
        .collect::<Vec<_>>();

    // All inserted items should be found (no false negatives)
    for item in inserted_items {
        assert!(filter.contains(&item), "False negative for item: {item}");
    }
}

#[test]
fn test_full_filter_insertion() {
    let filter = CuckooFilter::builder()
        .capacity(16) // Very small capacity
        .max_evictions(0) // No evictions
        .build()
        .unwrap();

    let mut successful_inserts = 0;

    // Try to insert many items
    for i in 0..100 {
        if filter.insert(&i).is_ok() {
            successful_inserts += 1;
        } else {
            break; // Filter is full
        }
    }

    // Should fill up and then start failing
    assert!(successful_inserts <= filter.capacity());
    assert!(successful_inserts > 0);
    assert_eq!(filter.len(), successful_inserts);
}

#[test]
fn test_eviction_behavior() {
    let filter_no_evict = CuckooFilter::builder()
        .capacity(1024)
        .max_evictions(0)
        .build()
        .unwrap();

    let filter_10_evict = CuckooFilter::builder()
        .capacity(1024)
        .max_evictions(10)
        .build()
        .unwrap();

    let filter_100_evict = CuckooFilter::builder()
        .capacity(1024)
        .max_evictions(100)
        .build()
        .unwrap();

    let mut no_evict_count = 0;
    let mut evict_10_count = 0;
    let mut evict_100_count = 0;

    for i in 0..1024 {
        if filter_no_evict.insert(&i).is_ok() {
            no_evict_count += 1;
        }
        if filter_10_evict.insert(&i).is_ok() {
            evict_10_count += 1;
        }
        if filter_100_evict.insert(&i).is_ok() {
            evict_100_count += 1;
        }
    }

    // Filter with evictions should accommodate more items
    assert!(no_evict_count < evict_10_count);
    assert!(evict_10_count < evict_100_count);
    assert_eq!(filter_no_evict.len(), no_evict_count);
    assert_eq!(filter_10_evict.len(), evict_10_count);
    assert_eq!(filter_100_evict.len(), evict_100_count);
}

#[test]
fn test_fingerprint_sizes() {
    let sizes = [4, 8, 16, 32];

    for &size in &sizes {
        let filter = CuckooFilter::builder()
            .capacity(1024)
            .fingerprint_size(size)
            .build()
            .unwrap();

        // insert items to ensure the filter is fully loaded
        let mut i = 0;
        while filter.len() < 1024 {
            let _ = filter.insert(&i);
            i += 1;
        }

        // test false positive rate
        let non_existing_items = 10000..110000;
        let false_positives = non_existing_items.filter(|i| filter.contains(i)).count();
        // Calculate the expected false positive rate (FPR) based on fingerprint size
        let expected_fpr = 1.0 - (1.0 - 1.0 / (1u64 << size) as f64).powi(8);
        let fpr = false_positives as f64 / 100000.0;
        if size == 4 {
            assert_eq!(expected_fpr, 0.4032805261667818);
        } else if size == 8 {
            assert_eq!(expected_fpr, 0.030826075519044704);
        } else if size == 16 {
            assert_eq!(expected_fpr, 0.00012206379344092966);
        } else if size == 32 {
            assert_eq!(expected_fpr, 0.000000001862645149230957);
        }
        // Allow a small margin due to randomness
        let tolerance = expected_fpr * 0.1 + 0.0001;

        assert!(
            (fpr - expected_fpr).abs() < tolerance,
            "Observed FPR ({fpr}) deviates too much from expected FPR ({expected_fpr}) for fingerprint size {size} ({false_positives} false positives)"
        );
    }
}

#[test]
fn test_bucket_sizes() {
    let sizes = [1, 2, 4, 8];

    for &size in &sizes {
        let filter = CuckooFilter::builder()
            .capacity(1024)
            .bucket_size(size)
            .build()
            .unwrap();

        // Should be able to insert items regardless of bucket size
        for i in 0..100 {
            assert!(filter.insert(&i).is_ok());
        }

        // Should be able to find all items
        for i in 0..100 {
            assert!(filter.contains(&i));
        }

        assert_eq!(filter.len(), 100);
    }
}

#[test]
fn test_custom_hasher() {
    // Test that we can use different hashers
    let filter = CuckooFilterBuilder::<AHasher>::default()
        .capacity(1024)
        .build()
        .unwrap();

    let items = test_items(100);
    for item in &items {
        assert!(filter.insert(item).is_ok());
    }

    for item in &items {
        assert!(filter.contains(item));
    }

    assert_eq!(filter.len(), 100);
}
