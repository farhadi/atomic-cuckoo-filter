use atomic_cuckoo_filter::CuckooFilter;

fn fill_until_stuck(filter: &CuckooFilter, failure_streak_limit: usize) {
    let mut i = 0usize;
    let mut consecutive_failures = 0usize;
    while consecutive_failures < failure_streak_limit {
        if filter.insert(&i).is_ok() {
            consecutive_failures = 0;
        } else {
            consecutive_failures += 1;
        }
        i += 1;
    }
}

#[test]
fn test_overlap_validation() {
    let result = CuckooFilter::builder()
        .bucket_size(4)
        .overlap(4) // invalid: must be < bucket_size
        .build();
    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("overlap must be less than bucket_size")
    );

    // Valid overlaps should build
    let filter_ok = CuckooFilter::builder()
        .capacity(1024)
        .bucket_size(4)
        .overlap(1)
        .build()
        .unwrap();
    assert_eq!(filter_ok.len(), 0);
}

#[test]
fn test_overlap_load_factor_comparison() {
    let filter_plain = CuckooFilter::builder()
        .capacity(32 * 1024)
        .fingerprint_size(8)
        .bucket_size(4)
        .max_evictions(200)
        .overlap(0)
        .build()
        .unwrap();
    let filter_overlap = CuckooFilter::builder()
        .capacity(32 * 1024)
        .fingerprint_size(8)
        .bucket_size(4)
        .max_evictions(200)
        .overlap(3)
        .build()
        .unwrap();

    // Fill until we appear stuck (several consecutive failures)
    fill_until_stuck(&filter_plain, 10);
    fill_until_stuck(&filter_overlap, 10);

    let load_plain = filter_plain.len() as f64 / filter_plain.capacity() as f64;
    let load_overlap = filter_overlap.len() as f64 / filter_overlap.capacity() as f64;

    println!("load factors => plain: {load_plain:.4}, overlap(3): {load_overlap:.4}");

    assert!(load_plain < load_overlap);
    assert!(load_plain > 0.70);
    assert!(load_overlap > 0.70);
    assert!(load_plain <= 1.0);
    assert!(load_overlap <= 1.0);
}
