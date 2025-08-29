use atomic_cuckoo_filter::CuckooFilter;
use std::sync::Arc;
use std::thread;

#[test]
fn test_concurrent_reads() {
    let filter = Arc::new(CuckooFilter::with_capacity(1024));

    // Insert test data
    for i in 0..100 {
        assert!(filter.insert(&i).is_ok());
    }

    let mut handles = vec![];

    // Spawn multiple reader threads
    for _ in 0..5 {
        let filter_clone = Arc::clone(&filter);
        handles.push(thread::spawn(move || {
            for i in 0..100 {
                assert!(filter_clone.contains(&i));
            }
        }));
    }

    // All reads should succeed
    for handle in handles {
        handle.join().unwrap();
    }
}

#[test]
fn test_concurrent_insert() {
    let filter = Arc::new(CuckooFilter::with_capacity(10000));
    let mut handles = vec![];

    // Spawn writer threads
    for thread_id in 0..5 {
        let filter_clone = Arc::clone(&filter);
        handles.push(thread::spawn(move || {
            for i in 0..100 {
                let item = format!("thread_{thread_id}_item_{i}");
                filter_clone.insert(&item).unwrap();
            }
        }));
    }

    // Wait for all threads
    for handle in handles {
        handle.join().unwrap();
    }

    // check if all items are inserted
    for thread_id in 0..5 {
        for i in 0..100 {
            let item = format!("thread_{thread_id}_item_{i}");
            assert!(filter.contains(&item));
        }
    }

    // Should have inserted 500 items total
    assert_eq!(filter.len(), 500);
}

#[test]
fn test_concurrent_insert_unique() {
    let filter = Arc::new(CuckooFilter::with_capacity(131072));
    let mut handles = vec![];

    for _ in 0..5 {
        let filter_clone = filter.clone();
        handles.push(thread::spawn(move || {
            (0..100000)
                .filter(|i| filter_clone.insert_unique(i).unwrap())
                .count()
        }));
    }

    let inserted: usize = handles.into_iter().map(|h| h.join().unwrap()).sum();

    for i in 0..100000 {
        assert!(filter.contains(&i));
    }

    assert_eq!(inserted, filter.len());

    // inserted items might be less than 100000 due to false positives
    assert!(inserted <= 100000);
}

#[test]
fn concurrent_remove() {
    let filter = Arc::new(CuckooFilter::with_capacity(131072));
    let mut handles = vec![];

    for i in 0..100000 {
        assert!(filter.insert(&i).is_ok());
    }

    for _ in 0..5 {
        let f = filter.clone();
        handles.push(thread::spawn(move || {
            (0..100000).filter(|i| f.remove(i)).count()
        }));
    }

    let removed: usize = handles.into_iter().map(|h| h.join().unwrap()).sum();
    assert_eq!(removed, 100000)
}

#[test]
fn test_concurrent_insert_and_remove() {
    let filter = Arc::new(CuckooFilter::with_capacity(10000));
    let mut handles = vec![];

    // Spawn writer threads
    for thread_id in 0..5 {
        let filter_clone = Arc::clone(&filter);
        handles.push(thread::spawn(move || {
            for i in 0..100 {
                let item = format!("thread_{thread_id}_item_{i}");
                filter_clone.insert(&item).unwrap();
            }
        }));
    }

    // Spawn remover threads
    for thread_id in 0..5 {
        let filter_clone = Arc::clone(&filter);
        handles.push(thread::spawn(move || {
            for i in 0..100 {
                let item = format!("thread_{thread_id}_item_{i}");
                while !filter_clone.remove(&item) {}
            }
        }));
    }

    // Wait for all threads
    for handle in handles {
        handle.join().unwrap();
    }

    // Should have removed all items
    assert_eq!(filter.len(), 0);
}
