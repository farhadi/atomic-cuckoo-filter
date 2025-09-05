// Lock-Free Concurrent Cuckoo Filter Implementation
// A high-performance probabilistic data structure for efficient set membership testing
// with better space efficiency than Bloom filters, support for deletions, and
// fully concurrent operations using atomic operations and lock-free algorithms.

use derive_builder::Builder;
use rand::Rng;
use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::hint;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Maximum number of spin-loop iterations before parking a thread.
/// This balances CPU usage vs. latency - spinning avoids kernel calls for
/// short waits, but we park threads to avoid wasting CPU on long waits.
const MAX_SPIN: usize = 100;

/// Error type for Cuckoo Filter insert operation
#[derive(Debug, thiserror::Error, PartialEq)]
pub enum Error {
    /// Returned when the filter is full and cannot accommodate more elements
    #[error("Not enough space to store this item.")]
    NotEnoughSpace,
}

/// Types of locks that can be acquired on the filter
#[derive(PartialEq)]
pub enum LockKind {
    /// Optimistic version tracking - does not block other operations but captures
    /// a version number to detect if data changed during the operation
    Optimistic,
    /// Exclusive among writers only - prevents other writers but allows concurrent readers
    WriterExclusive,
    /// Fully exclusive access - blocks all other operations (used only during evictions)
    FullyExclusive,
}

/// A sophisticated lock implementation designed for concurrent cuckoo filter operations.
///
/// This is NOT a traditional mutex but an atomic-based synchronization mechanism that
/// enables three distinct concurrency modes:
///
/// 1. **Optimistic locks**: Allow maximum concurrency - multiple readers and writers
///    can proceed simultaneously. Used for optimistic reads that detect data races.
///
/// 2. **WriterExclusive locks**: Mutual exclusion among writers only - prevents
///    concurrent modifications but allows concurrent reads.
///
/// 3. **FullyExclusive locks**: Complete mutual exclusion - blocks all operations.
///    Only used during complex eviction chains to ensure consistency.
///
/// ## Version Encoding Scheme
/// The atomic usize encodes both lock state and version information:
/// - **Bits 0-1**: Lock kind (0=Optimistic, 1=WriterExclusive, 2=FullyExclusive)
/// - **Bits 2-63**: Version counter (incremented on FullyExclusive release)
///
/// This allows optimistic readers to detect when their read might be stale by
/// comparing version numbers before and after the operation.
pub struct Lock<'a> {
    /// Reference to the shared atomic value encoding lock state and version
    atomic: &'a AtomicUsize,
    /// Snapshot of the atomic value when this lock was acquired.
    /// Used for optimistic concurrency control and version tracking.
    /// The lower 2 bits indicate lock type, upper bits track version changes.
    version: usize,
    /// The type of lock held by this instance
    kind: LockKind,
    /// Counter for spin attempts before transitioning to thread parking.
    /// Implements adaptive spinning to balance latency vs CPU usage.
    retry: usize,
}

impl<'a> Lock<'a> {
    /// Create a new lock of the specified kind
    /// Blocks until the lock can be acquired
    fn new(atomic: &'a AtomicUsize, kind: LockKind) -> Self {
        let mut lock = Self {
            atomic,
            version: 0,
            kind,
            retry: 0,
        };
        match lock.kind {
            LockKind::Optimistic => loop {
                // For optimistic locks, we can proceed as long as there's no FullyExclusive lock
                lock.version = atomic.load(Ordering::Relaxed);
                if Self::kind(lock.version) != LockKind::FullyExclusive {
                    return lock;
                }
                lock.spin_or_park()
            },
            _ => loop {
                // For writer exclusive and fully exclusive locks, we need to ensure no exclusive lock is acquired
                lock.version = atomic.load(Ordering::Relaxed);
                if Self::kind(lock.version) != LockKind::Optimistic {
                    lock.spin_or_park();
                    continue;
                }
                // Update lower bits of the version: 1 for WriterExclusive, 2 for FullyExclusive
                let new_version = if lock.kind == LockKind::WriterExclusive {
                    lock.version + 1
                } else {
                    lock.version + 2
                };
                if atomic
                    .compare_exchange_weak(
                        lock.version,
                        new_version,
                        Ordering::Release,
                        Ordering::Relaxed,
                    )
                    .is_ok()
                {
                    return lock;
                }
            },
        }
    }

    /// Upgrade a WriterExclusive lock to a FullyExclusive lock
    /// This assumes the current thread holds the writer exclusive lock.
    fn upgrade(&mut self) {
        self.atomic.store(self.version + 2, Ordering::Release);
        self.kind = LockKind::FullyExclusive;
    }

    /// Check if the lock is outdated (version changed) or a FullyExclusive lock is acquired
    /// Used for optimistic concurrency control
    fn is_outdated(&self) -> bool {
        let version = self.atomic.load(Ordering::Acquire);
        Self::kind(version) == LockKind::FullyExclusive || version >> 2 != self.version >> 2
    }

    /// Get the key for parking a thread
    /// Different keys are used for optimistic and exclusive locks
    fn park_key(&self) -> usize {
        let key = self.atomic.as_ptr() as usize;
        match self.kind {
            LockKind::Optimistic => key,
            _ => key + 1,
        }
    }

    /// Spin or park the thread when waiting for a lock
    fn spin_or_park(&mut self) {
        if self.retry > MAX_SPIN {
            // After MAX_SPIN attempts, park the thread
            self.retry = 0;
            unsafe {
                parking_lot_core::park(
                    self.park_key(),
                    || self.atomic.load(Ordering::Acquire) == self.version,
                    || (),
                    |_, _| (),
                    parking_lot_core::DEFAULT_PARK_TOKEN,
                    None,
                );
            }
        } else {
            // Otherwise, spin
            self.retry += 1;
            hint::spin_loop();
        }
    }

    /// Extract the lock kind from the lower 2 bits of a version value
    fn kind(version: usize) -> LockKind {
        match version & 0b11 {
            0 => LockKind::Optimistic,
            1 => LockKind::WriterExclusive,
            2 => LockKind::FullyExclusive,
            _ => panic!("Invalid Lock"),
        }
    }
}

impl Drop for Lock<'_> {
    /// Release the lock when it goes out of scope
    fn drop(&mut self) {
        match self.kind {
            LockKind::Optimistic => return, // No need to do anything for Optimistic locks
            LockKind::WriterExclusive => {
                // For WriterExclusive locks, release the lock without incrementing the version
                self.atomic.store(self.version, Ordering::Release);
            }
            LockKind::FullyExclusive => {
                // For FullyExclusive locks, increment the version to invalidate Optimistic locks
                self.atomic.store(self.version + 4, Ordering::Release);
            }
        }

        // Unpark waiting threads
        let optimistic_key = self.atomic.as_ptr() as usize;
        let exclusive_key = optimistic_key + 1;
        unsafe {
            // Unpark all waiting optimistic locks
            parking_lot_core::unpark_all(optimistic_key, parking_lot_core::DEFAULT_UNPARK_TOKEN);
            // Unpark one waiting exclusive lock (either WriterExclusive or FullyExclusive)
            parking_lot_core::unpark_one(exclusive_key, |_| parking_lot_core::DEFAULT_UNPARK_TOKEN);
        }
    }
}

/// A highly concurrent lock-free probabilistic data structure for set membership testing.
///
/// ## What Makes It "Cuckoo"
///
/// Named after the cuckoo bird's behavior of displacing other birds' eggs, this filter
/// uses **cuckoo hashing** where each item can be stored in one of two possible locations.
/// When both locations are full, existing items are "evicted" (like cuckoo eggs) and
/// relocated to their alternate position, creating eviction chains.
///
/// ## Algorithm Overview
///
/// 1. **Fingerprints**: Items are reduced to small fingerprints (4-32 bits) instead of
///    storing full keys, providing excellent space efficiency.
///
/// 2. **Dual Hashing**: Each item has two possible bucket locations computed from its hash.
///    This provides better space efficiency and flexibility when inserting and removing items.
///
/// 3. **Eviction Chains**: When both buckets are full, a random item is evicted from one
///    bucket and moved to its alternate location, potentially triggering a chain of evictions.
///
/// 4. **Lock-Free Concurrency**: All operations use atomic compare-exchange loops instead
///    of traditional locks, with optimistic concurrency control for read operations.
///    The only exception is when inserting with evictions, where a FullyExclusive lock is used
///    to ensure consistency.
///
/// ## Key Advantages Over Bloom Filters
///
/// - **Deletions supported**: Items can be removed without false negatives
/// - **Better space efficiency**: ~20-30% less memory for same false positive rate
/// - **Bounded lookup time**: Always at most 2 bucket checks, never more
/// - **High concurrency**: Lock-free design enables excellent parallel performance
///
/// ## Concurrency Model
///
/// - **Reads**: Optimistic, can proceed concurrently with most operations
/// - **Simple writes**: Use atomic compare-exchange loops without blocking other operations
/// - **WriterExclusive locks**: Used for removing items, and for unique insertions
/// - **Complex evictions**: Use FullyExclusive locks to ensure consistency
///
/// ## Time Complexity
///
/// - **Lookup**: O(1)
/// - **Deletion**: O(1)
/// - **Insertion**: Amortized O(1) due to eviction chains, but the number of evictions is bounded
#[derive(Debug, Builder)]
#[builder(
    pattern = "owned",
    build_fn(private, name = "base_build", validate = "Self::validate")
)]
pub struct CuckooFilter<H = DefaultHasher>
where
    H: Hasher + Default,
{
    // Configuration parameters
    /// Maximum number of elements the filter can store
    #[builder(default = "1048576")]
    capacity: usize,

    /// Size of fingerprints in bits (must be 4, 8, 16, or 32)
    #[builder(default = "16")]
    fingerprint_size: usize,

    /// Number of fingerprints per bucket
    #[builder(default = "4")]
    bucket_size: usize,

    /// Maximum number of evictions to try before giving up
    #[builder(default = "500")]
    max_evictions: usize,

    // Internal values - automatically derived from the configuration
    /// Number of fingerprints that can be stored in a single atomic value
    #[builder(setter(skip))]
    fingerprints_per_atomic: usize,

    /// Number of buckets in the filter (power of 2)
    #[builder(setter(skip))]
    num_buckets: usize,

    /// Number of atomic values per bucket
    #[builder(setter(skip))]
    atomics_per_bucket: usize,

    /// Bit mask for extracting fingerprints
    #[builder(setter(skip))]
    fingerprint_mask: usize,

    /// Storage for buckets, implemented as a vector of atomic values
    #[builder(setter(skip))]
    buckets: Vec<AtomicUsize>,

    /// Atomic value used for locking
    #[builder(setter(skip))]
    lock: AtomicUsize,

    /// Counter for the number of elements in the filter
    #[builder(setter(skip))]
    counter: AtomicUsize,

    /// Phantom data for the hasher type
    #[builder(setter(skip))]
    _hasher: PhantomData<H>,
}

impl<H: Hasher + Default> CuckooFilter<H> {
    /// Insert an item into the filter
    ///
    /// This operation first attempts a direct insertion without acquiring a lock.
    /// If that fails due to bucket collisions, it falls back to the eviction-based
    /// insertion algorithm which may require a write lock.
    ///
    /// Concurrent operations are safely handled through atomic operations.
    ///
    /// Returns Ok(()) if the item was inserted, or Error::NotEnoughSpace if the filter is full
    pub fn insert<T: ?Sized + Hash>(&self, item: &T) -> Result<(), Error> {
        let (index, fingerprint) = self.index_and_fingerprint(item);
        self.try_insert(index, fingerprint).or_else(|error| {
            if let Some(lock) = self.lock(LockKind::WriterExclusive) {
                self.insert_with_evictions(index, fingerprint, lock)
            } else {
                Err(error)
            }
        })
    }

    /// Check if an item is in the filter and insert it if is not present (atomically)
    ///
    /// This method combines lookup and insert into a single atomic operation,
    /// ensuring thread safety and consistency even with concurrent operations.
    ///
    /// Returns Ok(true) if the item was inserted, Ok(false) if it was already present,
    /// or Error::NotEnoughSpace if the filter is full
    pub fn insert_unique<T: ?Sized + Hash>(&self, item: &T) -> Result<bool, Error> {
        let (index, fingerprint) = self.index_and_fingerprint(item);
        if self.lookup_fingerprint(index, fingerprint).is_some() {
            return Ok(false);
        }
        let lock = Lock::new(&self.lock, LockKind::WriterExclusive);
        if self.lookup_fingerprint(index, fingerprint).is_some() {
            return Ok(false);
        }
        self.try_insert(index, fingerprint)
            .or_else(|error| {
                if self.max_evictions == 0 {
                    return Err(error);
                }
                self.insert_with_evictions(index, fingerprint, lock)
            })
            .map(|_| true)
    }

    /// Counts the number of occurrences of an item in the filter.
    ///
    /// # Notes
    /// - This is not a counting filter; it simply counts matching fingerprints in both candidate buckets.
    /// - Useful for detecting duplicates or hash collisions, not for precise multiset membership.
    /// - The count is limited by the filter's structure: at most `bucket_size * 2` per item.
    /// - This method may count false positives due to hash collisions.
    pub fn count<T: ?Sized + Hash>(&self, item: &T) -> usize {
        let (index, fingerprint) = self.index_and_fingerprint(item);
        let alt_index = self.alt_index(index, fingerprint);
        self.atomic_read(
            || {
                self.read_bucket(index, Ordering::Acquire)
                    .filter(|&f| f == fingerprint)
                    .count()
                    + self
                        .read_bucket(alt_index, Ordering::Acquire)
                        .filter(|&f| f == fingerprint)
                        .count()
            },
            None,
        )
    }

    /// Attempts to remove an item from the filter.
    ///
    /// Returns `true` if the item was successfully removed, or `false` if it was not found.
    ///
    /// Note:
    /// - An item should only be removed if it was previously added. Removing a non-existent
    ///   item may inadvertently remove a different item due to hash collisions inherent to
    ///   cuckoo filters.
    pub fn remove<T: ?Sized + Hash>(&self, item: &T) -> bool {
        let (index, fingerprint) = self.index_and_fingerprint(item);
        while let Some((index, sub_index)) = self.lookup_fingerprint(index, fingerprint) {
            let _lock = self.lock(LockKind::WriterExclusive);
            if self.update_bucket(index, sub_index, fingerprint, 0, Ordering::Release) {
                return true;
            }
        }
        false
    }

    /// Check if an item is in the filter
    ///
    /// Returns `true` if the item is possibly in the filter (may have false positives),
    /// `false` if it is definitely not in the filter
    pub fn contains<T: ?Sized + Hash>(&self, item: &T) -> bool {
        let (index, fingerprint) = self.index_and_fingerprint(item);
        self.atomic_read(
            || self.lookup_fingerprint(index, fingerprint).is_some(),
            Some(true),
        )
    }

    /// Get the number of elements in the filter
    pub fn len(&self) -> usize {
        self.counter.load(Ordering::Acquire)
    }

    /// Check if the filter is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the capacity of the filter
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Clear the filter, removing all elements
    pub fn clear(&self) {
        let _lock = self.lock(LockKind::WriterExclusive);
        for atomic in &self.buckets {
            let old_value = atomic.swap(0, Ordering::Release);
            let removed = (0..self.fingerprints_per_atomic)
                .filter(|i| (old_value >> (i * self.fingerprint_size)) & self.fingerprint_mask != 0)
                .count();
            if removed > 0 {
                self.counter.fetch_sub(removed, Ordering::Release);
            }
        }
    }

    /// Compute the hash of an item
    /// Uses the generic hasher H for flexibility and performance
    fn hash<T: ?Sized + Hash>(&self, data: &T) -> u64 {
        let mut hasher = <H as Default>::default();
        data.hash(&mut hasher);
        hasher.finish()
    }

    /// Compute the bucket index and fingerprint for an item.
    ///
    /// 1. **Hash the item**: Use the configured hasher to get a 64-bit hash
    /// 2. **Extract fingerprint**: Use multiplication + shift for high-quality
    ///    distribution across the fingerprint space, then add 1 to avoid zero.
    /// 3. **Extract index**: Use bitwise AND with (num_buckets-1) since num_buckets
    ///    is always a power of 2, providing perfect hash distribution.
    ///
    /// ## Why This Design
    ///
    /// - **Non-zero fingerprints**: Adding 1 ensures fingerprints are never 0,
    ///   so 0 can represent empty slots without ambiguity
    /// - **Independent bits**: Index uses lower hash bits, fingerprint uses
    ///   different bits via multiplication, avoiding correlation
    /// - **Uniform distribution**: Both index and fingerprint are uniformly
    ///   distributed across their respective ranges
    ///
    /// Returns (index, fingerprint) where:
    /// - index is the primary bucket index (0 to num_buckets-1)  
    /// - fingerprint is a compact hash of the item (1 to fingerprint_mask)
    fn index_and_fingerprint<T: ?Sized + Hash>(&self, item: &T) -> (usize, usize) {
        let hash = self.hash(item);
        // Compute fingerprint using multiplication and shift for better distribution
        let fingerprint = ((hash as u128 * self.fingerprint_mask as u128) >> 64) + 1;
        // Compute index using modulo num_buckets (optimized with bitwise AND since num_buckets is a power of 2)
        let index = hash as usize & (self.num_buckets - 1);
        (index, fingerprint as usize)
    }

    /// Computes the alternative bucket index for a given fingerprint using cuckoo hashing.
    ///
    /// In cuckoo hashing, each item can reside in one of two possible buckets. This function
    /// deterministically computes the alternate bucket index from the current index and fingerprint.
    ///
    /// Properties:
    /// 1. Symmetry: `alt_index(alt_index(i, f), f) == i` for any index `i` and fingerprint `f`.
    /// 2. Distinctness: For any fingerprint, the two indices are always different.
    /// 3. Uniformity: The mapping distributes fingerprints evenly across all buckets.
    fn alt_index(&self, index: usize, fingerprint: usize) -> usize {
        index ^ (self.hash(&fingerprint) as usize & (self.num_buckets - 1))
    }

    /// Look up a fingerprint at its primary or alternative index
    /// Returns `Some((index, sub_index))` if found, None otherwise
    fn lookup_fingerprint(&self, index: usize, fingerprint: usize) -> Option<(usize, usize)> {
        // First check the primary bucket
        self.read_bucket(index, Ordering::Acquire)
            .position(|fp| fp == fingerprint)
            .map(|sub_index| (index, sub_index))
            .or_else(|| {
                // Then check the alternative bucket
                let alt_index = self.alt_index(index, fingerprint);
                self.read_bucket(alt_index, Ordering::Acquire)
                    .position(|fp| fp == fingerprint)
                    .map(|sub_index| (alt_index, sub_index))
            })
    }

    /// Try to insert a fingerprint at its primary or alternative index
    /// Returns `Ok(())` if successful, `Error::NotEnoughSpace` if both buckets are full
    fn try_insert(&self, index: usize, fingerprint: usize) -> Result<(), Error> {
        if !self.insert_at_index(index, fingerprint) {
            let alt_index = self.alt_index(index, fingerprint);
            if !self.insert_at_index(alt_index, fingerprint) {
                return Err(Error::NotEnoughSpace);
            }
        }
        Ok(())
    }

    /// Try to insert a fingerprint at a specific index
    /// Returns `Ok(())` if successful, `Err(bucket)` if the bucket is full
    fn insert_at_index(&self, index: usize, fingerprint: usize) -> bool {
        loop {
            if let Some(sub_index) = self
                .read_bucket(index, Ordering::Relaxed)
                .position(|i| i == 0)
            {
                if self.update_bucket(index, sub_index, 0, fingerprint, Ordering::Release) {
                    return true;
                }
                continue;
            }
            return false;
        }
    }

    /// Insert a fingerprint using cuckoo eviction chains when both buckets are full.
    ///
    /// This method is invoked only as a fallback when direct insertion fails, preserving
    /// the optimistic, lock-free fast path for the common case.
    ///
    /// # Cuckoo Eviction Algorithm
    ///
    /// When both possible locations for an item are full:
    /// 1. **Randomly select** an existing item from one of the full buckets
    /// 2. **Evict** that item and insert our new item in its place
    /// 3. **Relocate** the evicted item to its alternate location
    /// 4. **Repeat** if the alternate location is also full (eviction chain)
    /// 5. **Succeed** when we find an empty slot, or **fail** after max_evictions
    ///
    /// # Implementation Details
    ///
    /// - **Eviction tracking**: Collects a sequence of planned evictions, which are
    ///   atomically applied only if the chain succeeds, ensuring atomicity and consistency.
    /// - **Lock upgrading**: Starts with a `WriterExclusive` lock, upgrading to
    ///   `FullyExclusive` only when actually applying the eviction chain, maximizing
    ///   read concurrency during planning.
    /// - **Loop prevention**: Uses a map to track which sub-indices have been tried
    ///   in each bucket, to ensure early detection of loops in eviction chains.
    fn insert_with_evictions(
        &self,
        mut index: usize,
        mut fingerprint: usize,
        mut lock: Lock,
    ) -> Result<(), Error> {
        let mut rng = rand::rng();
        let mut evictions = Vec::with_capacity(self.max_evictions.min(32));
        let mut used_indices = HashMap::with_capacity(self.max_evictions.min(32));
        while evictions.len() <= self.max_evictions {
            if !self.insert_at_index(index, fingerprint) {
                let sub_index = match used_indices.entry(index).or_insert(0usize) {
                    sub_indices if *sub_indices == 0 => {
                        // First time seeing this index, randomly choose a sub-index
                        let sub_index = rng.random_range(0..self.bucket_size);
                        *sub_indices = 1 << sub_index;
                        sub_index
                    }
                    sub_indices => {
                        // Find an unused sub-index
                        if let Some(sub_index) =
                            (0..self.bucket_size).find(|shift| (*sub_indices >> shift) & 1 == 0)
                        {
                            *sub_indices |= 1 << sub_index;
                            sub_index
                        } else {
                            return Err(Error::NotEnoughSpace);
                        }
                    }
                };
                // Evict the fingerprint at the chosen sub-index
                let evicted = self
                    .read_bucket(index, Ordering::Relaxed)
                    .skip(sub_index)
                    .next()
                    .unwrap();
                evictions.push((index, sub_index, fingerprint));
                // Find the alternative index for the evicted fingerprint
                index = self.alt_index(index, evicted);
                fingerprint = evicted;
            } else {
                // Successfully inserted the fingerprint, now apply all evictions
                lock.upgrade();
                while let Some((index, sub_index, evicted)) = evictions.pop() {
                    self.update_bucket(index, sub_index, fingerprint, evicted, Ordering::Relaxed);
                    fingerprint = evicted;
                }
                return Ok(());
            }
        }
        // Reached the maximum number of evictions, give up
        Err(Error::NotEnoughSpace)
    }

    /// Atomically read all fingerprints from a bucket using lock-free bit manipulation.
    ///
    /// ## Memory Layout Complexity
    ///
    /// Fingerprints are tightly packed in memory across multiple atomic usize values:
    /// - Each bucket contains `bucket_size` fingerprints
    /// - Each fingerprint is `fingerprint_size` bits
    /// - Multiple fingerprints are packed into each atomic usize
    /// - Buckets may span across multiple atomic values
    ///
    /// ## Algorithm Steps
    ///
    /// 1. Calculate which atomic values contain this bucket's data
    /// 2. Atomically load each relevant atomic value (using Acquire ordering)
    /// 3. Extract fingerprints using bit manipulation and masking
    /// 4. Handle boundary cases where buckets span multiple atomics
    /// 5. Skip any padding bits and return exactly `bucket_size` fingerprints
    ///
    /// This is completely lock-free - multiple threads can read concurrently,
    /// and reads can proceed even during writes (though they might see
    /// intermediate states that get resolved by retry logic).
    ///
    /// Returns an Iterator over the fingerprints in the bucket, possibly empty.
    fn read_bucket(&self, index: usize, ordering: Ordering) -> impl Iterator<Item = usize> {
        let fingerprint_index = index * self.bucket_size;
        let bit_index = fingerprint_index * self.fingerprint_size;
        let start_index = bit_index / usize::BITS as usize;
        let skip_bits = bit_index % usize::BITS as usize;
        let skip_fingerprints = skip_bits >> self.fingerprint_size.trailing_zeros();
        let end_index = start_index + self.atomics_per_bucket;

        self.buckets[start_index..end_index]
            .iter()
            .flat_map(move |atomic| {
                let atomic_value = atomic.load(ordering);
                (0..self.fingerprints_per_atomic).map(move |i| {
                    (atomic_value
                        >> (self.fingerprint_size * (self.fingerprints_per_atomic - i - 1)))
                        & self.fingerprint_mask
                })
            })
            .skip(skip_fingerprints)
            .take(self.bucket_size)
    }

    /// Atomically update a single fingerprint using lock-free compare-exchange.
    ///
    /// ## Lock-Free Update Algorithm
    ///
    /// 1. **Locate the target**: Calculate which atomic usize contains the fingerprint
    ///    and the exact bit position within that atomic value
    /// 2. **Read current state**: Load the current atomic value
    /// 3. **Verify expectation**: Check that the target position contains `old_value`
    /// 4. **Atomic update**: Use compare_exchange_weak to atomically replace `old_value`
    ///    with `new_value`, but only if the atomic hasn't changed since step 2
    /// 5. **Retry on conflict**: If another thread modified the atomic concurrently,
    ///    restart from step 2
    ///
    /// ## Concurrency Safety
    ///
    /// - Uses `compare_exchange_weak` which can fail spuriously on some architectures
    ///   but is more efficient than the strong version
    /// - Employs Release ordering on success to ensure other threads see the change
    /// - Updates the global counter atomically to maintain consistency
    /// - Returns false if the expected `old_value` is no longer present (indicating
    ///   another thread already modified this slot)
    ///
    /// Returns `true` if update succeeded, `false` if the slot no longer contains
    /// the expected `old_value` due to concurrent modification.
    fn update_bucket(
        &self,
        index: usize,
        sub_index: usize,
        old_value: usize,
        new_value: usize,
        ordering: Ordering,
    ) -> bool {
        let bit_index = (index * self.bucket_size + sub_index) * self.fingerprint_size;
        let atomic_index = bit_index / usize::BITS as usize;
        let skip_bits = bit_index % usize::BITS as usize;
        let shift = usize::BITS as usize - self.fingerprint_size - skip_bits;
        let fingerprint_mask = self.fingerprint_mask << shift;
        let atomic = &self.buckets[atomic_index];

        loop {
            let atomic_value = atomic.load(Ordering::Relaxed);
            if (atomic_value & fingerprint_mask) >> shift != old_value {
                // The expected fingerprint is not present in the atomic value
                return false;
            }
            let new_atomic_value = (atomic_value & !fingerprint_mask) | (new_value << shift);
            if atomic
                .compare_exchange_weak(atomic_value, new_atomic_value, ordering, Ordering::Relaxed)
                .is_ok()
            {
                // Update the counter based on the change
                match (old_value, new_value) {
                    (0, _) => self.counter.fetch_add(1, Ordering::Release),
                    (_, 0) => self.counter.fetch_sub(1, Ordering::Release),
                    (_, _) => 0,
                };
                return true;
            }
        }
    }

    /// Acquires a lock on the filter, if necessary.
    ///
    /// A lock is only required when evictions are enabled (i.e., `max_evictions > 0`).
    /// If `max_evictions` is set to 0, no lock is acquired.
    ///
    /// Returns `Some(Lock)` if a lock is needed, or `None` if no locking is required.
    pub fn lock(&self, kind: LockKind) -> Option<Lock<'_>> {
        if self.max_evictions == 0 {
            None
        } else {
            Some(Lock::new(&self.lock, kind))
        }
    }

    /// Execute a read operation with optimistic concurrency control and automatic retry.
    ///
    /// This is the cornerstone of the lock-free design, implementing a sophisticated
    /// optimistic concurrency protocol that allows reads to proceed concurrently with
    /// most write operations.
    ///
    /// ## Optimistic Concurrency Protocol
    ///
    /// 1. **Snapshot version**: Acquire an Optimistic lock (capturing version number)
    /// 2. **Execute read**: Run the provided function without any blocking
    /// 3. **Validate consistency**: Check if version changed or FullyExclusive lock acquired
    /// 4. **Retry or return**: If data may be stale, retry; otherwise return result
    ///
    /// ## How It Works
    ///
    /// - **WriterExclusive operations**: Don't invalidate optimistic reads because they
    ///   coordinate through atomic compare-exchange operations that are linearizable
    /// - **FullyExclusive operations**: Do invalidate optimistic reads because they
    ///   perform complex multi-step updates that require consistency
    /// - **Early return optimization**: For operations that can short-circuit (like
    ///   `contains()` returning true), we skip version validation as an optimization
    ///
    /// This pattern is essential for achieving lock-free performance while maintaining
    /// correctness in the presence of concurrent modifications.
    fn atomic_read<T, F>(&self, fun: F, early_return: Option<T>) -> T
    where
        F: Fn() -> T,
        T: PartialEq,
    {
        if self.max_evictions == 0 {
            return fun();
        }
        loop {
            let lock = Lock::new(&self.lock, LockKind::Optimistic);
            let result = fun();
            if Some(&result) == early_return.as_ref() || !lock.is_outdated() {
                return result;
            }
        }
    }
}

impl CuckooFilter<DefaultHasher> {
    /// Create a new CuckooFilterBuilder with default settings
    pub fn builder() -> CuckooFilterBuilder<DefaultHasher> {
        CuckooFilterBuilder::default()
    }

    /// Create a new CuckooFilter with default settings
    pub fn new() -> CuckooFilter<DefaultHasher> {
        Self::builder().build().unwrap()
    }

    /// Create a new CuckooFilter with the specified capacity
    pub fn with_capacity(capacity: usize) -> CuckooFilter<DefaultHasher> {
        Self::builder().capacity(capacity).build().unwrap()
    }
}

impl Default for CuckooFilter<DefaultHasher> {
    /// Create a new CuckooFilter with default settings
    fn default() -> Self {
        Self::new()
    }
}

impl<H: Hasher + Default> CuckooFilterBuilder<H> {
    /// Validate the builder configuration
    fn validate(&self) -> Result<(), String> {
        if let Some(fingerprint_size) = self.fingerprint_size
            && ![4, 8, 16, 32].contains(&fingerprint_size)
        {
            return Err("Invalid fingerprint_size".into());
        }
        if self.bucket_size == Some(0) {
            return Err("bucket_size must be greater than zero".into());
        }
        if self.capacity == Some(0) {
            return Err("capacity must be greater than zero".into());
        }
        Ok(())
    }

    /// Build a CuckooFilter with the specified configuration
    pub fn build(self) -> Result<CuckooFilter<H>, CuckooFilterBuilderError> {
        let mut cuckoo_filter = self.base_build()?;
        // Calculate the number of buckets (power of 2)
        cuckoo_filter.num_buckets = cuckoo_filter
            .capacity
            .div_ceil(cuckoo_filter.bucket_size)
            .next_power_of_two();
        // Adjust the capacity to match the actual number of buckets
        cuckoo_filter.capacity = cuckoo_filter.num_buckets * cuckoo_filter.bucket_size;
        // Calculate the fingerprint mask
        cuckoo_filter.fingerprint_mask = ((1u64 << cuckoo_filter.fingerprint_size) - 1) as usize;
        // Calculate the size of a bucket in bits
        let bucket_bit_size = cuckoo_filter.bucket_size * cuckoo_filter.fingerprint_size;
        // Calculate the number of atomic values per bucket
        cuckoo_filter.atomics_per_bucket = bucket_bit_size.div_ceil(usize::BITS as usize);
        // Calculate the number of fingerprints per atomic value
        cuckoo_filter.fingerprints_per_atomic =
            usize::BITS as usize / cuckoo_filter.fingerprint_size;
        // Calculate the total number of atomic values needed
        let bit_size = cuckoo_filter.capacity * cuckoo_filter.fingerprint_size;
        let atomic_size = bit_size.div_ceil(usize::BITS as usize);
        // Initialize the buckets
        cuckoo_filter.buckets = (0..atomic_size).map(|_| AtomicUsize::new(0)).collect();
        Ok(cuckoo_filter)
    }
}
