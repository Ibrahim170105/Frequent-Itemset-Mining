"""
Optimized Apriori Algorithm for Frequent Itemset Mining

This module implements an optimized version of the Apriori algorithm using:
- Vertical data format (TID lists) for efficient support counting
- Dataset partitioning to reduce memory usage
- Early termination when too many candidates are generated

The algorithm mines frequent itemsets from transactional data and returns
performance metrics including execution time and memory usage.
"""

import time
import tracemalloc
import math
from itertools import combinations


# ─────────────────────────────────────────────
# DATASET LOADER
# ─────────────────────────────────────────────

def load_dataset(filepath):
    """
    Load transactions from a dataset file.

    Args:
        filepath (str): Path to the dataset file. Each line represents a transaction
                       with space-separated integer item IDs.

    Returns:
        list: List of frozensets, where each frozenset represents a transaction.
    """
    transactions = []

    try:
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()

                if line:
                    transactions.append(
                        frozenset(map(int, line.split()))
                    )

    except FileNotFoundError:
        print(f"Dataset file not found: {filepath}")
        return []

    return transactions


# ─────────────────────────────────────────────
# BUILD TID LISTS
# ─────────────────────────────────────────────

def build_tidlists(transactions):
    """
    Build TID (Transaction ID) lists for each item.

    Args:
        transactions (list): List of frozensets representing transactions.

    Returns:
        dict: Dictionary where keys are item IDs and values are sets of transaction IDs
              containing that item.
    """
    tidlists = {}

    for tid, transaction in enumerate(transactions):

        for item in transaction:

            if item not in tidlists:
                tidlists[item] = set()

            tidlists[item].add(tid)

    return tidlists


# ─────────────────────────────────────────────
# SUPPORT USING TID INTERSECTION
# ─────────────────────────────────────────────

def tidlist_support(tidlists, candidate):
    """
    Calculate the support count of a candidate itemset using TID list intersection.

    Args:
        tidlists (dict): TID lists for individual items.
        candidate (frozenset): Candidate itemset.

    Returns:
        int: Support count (number of transactions containing all items in candidate).
    """
    if not candidate:
        return 0

    items = list(candidate)

    result = tidlists.get(items[0], set()).copy()

    for item in items[1:]:

        result &= tidlists.get(item, set())

        if not result:
            return 0

    return len(result)


# ─────────────────────────────────────────────
# CANDIDATE GENERATION
# ─────────────────────────────────────────────

def generate_candidates(frequent_itemsets, k):
    """
    Generate candidate itemsets of size k from frequent itemsets of size k-1.

    Args:
        frequent_itemsets (list): List of frequent itemsets of size k-1.
        k (int): Size of candidate itemsets to generate.

    Returns:
        list: List of candidate itemsets of size k.
    """
    candidates = set()

    itemsets = sorted([sorted(iset) for iset in frequent_itemsets])

    n = len(itemsets)

    for i in range(n):

        for j in range(i + 1, n):

            if itemsets[i][:k - 2] == itemsets[j][:k - 2]:

                candidate = frozenset(itemsets[i]) | frozenset(itemsets[j])

                if len(candidate) == k:
                    candidates.add(candidate)

            else:
                break

    return list(candidates)


# ─────────────────────────────────────────────
# PRUNING
# ─────────────────────────────────────────────

def prune_candidates(candidates,
                     frequent_prev,
                     k):
    """
    Prune candidate itemsets using the Apriori property.

    Args:
        candidates (list): List of candidate itemsets.
        frequent_prev (set): Set of frequent itemsets of size k-1.
        k (int): Size of candidate itemsets.

    Returns:
        list: List of pruned candidate itemsets where all subsets of size k-1 are frequent.
    """
    pruned = []

    for candidate in candidates:

        subsets = [
            frozenset(s)
            for s in combinations(candidate, k - 1)
        ]

        if all(s in frequent_prev for s in subsets):
            pruned.append(candidate)

    return pruned


# ─────────────────────────────────────────────
# LOCAL PARTITION MINING
# ─────────────────────────────────────────────

def mine_partition(partition,
                   local_min_count,
                   max_k):
    """
    Mine frequent itemsets from a single partition using optimized Apriori.

    Args:
        partition (list): List of transactions in this partition.
        local_min_count (int): Minimum support count for this partition.
        max_k (int): Maximum size of itemsets to mine.

    Returns:
        set: Set of frequent itemsets found in this partition.
    """
    tidlists = build_tidlists(partition)

    local_frequent = {}

    # LEVEL 1: Find frequent single items
    l1 = {}

    for item, tids in tidlists.items():

        sup = len(tids)

        if sup >= local_min_count:
            l1[frozenset([item])] = sup

    local_frequent.update(l1)

    prev_frequent = l1
    k = 2

    while prev_frequent and k <= max_k:

        ck = generate_candidates(
            list(prev_frequent.keys()),
            k
        )

        ck = prune_candidates(
            ck,
            set(prev_frequent.keys()),
            k
        )

        print(f"Partition Level {k}: {len(ck):,} candidates")

        if len(ck) > 100000:
            print("Too many candidates — stopping early")
            break

        if not ck:
            break

        lk = {}

        for candidate in ck:

            sup = tidlist_support(
                tidlists,
                candidate
            )

            if sup >= local_min_count:
                lk[candidate] = sup

        local_frequent.update(lk)

        prev_frequent = lk

        k += 1

    return set(local_frequent.keys())


# ─────────────────────────────────────────────
# MAIN OPTIMIZED APRIORI
# ─────────────────────────────────────────────

def apriori_optimized(transactions,
                      min_sup,
                      n_partitions=2,
                      max_k=4):
    """
    Run the optimized Apriori algorithm with partitioning for frequent itemset mining.

    This implementation uses vertical data format (TID lists) and partitions the dataset
    to reduce memory usage and improve performance.

    Args:
        transactions (list): List of frozensets representing transactions.
        min_sup (float): Minimum support threshold (0.0 to 1.0).
        n_partitions (int): Number of partitions to divide the dataset into.
        max_k (int): Maximum size of itemsets to mine.

    Returns:
        dict: Results containing frequent itemsets, rules, and performance metrics.
    """
    if not transactions:
        raise ValueError("Dataset is empty")

    n = len(transactions)

    min_count = math.ceil(min_sup * n)

    tracemalloc.start()

    start_time = time.perf_counter()

    # ─────────────────────────────
    # PARTITIONING
    # ─────────────────────────────

    part_size = math.ceil(n / n_partitions)

    partitions = [
        transactions[i:i + part_size]
        for i in range(0, n, part_size)
    ]

    global_candidates = set()

    local_candidate_counts = []

    # ─────────────────────────────
    # LOCAL MINING
    # ─────────────────────────────

    for partition in partitions:

        local_min_count = math.ceil(
            min_sup * len(partition)
        )

        local_frequent = mine_partition(
            partition,
            local_min_count,
            max_k
        )

        global_candidates |= local_frequent

        local_candidate_counts.append(
            len(local_frequent)
        )

    total_candidates = len(global_candidates)

    # ─────────────────────────────
    # GLOBAL VERIFICATION
    # ─────────────────────────────

    global_tidlists = build_tidlists(transactions)

    all_frequent = {}

    for candidate in global_candidates:

        sup = tidlist_support(
            global_tidlists,
            candidate
        )

        if sup >= min_count:
            all_frequent[candidate] = sup

    elapsed = time.perf_counter() - start_time

    _, peak_mem = tracemalloc.get_traced_memory()

    tracemalloc.stop()

    # RULE GENERATION DISABLED
    rules = []

    return {

        "frequent_itemsets": all_frequent,

        "rules": rules,

        "metrics": {

            "execution_time_s":
                round(elapsed, 4),

            "peak_memory_mb":
                round(peak_mem / (1024 ** 2), 4),

            "n_frequent_itemsets":
                len(all_frequent),

            "n_candidates_total":
                total_candidates,

            "n_partitions":
                n_partitions,

            "local_candidates_per_partition":
                local_candidate_counts
        }
    }


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":

    # Configuration parameters
    DATASET_PATH = "connect.dat"  # ← change to: chess.dat / accidents.dat

    MIN_SUP = 0.95   # ← change threshold (0.0 – 1.0)
    
    N_PARTITIONS = 2

    MAX_K = 4

    # Load dataset
    transactions = load_dataset(DATASET_PATH)

    # Run optimized Apriori algorithm
    result = apriori_optimized(
        transactions,
        min_sup=MIN_SUP,
        n_partitions=N_PARTITIONS,
        max_k=MAX_K
    )

    # Extract metrics for display
    m = result["metrics"]

    # Display results
    print(f"\n{'═'*60}")
    print("OPTIMIZED APRIORI RESULTS")
    print(f"{'═'*60}")

    print(f"Dataset              : {DATASET_PATH}")
    print(f"Transactions         : {len(transactions):,}")
    print(f"Minimum Support      : {MIN_SUP:.0%}")
    print(f"Partitions           : {N_PARTITIONS}")
    print(f"Maximum k            : {MAX_K}")

    print(f"\nExecution Time       : {m['execution_time_s']} s")
    print(f"Peak Memory          : {m['peak_memory_mb']} MB")

    print(f"Frequent Itemsets    : {m['n_frequent_itemsets']:,}")
    print(f"Global Candidates    : {m['n_candidates_total']:,}")

    print("\nLocal Candidates Per Partition:")

    for i, count in enumerate(
            m["local_candidates_per_partition"], 1):

        print(f"Partition {i}: {count:,}")

    print(f"{'═'*60}\n")