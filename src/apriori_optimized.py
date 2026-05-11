import time
import tracemalloc
import math
from itertools import combinations


# ─────────────────────────────────────────────
# DATASET LOADER
# ─────────────────────────────────────────────

def load_dataset(filepath):

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

    tidlists = build_tidlists(partition)

    local_frequent = {}

    # LEVEL 1

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

    DATASET_PATH = "connect.dat"

    MIN_SUP = 0.95

    N_PARTITIONS = 2

    MAX_K = 4

    transactions = load_dataset(DATASET_PATH)

    result = apriori_optimized(
        transactions,
        min_sup=MIN_SUP,
        n_partitions=N_PARTITIONS,
        max_k=MAX_K
    )

    m = result["metrics"]

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