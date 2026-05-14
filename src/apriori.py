import time
import tracemalloc
import math
from itertools import combinations


# ─────────────────────────────────────────────
# DATASET LOADER
# ─────────────────────────────────────────────

def load_dataset(filepath: str) -> list[frozenset]:
    transactions = []

    try:
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    transactions.append(frozenset(map(int, line.split())))

    except FileNotFoundError:
        print(f"Dataset file not found: {filepath}")
        return []

    return transactions


# ─────────────────────────────────────────────
# SUPPORT COUNTING
# ─────────────────────────────────────────────

def count_support(transactions, candidates):

    support_count = {c: 0 for c in candidates}

    for transaction in transactions:
        for candidate in candidates:
            if candidate.issubset(transaction):
                support_count[candidate] += 1

    return support_count


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

def prune_candidates(candidates, frequent_prev, k):

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
# MAIN APRIORI
# ─────────────────────────────────────────────

def apriori(transactions,
            min_sup,
            max_k=4):

    if not transactions:
        raise ValueError("Dataset is empty")

    if not (0 < min_sup <= 1):
        raise ValueError("min_sup must be between 0 and 1")

    n = len(transactions)

    min_count = math.ceil(min_sup * n)

    tracemalloc.start()
    start_time = time.perf_counter()

    all_frequent = {}

    total_candidates = 0
    candidates_per_level = {}

    # ─────────────────────────────
    # LEVEL 1
    # ─────────────────────────────

    all_items = set(item for t in transactions for item in t)

    c1 = [frozenset([item]) for item in all_items]

    total_candidates += len(c1)
    candidates_per_level[1] = len(c1)

    support_c1 = count_support(transactions, c1)

    l1 = {
        iset: cnt
        for iset, cnt in support_c1.items()
        if cnt >= min_count
    }

    all_frequent.update(l1)

    prev_frequent = l1
    k = 2

    # ─────────────────────────────
    # MAIN LOOP
    # ─────────────────────────────

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

        total_candidates += len(ck)
        candidates_per_level[k] = len(ck)

        print(f"Level {k}: {len(ck):,} candidates")

        # Early stopping safeguard
        if len(ck) > 100000:
            print("Too many candidates — stopping early")
            break

        if not ck:
            break

        support_ck = count_support(transactions, ck)

        lk = {
            iset: cnt
            for iset, cnt in support_ck.items()
            if cnt >= min_count
        }

        all_frequent.update(lk)

        prev_frequent = lk

        k += 1

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

            "candidates_per_level":
                candidates_per_level,
        }
    }


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":

    DATASET_PATH = "connect.dat"  # ← change to: chess.dat / accidents.dat

    MIN_SUP = 0.95   # ← change threshold (0.0 – 1.0)

    MAX_K = 4

    transactions = load_dataset(DATASET_PATH)

    result = apriori(
        transactions,
        min_sup=MIN_SUP,
        max_k=MAX_K
    )

    m = result["metrics"]

    print(f"\n{'═'*55}")
    print("BASELINE APRIORI RESULTS")
    print(f"{'═'*55}")

    print(f"Dataset              : {DATASET_PATH}")
    print(f"Transactions         : {len(transactions):,}")
    print(f"Minimum Support      : {MIN_SUP:.0%}")
    print(f"Maximum k            : {MAX_K}")

    print(f"\nExecution Time       : {m['execution_time_s']} s")
    print(f"Peak Memory          : {m['peak_memory_mb']} MB")

    print(f"Frequent Itemsets    : {m['n_frequent_itemsets']:,}")
    print(f"Candidates Generated : {m['n_candidates_total']:,}")

    print("\nCandidates Per Level:")

    for level, count in m["candidates_per_level"].items():
        print(f"Level {level}: {count:,}")

    print(f"{'═'*55}\n")