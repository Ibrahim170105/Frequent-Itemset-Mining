import math
import time
import tracemalloc
from itertools import combinations
from collections import defaultdict


# ─────────────────────────────────────────────
# 1. DATASET LOADER
# ─────────────────────────────────────────────

def load_dataset(filepath: str) -> list[list]:
    
    transactions = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                transactions.append(list(map(int, line.split())))
    return transactions


# ─────────────────────────────────────────────
# 2. FP-TREE NODE
# ─────────────────────────────────────────────

class FPNode:
    
    __slots__ = ("item", "count", "parent", "children", "node_link")

    def __init__(self, item, count, parent):
        self.item      = item
        self.count     = count
        self.parent    = parent
        self.children  = {}
        self.node_link = None

    def increment(self, count):
        self.count += count


# ─────────────────────────────────────────────
# 3. FP-TREE
# ─────────────────────────────────────────────

class FPTree:
    

    def __init__(self):
        self.root         = FPNode(None, 0, None)
        self.header_table = {}

    def insert_transaction(self, transaction: list, count: int = 1):
       
        current = self.root
        for item in transaction:
            if item in current.children:
                # Node exists — increment it
                current.children[item].increment(count)
                # *** CRITICAL FIX ***
                # Also update the header table's support counter.
                # Without this, items that share prefixes (i.e. most
                # items in a dense dataset) appear to have very low
                # support in the header table, causing them to be
                # incorrectly pruned during mining.
                self.header_table[item][0] += count
            else:
                # New node — create and register in header table
                new_node = FPNode(item, count, current)
                current.children[item] = new_node
                if item not in self.header_table:
                    self.header_table[item] = [count, new_node]
                else:
                    self.header_table[item][0] += count
                    # Append to end of linked list for this item
                    tail = self.header_table[item][1]
                    while tail.node_link:
                        tail = tail.node_link
                    tail.node_link = new_node
            current = current.children[item]

    def is_single_path(self) -> bool:
       
        node = self.root
        while node.children:
            if len(node.children) > 1:
                return False
            node = next(iter(node.children.values()))
        return True


# ─────────────────────────────────────────────
# 4. BUILD FP-TREE FROM TRANSACTIONS  (2 scans)
# ─────────────────────────────────────────────

def build_fptree(transactions: list[list],
                 min_count: float,
                 item_order: dict = None) -> tuple[FPTree, dict]:
    
    if item_order is None:
        # Scan 1 — count 1-itemset support
        item_support = defaultdict(int)
        for transaction in transactions:
            for item in transaction:
                item_support[item] += 1
        item_order = {item: sup
                      for item, sup in item_support.items()
                      if sup >= min_count}

    # Scan 2 — build FP-tree
    tree = FPTree()
    for transaction in transactions:
        filtered = sorted(
            [item for item in transaction if item in item_order],
            key=lambda x: item_order[x],
            reverse=True       # higher support → closer to root → more sharing
        )
        if filtered:
            tree.insert_transaction(filtered)

    return tree, item_order


# ─────────────────────────────────────────────
# 5. CONDITIONAL PATTERN BASE EXTRACTION
# ─────────────────────────────────────────────

def extract_conditional_pattern_base(item: int,
                                     tree: FPTree) -> list[tuple]:
    
    patterns = []
    node = tree.header_table[item][1]
    while node:
        count  = node.count
        prefix = []
        parent = node.parent
        while parent.item is not None:
            prefix.append(parent.item)
            parent = parent.parent
        if prefix:
            patterns.append((prefix, count))
        node = node.node_link
    return patterns


# ─────────────────────────────────────────────
# 6. SINGLE-PATH OPTIMISATION
# ─────────────────────────────────────────────

def enumerate_single_path(tree: FPTree, prefix: list,
                           min_count: float, frequent_itemsets: dict):
    
    path_items = []
    node = tree.root
    while node.children:
        node = next(iter(node.children.values()))
        path_items.append((node.item, node.count))

    for size in range(1, len(path_items) + 1):
        for combo in combinations(path_items, size):
            count = min(c[1] for c in combo)
            if count >= min_count:
                itemset = frozenset(prefix + [c[0] for c in combo])
                frequent_itemsets[itemset] = count


# ─────────────────────────────────────────────
# 7. RECURSIVE FP-TREE MINING
# ─────────────────────────────────────────────

def mine_fptree(tree: FPTree, prefix: list,
                min_count: float, frequent_itemsets: dict):
    
    if tree.is_single_path():
        enumerate_single_path(tree, prefix, min_count, frequent_itemsets)
        return

    # Process items rarest-first (ascending support)
    items_sorted = sorted(
        tree.header_table.keys(),
        key=lambda x: tree.header_table[x][0]
    )

    for item in items_sorted:
        item_support = tree.header_table[item][0]
        if item_support < min_count:
            continue

        new_prefix  = prefix + [item]
        frequent_itemsets[frozenset(new_prefix)] = item_support

        # --- Build conditional FP-tree ---
        cond_patterns = extract_conditional_pattern_base(item, tree)
        if not cond_patterns:
            continue

        # Count item support within the conditional pattern base
        cond_support = defaultdict(int)
        for pattern, count in cond_patterns:
            for p_item in pattern:
                cond_support[p_item] += count

        # Keep only items frequent within the CPB
        cond_item_order = {i: s for i, s in cond_support.items()
                           if s >= min_count}
        if not cond_item_order:
            continue

        # Build conditional FP-tree from filtered, sorted CPB
        cond_tree = FPTree()
        for pattern, count in cond_patterns:
            filtered = sorted(
                [p for p in pattern if p in cond_item_order],
                key=lambda x: cond_item_order[x],
                reverse=True
            )
            if filtered:
                cond_tree.insert_transaction(filtered, count)

        if cond_tree.header_table:
            mine_fptree(cond_tree, new_prefix, min_count, frequent_itemsets)


# ─────────────────────────────────────────────
# 8. ASSOCIATION RULE GENERATION
# ─────────────────────────────────────────────

def generate_association_rules(frequent_itemsets: dict,
                                n_transactions: int,
                                min_confidence: float) -> list[dict]:
    """Generate rules A → B where confidence ≥ min_confidence."""
    rules       = []
    support_map = {iset: count / n_transactions
                   for iset, count in frequent_itemsets.items()}

    for itemset, count in frequent_itemsets.items():
        if len(itemset) < 2:
            continue
        for size in range(1, len(itemset)):
            for antecedent in combinations(itemset, size):
                antecedent = frozenset(antecedent)
                consequent = itemset - antecedent
                if antecedent in support_map:
                    confidence = (count / n_transactions) / support_map[antecedent]
                    if confidence >= min_confidence:
                        rules.append({
                            "antecedent" : set(antecedent),
                            "consequent" : set(consequent),
                            "support"    : round(count / n_transactions, 4),
                            "confidence" : round(confidence, 4),
                        })
    return rules


# ─────────────────────────────────────────────
# 9. MAIN FP-GROWTH FUNCTION
# ─────────────────────────────────────────────

def fpgrowth(transactions: list[list],
             min_sup: float,
             min_conf: float = 0.5) -> dict:
   
    n         = len(transactions)
    min_count = math.ceil(min_sup * n)

    tracemalloc.start()
    start_time = time.perf_counter()

    # Scan 1 + Scan 2: build FP-tree
    tree, item_order = build_fptree(transactions, min_count)

    # Collect frequent 1-itemsets
    frequent_itemsets = {}
    for item, sup in item_order.items():
        frequent_itemsets[frozenset([item])] = sup

    # Mine FP-tree — zero candidate generation
    mine_fptree(tree, [], min_count, frequent_itemsets)

    elapsed     = time.perf_counter() - start_time
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    rules = generate_association_rules(frequent_itemsets, n, min_conf)

    return {
        "frequent_itemsets" : frequent_itemsets,
        "rules"             : rules,
        "metrics": {
            "execution_time_s"    : round(elapsed, 4),
            "peak_memory_mb"      : round(peak_mem / (1024 ** 2), 4),
            "n_frequent_itemsets" : len(frequent_itemsets),
            "n_candidates_total"  : 0,     # FP-Growth generates ZERO candidates
            "n_rules"             : len(rules),
            "db_scans"            : 2,     # always exactly 2
        },
    }


# ─────────────────────────────────────────────
# 10. ENTRY POINT — only edit these 3 lines
# ─────────────────────────────────────────────

if __name__ == "__main__":

    DATASET_PATH = "connect.dat"   # ← change to: connect.dat / accidents.dat
    MIN_SUP      = 0.95      # ← change threshold (0.0 – 1.0)
    MIN_CONF     = 0.50          # ← minimum confidence for rules

    transactions = load_dataset(DATASET_PATH)
    result       = fpgrowth(transactions, min_sup=MIN_SUP, min_conf=MIN_CONF)
    m            = result["metrics"]

    print(f"\n{'═'*54}")
    print(f"  FP-GROWTH RESULTS")
    print(f"  Reference: Mi, X. (2022). Comput Intell Neurosci.")
    print(f"  DOI: 10.1155/2022/7022168")
    print(f"{'─'*54}")
    print(f"  Dataset              : {DATASET_PATH}")
    print(f"  Transactions         : {len(transactions):,}")
    print(f"  min_sup: {MIN_SUP:.0%}   min_conf: {MIN_CONF:.0%}")
    print(f"{'─'*54}")
    print(f"  Execution time       : {m['execution_time_s']} s")
    print(f"  Peak memory          : {m['peak_memory_mb']} MB")
    print(f"  Frequent itemsets    : {m['n_frequent_itemsets']:,}")
    print(f"  Candidates generated : {m['n_candidates_total']}  ← always 0")
    print(f"  DB scans             : {m['db_scans']}  ← always 2")
    print(f"  Association rules    : {m['n_rules']:,}")
    print(f"{'═'*54}\n")