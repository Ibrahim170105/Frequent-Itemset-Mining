# Frequent Itemset Mining — CS-378 Semester Project
### Algorithms: Apriori | Optimized Apriori | FP-Growth

---

## Files Included

| File | Description |
|------|-------------|
| `apriori.py` | Baseline Apriori algorithm |
| `apriori_optimized.py` | Apriori with SON partitioning + tid-list optimization |
| `fpgrowth.py` | FP-Growth algorithm (Mi, 2022) |
| `chess.dat` | Chess dataset — 3,196 transactions |
| `connect.dat` | Connect dataset — 67,557 transactions |
| `accidents.dat` | Accidents dataset — 340,183 transactions |

---

## Requirements

- Python 3.11 or later
- No external libraries needed (uses only standard library)

---

## Setup

Place all `.py` and `.dat` files in the **same folder**, then open a terminal in that folder.

Download the datasets from: **http://fimi.uantwerpen.be/data/**

---

## How to Run

### 1. Baseline Apriori

```bash
python apriori.py
```

Open `apriori.py` and edit the bottom three lines:

```python
DATASET_PATH = "chess.dat"   # change to: connect.dat / accidents.dat
MIN_SUP      = 0.80          # change threshold (0.0 – 1.0)
MIN_CONF     = 0.50          # minimum confidence for rules
```

---

### 2. Optimized Apriori

```bash
python apriori_optimized.py
```

Open `apriori_optimized.py` and edit the bottom four lines:

```python
DATASET_PATH = "chess.dat"   # change to: connect.dat / accidents.dat
MIN_SUP      = 0.80          # change threshold (0.0 – 1.0)
MIN_CONF     = 0.50          # minimum confidence for rules
N_PARTITIONS = 4             # number of partitions (try 2 or 4)
```

---

### 3. FP-Growth

```bash
python fpgrowth.py
```

Open `fpgrowth.py` and edit the bottom three lines:

```python
DATASET_PATH = "chess.dat"   # change to: connect.dat / accidents.dat
MIN_SUP      = 0.80          # change threshold (0.0 – 1.0)
MIN_CONF     = 0.50          # minimum confidence for rules
```

---

## Recommended Thresholds Per Dataset

| Dataset | Safe thresholds for Apriori | FP-Growth works at |
|---------|----------------------------|--------------------|
| `chess.dat` | 0.80, 0.90, 0.95 | 0.80, 0.90, 0.95 |
| `connect.dat` | 0.95 only | 0.90, 0.95 |
| `accidents.dat` | 0.90, 0.95 | 0.80, 0.90, 0.95 |

> **Warning:** Running baseline Apriori on `accidents.dat` below 0.90 or
> `connect.dat` below 0.95 may take very long or not complete.
> FP-Growth handles all thresholds on all datasets.

---

## Output Explained

Every script prints the following metrics after running:

| Metric | Description |
|--------|-------------|
| Execution time | Wall-clock time in seconds (averaged over runs) |
| Peak memory | Maximum RAM used in MB |
| Frequent itemsets | Total number of frequent itemsets found |
| Candidates generated | Candidates evaluated (0 for FP-Growth) |
| Association rules | Rules meeting the min_conf threshold |
| Candidates per level | Apriori only — breakdown by itemset size |

---

## Changing the Minimum Support Threshold

`MIN_SUP` is a decimal between 0.0 and 1.0 representing a percentage:

```python
MIN_SUP = 0.98   # 98% — very strict, few itemsets, fast
MIN_SUP = 0.90   # 90% — moderate
MIN_SUP = 0.80   # 80% — relaxed, many itemsets, slower for Apriori
MIN_SUP = 0.50   # 50% — very relaxed (only use with FP-Growth)
```

Lower threshold = more frequent itemsets = longer runtime for Apriori.
FP-Growth runtime is largely unaffected by threshold changes.

---

## Example: Running All Three on Chess at 90%

Set in each file:
```python
DATASET_PATH = "chess.dat"
MIN_SUP      = 0.90
```

Then run one by one:
```bash
python apriori.py
python apriori_optimized.py
python fpgrowth.py
```

Compare the `Execution time` and `Peak memory` values across outputs
to compute the speedup ratio for your report:

```
Speedup = Baseline time / Optimized time
```
