### Person 2 Benchmark Results & Parameter Tuning (MinHash / LSH)

Conducted empirical grid search across 9 parameter combinations on the fixed 10k Dev S1 sample (`common/dev_s1.tsv`) against a 100k candidate slice with true ground-truth targets.

#### Benchmark Grid Summary:
| n-gram | Threshold | Recall (%) | Avg Candidates | Max Candidates | Query Time (s) | Peak RAM (MB) |
|:------:|:---------:|:----------:|:--------------:|:--------------:|:--------------:|:-------------:|
| 2      | 0.3       | 99.42%     | 25,264.76      | 44,402         | 119.42s        | 17,147 MB     |
| 2      | 0.4 / 0.5 | 88.60%     | 689.98         | 4,304          | 35.88s         | 419 MB        |
| **3**  | **0.3**   | **98.54%** | **2,903.62**   | **12,610**     | **32.09s**     | **1,672 MB**  |
| 3      | 0.4 / 0.5 | 78.07%     | 12.37          | 287            | 16.57s         | 46 MB         |
| 4      | 0.3       | 97.66%     | 1,910.54       | 10,636         | 25.94s         | 1,144 MB      |
| 4      | 0.4 / 0.5 | 69.88%     | 20.39          | 940            | 16.31s         | 49 MB         |

#### Decision:
- **Default `threshold=0.5` is rejected**: Misses 21.93% of ground-truth matches (78.07% recall).
- **`n=2` is rejected**: 2-grams create an unscalable 17 GB memory explosion and 25k average candidates.
- **Selected configuration**: `n_gram=3`, `threshold=0.3`, `num_perm=64`.
- **Performance**: Delivers **98.54% true match recall** on Dev S1.
- **Full scale execution plan**: Will run `blocking/minhash_lsh.py` with `n=3, threshold=0.3` and an upper bound candidate cap of 150 per S1 to ensure efficient downstream feature generation.