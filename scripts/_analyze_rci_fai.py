"""Analyze correlation between RCI and FAI from stat_test_results.json."""
import json
import numpy as np
from scipy.stats import pearsonr
import os

def analyze_correlation(results_path: str = "evaluation_results/stat_test_results.json"):
    if not os.path.exists(results_path):
        print(f"Error: Results file not found at {results_path}")
        return

    with open(results_path, 'r') as f:
        data = json.load(f)

    print(f"Analyzing results from: {results_path}")
    print(f"Config: {json.dumps(data['config'], indent=2)}")

    all_fai = []
    all_rci_cat = []
    all_rci_nomove = []

    for alg, res in data['results'].items():
        if 'fai_mean' in res and 'rci' in res and 'rci_nomove' in res:
            print(f"\n--- Algorithm: {alg.upper()} ---")
            print(f"  FAI (mean across seeds): {res['fai_mean']}")
            print(f"  RCI_cat (mean across seeds): {res['rci']}")
            print(f"  RCI_nomove (mean across seeds): {res['rci_nomove']}")

            # For correlation, use raw values per seed
            fai_values = res['fai_mean']
            rci_cat_values = res['rci']
            rci_nomove_values = res['rci_nomove']

            # Ensure lists are of the same length
            min_len = min(len(fai_values), len(rci_cat_values), len(rci_nomove_values))
            if min_len == 0:
                print(f"  Skipping {alg}: No data for correlation (min_len=0)")
                continue
            fai_values = fai_values[:min_len]
            rci_cat_values = rci_cat_values[:min_len]
            rci_nomove_values = rci_nomove_values[:min_len]

            if min_len < 2:
                print(f"  Skipping {alg}: Need at least 2 seeds for correlation (min_len={min_len})")
                continue

            corr_fai_rci_cat, _ = pearsonr(fai_values, rci_cat_values)
            corr_fai_rci_nomove, _ = pearsonr(fai_values, rci_nomove_values)

            print(f"  Pearson correlation FAI vs RCI_cat:     {corr_fai_rci_cat:.4f}")
            print(f"  Pearson correlation FAI vs RCI_nomove:  {corr_fai_rci_nomove:.4f}")

            all_fai.extend(fai_values)
            all_rci_cat.extend(rci_cat_values)
            all_rci_nomove.extend(rci_nomove_values)

    if len(all_fai) > 1:
        print(f"\n--- Overall Correlation (all algorithms combined) ---")
        corr_overall_fai_rci_cat, _ = pearsonr(all_fai, all_rci_cat)
        corr_overall_fai_rci_nomove, _ = pearsonr(all_fai, all_rci_nomove)
        print(f"  Overall Pearson correlation FAI vs RCI_cat:     {corr_overall_fai_rci_cat:.4f}")
        print(f"  Overall Pearson correlation FAI vs RCI_nomove:  {corr_overall_fai_rci_nomove:.4f}")
    else:
        print("\nNot enough data for overall correlation analysis (need >=2 combined seeds).")

if __name__ == "__main__":
    # Temporary. Need user to run stat_test first.
    # For testing, create a dummy results file:
    # dummy_data = {
    #     "config": {"seeds": 3, "eval_episodes": 10},
    #     "results": {
    #         "hmarl": {
    #             "fai_mean": [0.7, 0.72, 0.71],
    #             "rci": [0.4, 0.45, 0.42],
    #             "rci_nomove": [0.1, 0.12, 0.11]
    #         },
    #         "random": {
    #             "fai_mean": [0.5, 0.51, 0.5],
    #             "rci": [0.2, 0.21, 0.2],
    #             "rci_nomove": [0.05, 0.04, 0.05]
    #         }
    #     }
    # }
    # os.makedirs("evaluation_results", exist_ok=True)
    # with open("evaluation_results/stat_test_results.json", 'w') as f:
    #     json.dump(dummy_data, f, indent=2)

    analyze_correlation()
