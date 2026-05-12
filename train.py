"""
train.py
--------
Entry point to run the full training pipeline.

Usage:
    # Train all states (slow — use --states for a subset during dev)
    python train.py

    # Train specific states
    python train.py --states California Texas Florida

    # Parallel training (disable if using GPU for LSTM)
    python train.py --jobs 4
"""

import argparse
import sys
from pathlib import Path

# Make sure src is importable when running from project root
sys.path.insert(0, str(Path(__file__).parent))

from src.preprocessing import run_pipeline
from src.model_selector import train_all_states


def parse_args():
    p = argparse.ArgumentParser(description="Run forecasting training pipeline")
    p.add_argument(
        "--states", nargs="*", default=None,
        help="List of states to train (default: all)",
    )
    p.add_argument(
        "--jobs", type=int, default=1,
        help="Parallel jobs (default 1; set >1 to parallelise non-LSTM models)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("  Sales Forecasting — Training Pipeline")
    print("=" * 60)

    # Phase 1 + 2: data prep & feature engineering
    state_data = run_pipeline()

    # Phase 3 + 4: model training & selection
    registry = train_all_states(
        state_data,
        n_jobs=args.jobs,
        states_subset=args.states,
    )

    # Summary
    print("\n" + "=" * 60)
    print("  Training Complete — Summary")
    print("=" * 60)
    model_counts = {}
    for state, info in registry.items():
        m = info["best_model"]
        model_counts[m] = model_counts.get(m, 0) + 1

    print(f"\n  States trained : {len(registry)}")
    print("  Best model distribution:")
    for model, count in sorted(model_counts.items(), key=lambda x: -x[1]):
        print(f"    {model:12s} → {count} states")

    avg_mape = sum(v["best_mape"] for v in registry.values()) / len(registry)
    print(f"\n  Average MAPE   : {avg_mape:.2f}%")
    print("\n  Model registry saved to: model_registry/registry.json")
    print("\nTo start the API:")
    print("  uvicorn src.api:app --reload --host 0.0.0.0 --port 8000")


if __name__ == "__main__":
    main()
