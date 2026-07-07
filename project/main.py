from __future__ import annotations

from pprint import pprint

from src.pipeline import run_pipeline


def main() -> None:
    summary = run_pipeline()
    print("Best Model:", summary["best_model"])
    print(f"Final Accuracy (or R²): {summary['final_metric_value']:.4f}")
    print("Top 10 Important Features:")
    pprint(summary["top_10_features"])
    print("Final Business Summary:")
    for item in summary["business_summary"]:
        print("-", item)


if __name__ == "__main__":
    main()
