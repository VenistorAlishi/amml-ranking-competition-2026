from __future__ import annotations

import argparse
from pathlib import Path

from src.pipeline import load_raw, run_cv_and_train


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AMML ranking competition 2026 pipeline")
    p.add_argument("--train-path", default="data/train.csv")
    p.add_argument("--test-path", default="data/test.csv")
    p.add_argument("--models-dir", default="models")
    p.add_argument("--submission-path", default="results/submission.csv")
    p.add_argument("--no-nn", action="store_true", help="Skip pairwise NN fine-tune")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    train, test = load_raw(args.train_path, args.test_path)
    sub = run_cv_and_train(
        train,
        test,
        models_dir=args.models_dir,
        do_nn=not args.no_nn,
    )
    out = Path(args.submission_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out, index=False)
    print(f"Wrote {out} rows={len(sub)}")


if __name__ == "__main__":
    main()
