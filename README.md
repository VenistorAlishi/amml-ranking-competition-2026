# AMML Ranking Competition 2026

Russian query–passage learning-to-rank. Metric: **NDCG@10** (binary relevance).

Competition: [amml-ranking-competition-2026](https://www.kaggle.com/competitions/amml-ranking-competition-2026)

## System

- OS: Windows 10
- Python: 3.12
- Device: CPU (`torch 2.15.0+cpu`)

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Put `train.csv`, `test.csv`, `sample_submission.csv` into `data/`.

## Reproduce final submission

```bash
python scripts/train_ce_fast_submit.py
```

Or:

```bash
jupyter nbconvert --to notebook --execute solution.ipynb --output solution_executed.ipynb
```

Outputs under `results/`:

- `submission_cb_ce.csv` — CatBoost + fine-tuned CE blend
- `submission_ce_only.csv` — fine-tuned CE only

## Pipeline

1. Clean text / coerce metadata
2. Lexical + structural features (BM25, TF-IDF, overlap, passage stats)
3. Train CatBoost `YetiRank` (`NDCG:top=10`), GroupKFold by `query_id`
4. **Fine-tune** `cointegrated/rubert-tiny2` cross-encoder head on competition pairs only (CPU, `max_len=128`, 1 epoch)
5. Select blend weight on a **query holdout** (no train-set leakage)
6. Submit blended scores

## Experiments / failure modes

| Approach | Local / holdout | Public |
| --- | --- | --- |
| CatBoost + non-OOF NN blend | optimistic ~0.84 | **0.751663** (leakage) |
| CatBoost YetiRank only | CV ~0.806 | **0.779692** |
| rubert-tiny2 CE fine-tune + honest holdout blend | (running / see logs) | TBD |

Baselines: `0.826590` / `0.899634` / `0.932861`.

No external labeled datasets are used for fine-tuning.

## Layout

- `solution.ipynb` — entry notebook
- `requirements.txt` — pinned deps
- `README.md` — this file
- `scripts/train_ce_fast_submit.py` — CE fine-tune + blend + submit
- `src/` — features, metrics, CatBoost helpers
- `kaggle_kernel/` — optional GPU notebook
