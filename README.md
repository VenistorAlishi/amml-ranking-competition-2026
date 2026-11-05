# AMML Ranking Competition 2026

Learning-to-rank for Russian query–passage pairs. Metric: **NDCG@10** (binary relevance).

Competition: [amml-ranking-competition-2026](https://www.kaggle.com/competitions/amml-ranking-competition-2026)

## Recommended compute: Kaggle GPU notebooks

Kaggle gives ~**30 GPU hours/week**. Heavy steps (CrossEncoder fine-tune, full CatBoost CV) should run there, not on a local CPU.

1. Open / fork the competition notebook (or push `solution.ipynb` as a kernel attached to this competition).
2. Enable **GPU** + **Internet**.
3. Run all cells → submit `/kaggle/working/submission.csv`.

Local Windows CPU is fine for feature debugging and CatBoost-only baselines.

## System notes

- Local: Windows 10 / Python 3.12 / CPU
- Kaggle: Linux notebook / GPU (CUDA) when available

## Setup (local)

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Place competition files in `data/`: `train.csv`, `test.csv`, `sample_submission.csv`.

## Reproduce final submission

```bash
# preferred (competition deliverable)
jupyter nbconvert --to notebook --execute solution.ipynb --output solution_executed.ipynb

# or CLI
python main.py --train-path data/train.csv --test-path data/test.csv --models-dir models --submission-path results/submission.csv
```

Output columns: `id,prediction`.

## Final pipeline

1. Text cleanup + numeric metadata coercion
2. Lexical features: overlap / Jaccard / numbers / BM25 (global+local) / TF-IDF word+char
3. Structural: passage index, article size, `ingest_source`, within-query ranks/z-scores
4. **Train** CatBoost `YetiRank` (GroupKFold by `query_id`)
5. **Fine-tune** `cointegrated/rubert-tiny2` cross-encoder on competition pairs only (GPU on Kaggle)
6. Within-query min-max blend; CE weight kept small because train CE scores are in-sample

No external labeled datasets are used for fine-tuning.

## Experiments

| Idea | Result |
| --- | --- |
| CatBoost YetiRank + handcrafted features | solid base; CV ~0.80 NDCG@10 |
| Pairwise MLP on TF-IDF blended 50/50 | **hurt LB** (public 0.751) — do not over-weight non-OOF channels |
| Frozen HF CE without fine-tune | disallowed / weak; we fine-tune on train |
| CE fine-tune on Kaggle GPU | preferred path to beat baselines 0.827 / 0.900 / 0.933 |

## Repository layout

- `solution.ipynb` — full reproducible pipeline (Kaggle GPU aware)
- `requirements.txt`
- `README.md` — experiments
- `src/` — preprocessing, features, CatBoost, CE fine-tune
- `main.py` — CLI entrypoint
