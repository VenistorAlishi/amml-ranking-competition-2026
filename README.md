# AMML Ranking Competition 2026

Learning-to-rank for Russian query–passage pairs. Metric: **NDCG@10** (binary relevance).

Competition: [amml-ranking-competition-2026](https://www.kaggle.com/competitions/amml-ranking-competition-2026)

## System

- OS: Windows 10
- Python: 3.12
- Device: CPU (no CUDA)

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Place competition files in `data/`:

- `train.csv`
- `test.csv`
- `sample_submission.csv`

## Reproduce final submission

Preferred (notebook required by competition rules):

```bash
jupyter nbconvert --to notebook --execute solution.ipynb --output solution_executed.ipynb
```

Or CLI:

```bash
python main.py --train-path data/train.csv --test-path data/test.csv --models-dir models --submission-path results/submission.csv
```

Output: `results/submission.csv` with columns `id,prediction`.

## Final pipeline

1. Clean text / coerce numeric metadata
2. Lexical features: token overlap, Jaccard, number match, BM25 (global + per-query), TF-IDF word/char cosine
3. Structural features: passage index, article size, `ingest_source`
4. Within-query ranks / z-scores
5. **Train** CatBoost `YetiRank` with GroupKFold by `query_id`
6. **Fine-tune** a pairwise MLP on TF-IDF query/doc vectors (competition data only)
7. Blend CatBoost + NN scores with within-query min-max; blend weight chosen on OOF NDCG@10

No external labeled datasets are used for fine-tuning.

## Experiments (summary)

| Idea | Result |
| --- | --- |
| TF-IDF cosine alone | weak absolute corr; useful as a feature |
| BM25 global + local | strong within-query signal |
| CatBoost YetiRank + handcrafted features | main model; GroupKFold OOF used for selection |
| Pairwise MLP fine-tune on TF-IDF | added; blend weight selected on OOF |
| Frozen HF cross-encoders | skipped locally (torch/transformers version conflict on this machine) |

Baselines to beat (public LB orientation): 0.827 / 0.900 / 0.933.

## Repository layout

- `solution.ipynb` — full reproducible pipeline
- `requirements.txt` — pinned dependencies
- `README.md` — this file (experiments + run notes)
- `src/` — preprocessing, features, metrics, CatBoost train, NN fine-tune
- `main.py` — CLI entrypoint
