# Data notes (EDA + cleaning)

Competition text explicitly asks to inspect data: missing values, uneven formats, processing errors.

## Findings (train / test)

| Issue | Train | Test | Action |
| --- | ---: | ---: | --- |
| Missing / empty `title` | ~2896 | ~1423 | Flag `title_missing`; fill with first sentence of `text` |
| Dirty `article_char_count` (`84 997`, `3\xa0533`, `2 919,0`) | 208 non-numeric | 119 | `parse_messy_number` |
| HTML tags in `text` | 309 | 159 | Strip in `clean_text`; flag `text_had_html` |
| NBSP / thin spaces | very common | very common | Normalize in `clean_text` |
| `passage_index >= article_passage_count` | 458 | 221 | Flag `passage_oob` |
| Dup `(query_id, article_id, passage_index)` | 67 | 51 | Kept (ids unique); model sees both rows |
| Conflicting labels on near-dup query+text | 4 keys | — | Rare; left as-is |
| Query/test leakage by `query_id` | 0 overlap | — | OK for GroupKFold |

## Preprocess changes (`src/preprocessing.py`)

- Stronger unicode / HTML / newline cleanup
- Wiki-style number parsing for `article_char_count`
- Synthetic titles when missing
- Quality flags + `char_vs_stated` ratio for CatBoost

## Rules reminder

- Cleaning / EDA on competition data: **allowed and encouraged**
- Fine-tuning on external open labeled datasets: **forbidden**
- Must fine-tune somewhere in the pipeline (retrieve / rerank / …)
