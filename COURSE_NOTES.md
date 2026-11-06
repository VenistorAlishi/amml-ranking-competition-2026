# Идеи из курса ПММО 2026 → соревнование

Курс: Чухров Н.С., РАНХиГС. Соревнование почти дословно из **лекций 9–10 (ранжирование)**.

## Что совпадает один-в-один

| Курс | Соревнование |
| --- | --- |
| Группы `(qi, Di, yi)`, признаки `xi,j = φ(qi, di,j)` | `query_id` + кандидаты, `relevance` ∈ {0,1} |
| Метрика **nDCG@k** | **NDCG@10** |
| Валидация **по запросам** (lec06) | GroupKFold по `query_id` |
| φ: BM25, эмбеддинги, … | наши lexical/BM25/TF-IDF фичи |
| Listwise / YetiRank в CatBoost (lec10) | `CatBoostRanker(loss=YetiRank)` |
| Bi-encoder + Cross-encoder (lec10) | CE fine-tune + (план) bi-encoder cosine |
| Нельзя путать pointwise loss с порядком | наш провал public 0.75 от плохого blend |

## Что взять в пайплайн (приоритет)

1. **BM25 с тюном `k1`, `b` на GroupKFold** (lec09: «параметры подбирают на валидации»).
2. **Pointwise фичи из курса:** число общих слов, coverage, TF/IDF-вариации, cosine эмбеддингов.
3. **Listwise loss:** YetiRank / YetiRankPairwise (уже есть); сравнить с `QueryRMSE` / pairwise.
4. **Bi-encoder → cosine как фича** (DSSM / BGE-M3 из lec10), затем CatBoost поверх.
5. **Cross-encoder re-rank top-k** (BERT CE, lec10) — fine-tune только на train соревнования.
6. **DeText-стиль:** кодировать поля `title` и `text` отдельно + lexical фичи в бустинг.
7. **Ordered target stats** для редких категорий (`ingest_source`, возможно `article_id`) — CatBoost CTR из lec05/sem09; не leak’ать таргет.
8. **Не раздувать blend** каналами без честного OOF (урок с MLP 50/50 → 0.75).

## Что из курса пока вторично

- PageRank / кликовые unbiased LambdaMART — нет графа ссылок и кликов.
- LLM listwise zero-shot (Qwen3-Reranker) — дорого; только если хватит Kaggle GPU quota.
- ColBERT late interaction — тяжело на CPU; на Kaggle GPU можно как эксперимент.

## Семинары

- `sem05` / `sem08` / `sem09`: CatBoost, ordered boosting, ordered CTR — про категориальные и leakage.
- Сканы `sem06_hw`–`sem09`: в основном упражнения по деревьям/CatBoost, не отдельный LTR homework текст.
- Ранжирование теоретически закрыто **lec09 + lec10** (как раз перед дедлайном соревнования 5–6 ноября).

## Текущий LB-контекст

- Baseline 1/2/3: 0.827 / 0.900 / 0.933
- Наш CB-only: ~0.780 (лучше MLP-blend 0.75, всё ещё ниже Baseline 1)
- Нужен семантический сигнал (bi/cross-encoder) + аккуратный listwise бустинг на Kaggle GPU.
