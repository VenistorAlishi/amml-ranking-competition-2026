from pathlib import Path
import re
import json

import fitz

FILES = [
    Path(r"C:\Users\KIRILL\Downloads\Telegram Desktop\pmmo-lec01.pdf"),
    Path(r"C:\Users\KIRILL\Downloads\Telegram Desktop\sem02_ex.pdf"),
    Path(r"C:\Users\KIRILL\Downloads\Telegram Desktop\lec02.pdf"),
    Path(r"C:\Users\KIRILL\Downloads\Telegram Desktop\lec03.pdf"),
    Path(r"C:\Users\KIRILL\Downloads\Telegram Desktop\sem03.pdf"),
    Path(r"C:\Users\KIRILL\Downloads\Telegram Desktop\lec04 (2).pdf"),
    Path(r"C:\Users\KIRILL\Downloads\Telegram Desktop\lec05 (2).pdf"),
    Path(r"C:\Users\KIRILL\Downloads\Telegram Desktop\lec06.pdf"),
    Path(r"C:\Users\KIRILL\Downloads\Telegram Desktop\sem06_hw.pdf"),
    Path(r"C:\Users\KIRILL\Downloads\Telegram Desktop\sem07.pdf"),
    Path(r"C:\Users\KIRILL\Downloads\Telegram Desktop\lec07_2.pdf"),
    Path(r"C:\Users\KIRILL\Downloads\Telegram Desktop\lec08.pdf"),
    Path(r"C:\Users\KIRILL\Downloads\Telegram Desktop\sem08.pdf"),
    Path(r"C:\Users\KIRILL\Downloads\Telegram Desktop\lec09.pdf"),
    Path(r"C:\Users\KIRILL\Downloads\Telegram Desktop\sem09.pdf"),
    Path(r"C:\Users\KIRILL\Downloads\Telegram Desktop\lec10.pdf"),
    Path(r"C:\Users\KIRILL\Downloads\Telegram Desktop\sem09 (2).pdf"),
]

NOTEBOOKS = [
    Path(r"C:\Users\KIRILL\Downloads\Telegram Desktop\sem04.ipynb"),
    Path(r"C:\Users\KIRILL\Downloads\Telegram Desktop\sem05.ipynb"),
    Path(r"C:\Users\KIRILL\Downloads\Telegram Desktop\sem09.ipynb"),
]

OUT = Path(r"C:\Users\KIRILL\Projects\amml-ranking-competition-2026\course_extract")
OUT.mkdir(exist_ok=True)

KEYWORDS = re.compile(
    r"rank|ранж|ndcg|bm25|tf-?idf|cross.?encod|bi-?encod|retriev|поиск|релевант|"
    r"catboost|lambdamart|yeti|learning.to.rank|\bltr\b|passage|эмбед|embed|"
    r"rerank|dense|sparse|colbert|\be5\b|sbert|миниlm|pointwise|pairwise|listwise|"
    r"\bdcg\b|\bmrr\b|инвертир|индекс|дублет|триплет|contrastive|инфо.?nce|"
    r"sentence.?transform|huggingface|bert|roberta|rubert|minilm|xgboost|lightgbm|"
    r"group.?kfold|query.?id|сабмит|kaggle|nDCG|MAP|precision@|recall@",
    re.I,
)


def extract_pdf(path: Path) -> str:
    doc = fitz.open(path)
    text = "\n".join(page.get_text() for page in doc)
    doc.close()
    return text


def extract_nb(path: Path) -> str:
    nb = json.loads(path.read_text(encoding="utf-8"))
    parts = []
    for cell in nb.get("cells", []):
        src = cell.get("source", [])
        if isinstance(src, list):
            src = "".join(src)
        parts.append(f"### {cell.get('cell_type')}\n{src}")
    return "\n\n".join(parts)


def summarize(name: str, text: str) -> str:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    head = " | ".join(lines[:15])[:500]
    hits = []
    for m in KEYWORDS.finditer(text):
        start = max(0, m.start() - 100)
        end = min(len(text), m.end() + 160)
        frag = re.sub(r"\s+", " ", text[start:end]).strip()
        hits.append(frag)
        if len(hits) >= 12:
            break
    out = [f"# {name} ({len(text)} chars)", f"HEAD: {head}", "HITS:"]
    out.extend(f"- {h}" for h in hits)
    return "\n".join(out) + "\n\n"


def main() -> None:
    overview = []
    for f in FILES:
        print("PDF", f.name, flush=True)
        text = extract_pdf(f)
        safe = re.sub(r"[^\w.\-]+", "_", f.stem)
        (OUT / f"{safe}.txt").write_text(text, encoding="utf-8", errors="replace")
        overview.append(summarize(f.name, text))
        print("  chars", len(text), flush=True)

    for f in NOTEBOOKS:
        print("NB", f.name, flush=True)
        text = extract_nb(f)
        safe = re.sub(r"[^\w.\-]+", "_", f.stem)
        (OUT / f"{safe}.nb.txt").write_text(text, encoding="utf-8", errors="replace")
        overview.append(summarize(f.name, text))
        print("  chars", len(text), flush=True)

    (OUT / "_overview.txt").write_text("".join(overview), encoding="utf-8")
    print("DONE", OUT)


if __name__ == "__main__":
    main()
