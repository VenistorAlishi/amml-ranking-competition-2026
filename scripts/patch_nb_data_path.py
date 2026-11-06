import json
from pathlib import Path

nb_path = Path(r"C:\Users\KIRILL\Projects\amml-ranking-competition-2026\solution.ipynb")
nb = json.loads(nb_path.read_text(encoding="utf-8"))

old = """    DATA = Path('/kaggle/input/amml-ranking-competition-2026')
    OUT = Path('/kaggle/working')
    MODELS = OUT / 'models'
"""

new = """    OUT = Path('/kaggle/working')
    MODELS = OUT / 'models'
    # Competition data path differs across Kaggle images / attach modes
    candidates = [
        Path('/kaggle/input/amml-ranking-competition-2026'),
        Path('/kaggle/input/competitions/amml-ranking-competition-2026'),
    ]
    DATA = next((p for p in candidates if (p / 'train.csv').exists()), None)
    if DATA is None:
        hits = list(Path('/kaggle/input').rglob('train.csv'))
        DATA = hits[0].parent if hits else candidates[0]
"""

patched = False
for cell in nb["cells"]:
    if cell.get("id") != "setup":
        continue
    src = "".join(cell["source"])
    if old not in src:
        raise SystemExit("OLD BLOCK NOT FOUND:\n" + src[:1000])
    cell["source"] = [src.replace(old, new)]
    patched = True

if not patched:
    raise SystemExit("setup cell missing")

nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
kernel_nb = Path(r"C:\Users\KIRILL\Projects\amml-ranking-competition-2026\kaggle_kernel\amml-ranking-2026-gpu.ipynb")
kernel_nb.write_text(nb_path.read_text(encoding="utf-8"), encoding="utf-8")
print("patched ok")
