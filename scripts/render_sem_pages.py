from pathlib import Path
import fitz

files = [
    Path(r"C:\Users\KIRILL\Downloads\Telegram Desktop\sem06_hw.pdf"),
    Path(r"C:\Users\KIRILL\Downloads\Telegram Desktop\sem07.pdf"),
    Path(r"C:\Users\KIRILL\Downloads\Telegram Desktop\sem08.pdf"),
    Path(r"C:\Users\KIRILL\Downloads\Telegram Desktop\sem09.pdf"),
    Path(r"C:\Users\KIRILL\Downloads\Telegram Desktop\sem09 (2).pdf"),
]
out = Path(r"C:\Users\KIRILL\Projects\amml-ranking-competition-2026\course_extract\pages")
out.mkdir(parents=True, exist_ok=True)
for f in files:
    doc = fitz.open(f)
    print(f.name, "pages", doc.page_count)
    for i in range(min(4, doc.page_count)):
        pix = doc[i].get_pixmap(matrix=fitz.Matrix(2, 2))
        name = f"{f.stem.replace(' ', '_')}_p{i+1}.png"
        pix.save(str(out / name))
        print(" saved", name)
    doc.close()
