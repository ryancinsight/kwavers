import os

src_dir = r"d:\kwavers\kwavers\src"
files_by_lines = []

for root, _, files in os.walk(src_dir):
    for f in files:
        if f.endswith(".rs"):
            full_path = os.path.join(root, f)
            try:
                with open(full_path, "r", encoding="utf-8") as file:
                    lines = sum(1 for line in file)
                    files_by_lines.append((lines, full_path))
            except Exception:
                pass

files_by_lines.sort(reverse=True)
print("Top largest files:")
for lines, path in files_by_lines[:15]:
    if lines > 400:
        print(f"{lines} lines - {path}")
