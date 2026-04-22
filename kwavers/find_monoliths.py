import os

def count_lines(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return sum(1 for line in f)
    except:
        return 0

src_dir = r"d:\kwavers\kwavers\src"
files_with_lines = []

for root, _, files in os.walk(src_dir):
    for filename in files:
        if filename.endswith(".rs"):
            filepath = os.path.join(root, filename)
            lines = count_lines(filepath)
            files_with_lines.append((lines, filepath))

files_with_lines.sort(reverse=True, key=lambda x: x[0])

print("=== LARGEST RUST FILES IN SRC ===")
for lines, path in files_with_lines[:15]:
    rel_path = os.path.relpath(path, src_dir)
    print(f"{lines:>6} {rel_path}")
