import os

def count_lines(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            return sum(1 for _ in f)
    except:
        return 0

src_dir = r"d:\kwavers\kwavers\src"
file_counts = []

for root, _, files in os.walk(src_dir):
    for file in files:
        if file.endswith(".rs"):
            filepath = os.path.join(root, file)
            file_counts.append((count_lines(filepath), filepath.replace(src_dir, '')))

file_counts.sort(reverse=True)
with open("top20_results.txt", "w") as f:
    for lines, path in file_counts[:20]:
        f.write(f"{lines:6d} {path}\n")
