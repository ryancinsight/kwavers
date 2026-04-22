import os
import json

src_dir = r"d:\kwavers\kwavers\src"
files_by_lines = []

for root, _, files in os.walk(src_dir):
    for f in files:
        if f.endswith(".rs"):
            full_path = os.path.join(root, f)
            try:
                with open(full_path, "r", encoding="utf-8") as file:
                    # Ignore empty lines and comments if we want strict logic, but let's just do line count for now
                    lines = sum(1 for line in file)
                    if lines > 400:
                        files_by_lines.append({"lines": lines, "path": full_path})
            except Exception:
                pass

files_by_lines.sort(key=lambda x: x["lines"], reverse=True)
with open(r"d:\kwavers\kwavers\large_files.json", "w") as out:
    json.dump(files_by_lines[:20], out, indent=2)
