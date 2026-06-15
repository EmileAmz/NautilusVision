from pathlib import Path

# ====== CHANGE THIS ======
LABEL_DIR = Path(r"C:\Users\eaime\Downloads\Labels_OBB\Labels_OBB")
# =========================

ID_MAP = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,  # remove this class
    5: 5,
    6: 7,
    7: 6
}

def remap_file(file_path):
    new_lines = []

    with open(file_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue

        old_id = int(parts[0])

        if old_id not in ID_MAP:
            print(f"⚠️ Unknown class {old_id} in {file_path}")
            continue

        new_id = ID_MAP[old_id]

        # Skip unwanted class (compass_hamme)
        if new_id is None:
            continue

        parts[0] = str(new_id)
        new_lines.append(" ".join(parts))

    # Overwrite file
    with open(file_path, "w") as f:
        f.write("\n".join(new_lines) + "\n")


def main():
    files = list(LABEL_DIR.glob("*.txt"))
    print(f"Found {len(files)} label files")

    for f in files:
        remap_file(f)

    print("✅ Done remapping labels")


if __name__ == "__main__":
    main()