import os

# Đường dẫn đến thư mục chứa annotation .txt (YOLO format)
TEST_DIR = "yolov5/train/test/labels"  # đổi lại nếu cần
TRAIN_DIR = "yolov5/train/train/labels"  # đổi lại nếu cần
VALID_DIR = "yolov5/train/valid/labels"  # đổi lại nếu cần

# Class cũ cần đổi và class mới tương ứng
OLD_CLASS_ID = "0"
NEW_CLASS_ID = "80"

def convert_annotation(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        if not line.strip():
            continue
        parts = line.strip().split()
        if parts[0] == OLD_CLASS_ID:
            parts[0] = NEW_CLASS_ID
        new_lines.append(" ".join(parts))

    with open(file_path, "w") as f:
        f.write("\n".join(new_lines) + "\n")

def main():
    for root, dirs, files in os.walk(TEST_DIR):
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)
                convert_annotation(file_path)
                print(f"✔ Đã sửa: {file_path}")

    for root, dirs, files in os.walk(TRAIN_DIR):
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)
                convert_annotation(file_path)
                print(f"✔ Đã sửa: {file_path}")

    for root, dirs, files in os.walk(VALID_DIR):
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)
                convert_annotation(file_path)
                print(f"✔ Đã sửa: {file_path}")

if __name__ == "__main__":
    main()