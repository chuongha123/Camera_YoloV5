import os
import glob

def convert_labels(input_dir):
    # Tìm tất cả các file .txt trong thư mục
    label_files = glob.glob(os.path.join(input_dir, "*.txt"))
    
    for label_file in label_files:
        with open(label_file, 'r') as f:
            lines = f.readlines()
        
        # Thay đổi class index từ 0 thành 80
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if parts[0] == '0':  # Nếu class index là 0
                parts[0] = '80'  # Đổi thành 80
            new_lines.append(' '.join(parts) + '\n')
        
        # Ghi lại file
        with open(label_file, 'w') as f:
            f.writelines(new_lines)
        print(f"Đã sửa: {label_file}")

# Chuyển đổi labels trong các thư mục train, valid và test
print("Đang chuyển đổi labels trong thư mục train...")
convert_labels('custom_train/train/labels')

print("Đang chuyển đổi labels trong thư mục valid...")
convert_labels('custom_train/valid/labels')

print("Đang chuyển đổi labels trong thư mục test...")
convert_labels('custom_train/test/labels')

print("Đã chuyển đổi xong tất cả các file label!") 