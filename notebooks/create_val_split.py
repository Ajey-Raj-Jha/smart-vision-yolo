import os
import random
import shutil

DATA_DIR = "classification_data"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")

SPLIT_RATIO = 0.15  # 15% validation

os.makedirs(VAL_DIR, exist_ok=True)

for class_name in os.listdir(TRAIN_DIR):
    class_train_path = os.path.join(TRAIN_DIR, class_name)
    class_val_path = os.path.join(VAL_DIR, class_name)

    if not os.path.isdir(class_train_path):
        continue

    os.makedirs(class_val_path, exist_ok=True)

    images = os.listdir(class_train_path)
    random.shuffle(images)

    split_count = int(len(images) * SPLIT_RATIO)
    val_images = images[:split_count]

    for img in val_images:
        src = os.path.join(class_train_path, img)
        dst = os.path.join(class_val_path, img)
        shutil.move(src, dst)

print("Validation split created successfully.")