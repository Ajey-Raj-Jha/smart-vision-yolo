import os

# COCO → CUSTOM mapping
COCO_TO_CUSTOM = {
    0: 0,   # person
    2: 1,   # car
    5: 2,   # bus
    6: 3,   # train
    1: 4,   # bicycle
    3: 5,   # motorcycle
    4: 6,   # airplane
    16: 7,  # dog
    15: 8,  # cat
    19: 9,  # cow
    17: 10, # horse
    22: 11  # zebra
}

LABEL_DIR = "yolo/labels"

for file in os.listdir(LABEL_DIR):
    if not file.endswith(".txt"):
        continue

    path = os.path.join(LABEL_DIR, file)

    new_lines = []
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue

            coco_id = int(parts[0])
            if coco_id in COCO_TO_CUSTOM:
                parts[0] = str(COCO_TO_CUSTOM[coco_id])
                new_lines.append(" ".join(parts))

    # overwrite label file
    with open(path, "w") as f:
        f.write("\n".join(new_lines))
