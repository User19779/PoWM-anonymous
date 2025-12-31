import json
from pathlib import Path
from PIL import Image

# ---  ---
IMAGE_FOLDER = Path('./image_mask_reduced/image_reduced')
MASK_FOLDER = Path('./image_mask_reduced/mask_reduced')
OUTPUT_DIR = Path('./dataset_new')
SIZE_DIFFERENCE_LIMIT = 15
IMAGE_EXTENSION = '.png'

#  (Train:Val:Test = 6:2:2)
RATIOS = [0.6, 0.2, 0.2]

def split_and_save():
    # 1.  JSON 
    # 
    img_names = {f.stem for f in IMAGE_FOLDER.glob(f"*{IMAGE_EXTENSION}")}
    mask_names = {f.stem for f in MASK_FOLDER.glob(f"*{IMAGE_EXTENSION}")}
    base_names = sorted(list(img_names & mask_names), key=lambda x: [int(t) if t.isdigit() else t for t in x.split('_')])

    valid_names = []
    print(f"--- : {SIZE_DIFFERENCE_LIMIT}px ---")

    # 2. 
    for name in base_names:
        img_path = IMAGE_FOLDER / f"{name}{IMAGE_EXTENSION}"
        mask_path = MASK_FOLDER / f"{name}{IMAGE_EXTENSION}"

        with Image.open(img_path) as img, Image.open(mask_path) as mask:
            w_diff = abs(img.width - mask.width)
            h_diff = abs(img.height - mask.height)

            if w_diff <= SIZE_DIFFERENCE_LIMIT and h_diff <= SIZE_DIFFERENCE_LIMIT:
                valid_names.append(name)
            else:
                print(f": {name} (Diff: W:{w_diff}, H:{h_diff})")

    # 3.  ()
    total = len(valid_names)
    idx_val = int(total * RATIOS[0])
    idx_test = int(total * (RATIOS[0] + RATIOS[1]))

    splits = {
        "train.json": valid_names[:idx_val],
        "val.json": valid_names[idx_val:idx_test],
        "test.json": valid_names[idx_test:]
    }

    # 4. 
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for filename, data in splits.items():
        save_path = OUTPUT_DIR / filename
        save_path.write_text(json.dumps(data, indent=4), encoding='utf-8')
        print(f" {filename}: {len(data)} ")

if __name__ == '__main__':
    split_and_save()