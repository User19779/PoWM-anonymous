import json
from pathlib import Path
from natsort import natsorted

def generate_json(img_dir, mask_dir, output_file="file_list.json"):
    # 1.  pathlib  .png 
    img_names = {f.stem for f in Path(img_dir).glob("*.png")}
    mask_names = {f.stem for f in Path(mask_dir).glob("*.png")}

    # 2. 
    common_sorted = natsorted(list(img_names & mask_names))

    # 3.  JSON
    Path(output_file).write_text(json.dumps(common_sorted, indent=4), encoding='utf-8')
    
    print(f" {len(common_sorted)}  {output_file}")

if __name__ == "__main__":
    generate_json(
        "image_mask_reduced/image_reduced", 
        "image_mask_reduced/mask_reduced"
    )