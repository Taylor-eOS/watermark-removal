import os
import numpy as np
from PIL import Image

INPUT_DIR = "input_files"
OUTPUT_DIR = "output_files"
DIFF_DIR = "diff"
AMPLIFY = 40
VALID_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")

def load_rgb(path):
    image = Image.open(path)
    if image.mode == "RGBA":
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.getchannel("A"))
        return background
    return image.convert("RGB")

def find_output(name):
    candidate = os.path.join(OUTPUT_DIR, name + ".jpg")
    if os.path.isfile(candidate):
        return candidate
    return None

def compare(original_path, output_path):
    original = np.asarray(load_rgb(original_path)).astype(np.int16)
    result = np.asarray(load_rgb(output_path)).astype(np.int16)
    if original.shape != result.shape:
        return None, None
    difference = np.abs(original - result)
    stats = (difference.mean(), difference.max(), (difference.max(axis=2) > 0).mean() * 100.0)
    visual = np.clip(difference * AMPLIFY, 0, 255).astype(np.uint8)
    return stats, visual

def process_file(name):
    original_path = os.path.join(INPUT_DIR, name)
    stem = os.path.splitext(name)[0]
    output_path = find_output(stem)
    if output_path is None:
        print(f"No output found for {name}")
        return
    stats, visual = compare(original_path, output_path)
    if stats is None:
        print(f"Size mismatch: {name}")
        return
    Image.fromarray(visual).save(os.path.join(DIFF_DIR, stem + "_diff.png"))
    print(f"{name}: mean change {stats[0]:.3f}, max change {stats[1]}, pixels altered {stats[2]:.1f}%")

def main():
    os.makedirs(DIFF_DIR, exist_ok=True)
    names = sorted(entry for entry in os.listdir(INPUT_DIR) if entry.lower().endswith(VALID_EXTENSIONS))
    if not names:
        print(f"No images found in '{INPUT_DIR}'")
        return
    for name in names:
        process_file(name)

if __name__ == "__main__":
    main()
    print("Done")
