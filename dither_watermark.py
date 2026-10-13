from PIL import Image
import numpy as np
import os

INPUT_DIR = "images"
OUTPUT_DIR = "output"
NOISE_STRENGTH = 0.02
JPEG_QUALITY = 95
VALID_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")

def load_image(path):
    image = Image.open(path)
    if image.mode not in ("RGB", "RGBA"):
        image = image.convert("RGBA" if "A" in image.mode else "RGB")
    return image

def apply_correlated_noise(image, strength):
    array = np.asarray(image).astype(np.float32)
    has_alpha = array.shape[2] == 4
    rgb = array[:, :, :3]
    noise_amplitude = strength * 255.0
    noise = np.random.normal(loc=0.0, scale=noise_amplitude, size=rgb.shape[:2]).astype(np.float32)
    rgb = rgb + noise[:, :, np.newaxis]
    rgb = np.clip(rgb, 0, 255)
    array[:, :, :3] = rgb
    if has_alpha:
        return Image.fromarray(array.astype(np.uint8), mode="RGBA")
    return Image.fromarray(array.astype(np.uint8), mode="RGB")

def flatten_alpha(image):
    if image.mode != "RGBA":
        return image
    background = Image.new("RGB", image.size, (255, 255, 255))
    background.paste(image, mask=image.getchannel("A"))
    return background

def gather_image_paths(directory):
    if not os.path.isdir(directory):
        return []
    entries = sorted(os.listdir(directory))
    paths = []
    for name in entries:
        if name.lower().endswith(VALID_EXTENSIONS):
            paths.append(os.path.join(directory, name))
    return paths

def process_image(path, output_dir):
    try:
        image = load_image(path)
        result = apply_correlated_noise(image, NOISE_STRENGTH)
        result = flatten_alpha(result)
        name = os.path.splitext(os.path.basename(path))[0]
        output_path = os.path.join(output_dir, name + ".jpg")
        result.save(output_path, format="JPEG", quality=JPEG_QUALITY)
        print(f"Processed: {path} -> {output_path}")
    except Exception as error:
        print(f"Failed: {path} ({error})")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    paths = gather_image_paths(INPUT_DIR)
    if not paths:
        print(f"No images found in '{INPUT_DIR}'")
        return
    for path in paths:
        process_image(path, OUTPUT_DIR)

if __name__ == "__main__":
    main()
    print("Done")
