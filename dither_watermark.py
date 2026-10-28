from PIL import Image
import numpy as np
import os
from scipy import ndimage

INPUT_DIR = "images"
OUTPUT_DIR = "output"
NOISE_STRENGTH = 0.012
NOISE_FLOOR = 0.7
WARP_AMPLITUDE = 0.4
WARP_SMOOTHNESS = 20.0
JPEG_QUALITY = 95
VALID_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")

def load_image(path):
    image = Image.open(path)
    if image.mode not in ("RGB", "RGBA"):
        image = image.convert("RGBA" if "A" in image.mode else "RGB")
    return image

def build_noise(shape):
    return np.random.normal(0.0, 1.0, size=shape).astype(np.float32)

def build_displacement(shape):
    field = np.random.normal(0.0, 1.0, size=(2,) + shape).astype(np.float32)
    for axis in range(2):
        smooth = ndimage.gaussian_filter(field[axis], WARP_SMOOTHNESS, mode="reflect")
        deviation = smooth.std()
        if deviation > 0:
            smooth = smooth / deviation
        field[axis] = smooth * WARP_AMPLITUDE
    return field

def warp_rgb(rgb, displacement):
    height, width = rgb.shape[:2]
    grid_y, grid_x = np.meshgrid(np.arange(height, dtype=np.float32), np.arange(width, dtype=np.float32), indexing="ij")
    coords = np.stack([grid_y + displacement[0], grid_x + displacement[1]])
    warped = np.empty_like(rgb)
    for channel in range(3):
        warped[:, :, channel] = ndimage.map_coordinates(rgb[:, :, channel], coords, order=3, mode="reflect")
    return warped

def apply_correlated_noise(image, strength):
    array = np.asarray(image).astype(np.float32)
    rgb = array[:, :, :3]
    if WARP_AMPLITUDE > 0:
        rgb = warp_rgb(rgb, build_displacement(rgb.shape[:2]))
    noise = build_noise(rgb.shape[:2])[:, :, np.newaxis]
    rgb = rgb * (1.0 + strength * noise) + NOISE_FLOOR * noise
    array[:, :, :3] = np.clip(rgb, 0, 255)
    return Image.fromarray(np.rint(array).astype(np.uint8))

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
        result.save(output_path, format="JPEG", quality=JPEG_QUALITY, subsampling=0)
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
    print("Done")

if __name__ == "__main__":
    main()
