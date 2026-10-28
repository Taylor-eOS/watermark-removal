import os
import numpy as np
from PIL import Image

INPUT_DIR = "input_files"
OUTPUT_DIR = "output_files"
NOISE_STRENGTH = 0.006
NOISE_FLOOR = 0.5
TEXTURE_BLUR_SIGMA = 3.0
TEXTURE_SCALE = 8.0
MIN_WEIGHT = 0.5
JPEG_QUALITY = 95
VALID_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")

def load_image(path):
    image = Image.open(path)
    if image.mode not in ("RGB", "RGBA"):
        image = image.convert("RGBA" if "A" in image.mode else "RGB")
    return image

def gaussian_blur(array, sigma):
    height, width = array.shape
    freq_y = np.fft.fftfreq(height)[:, np.newaxis]
    freq_x = np.fft.rfftfreq(width)[np.newaxis, :]
    response = np.exp(-2.0 * (np.pi ** 2) * (sigma ** 2) * (freq_x ** 2 + freq_y ** 2))
    spectrum = np.fft.rfft2(array) * response
    return np.fft.irfft2(spectrum, s=array.shape).astype(np.float32)

def build_noise(shape):
    return np.random.uniform(-1.0, 1.0, size=shape).astype(np.float32)

def build_texture_weight(rgb):
    luminance = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    grad_y, grad_x = np.gradient(luminance)
    magnitude = np.hypot(grad_x, grad_y)
    texture = gaussian_blur(magnitude, TEXTURE_BLUR_SIGMA)
    weight = np.clip(texture / TEXTURE_SCALE, 0.0, 1.0)
    return MIN_WEIGHT + (1.0 - MIN_WEIGHT) * weight

def stochastic_round(values):
    offset = np.random.random(values.shape[:2]).astype(np.float32)
    return np.floor(values + offset[:, :, np.newaxis])

def apply_correlated_noise(image, strength):
    array = np.asarray(image).astype(np.float32)
    rgb = array[:, :, :3]
    weight = build_texture_weight(rgb)[:, :, np.newaxis]
    amplitude = (NOISE_FLOOR + strength * rgb) * weight
    shifted = rgb + build_noise(rgb.shape[:2])[:, :, np.newaxis] * amplitude
    array[:, :, :3] = np.clip(stochastic_round(shifted), 0, 255)
    return Image.fromarray(array.astype(np.uint8))

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

if __name__ == "__main__":
    main()
    print("Done")
