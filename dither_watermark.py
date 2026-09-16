from PIL import Image
import numpy as np

INPUT_PATH = input("Input file: ") or "input.png"
OUTPUT_PATH = "output.png"
NOISE_STRENGTH = 0.01

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

def main():
    image = load_image(INPUT_PATH)
    result = apply_correlated_noise(image, NOISE_STRENGTH)
    result.save(OUTPUT_PATH)

if __name__ == "__main__":
    main()
    print("Done")
