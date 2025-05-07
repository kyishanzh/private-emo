import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from PrivatizationRunner import PrivatizationRunner

# our input directories
celebA_dir = "CelebrityFacesDataset"
fer_dir = "data/real"

def show_images(images, titles=None, figsize=(15,5)):
    """Display multiple images side by side"""
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=figsize)

    if n == 1:
        axes = [axes]

    for i, (ax, img) in enumerate(zip(axes, images)):
        ax.imshow(img)
        ax.axis('off')
        if titles:
            ax.set_title(titles[i])

    plt.tight_layout()
    plt.show()

def quantize(img, bits):
    """Reduce color depth (8-bit example)"""
    return np.floor(img / (2**(8 - bits))) * (2**(8 - bits))

# Load image
# img = np.array(Image.open('data/inference/images.jpeg'))
# quantized = quantize(img, 3)  # 3-bit quantization
# show_images([img, quantized], titles=['Original', 'Quantized'])

def quantize_3bit(path):
    img = np.array(Image.open(path))
    img_out = np.clip(quantize(img, 3), 0, 255).astype(np.uint8)
    return Image.fromarray(img_out)

out_dir = "CelebA_Quantized"

runner = PrivatizationRunner(celebA_dir, out_dir, quantize_3bit, useCV = False)

runner.run()
