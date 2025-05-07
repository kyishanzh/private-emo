import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from PrivatizationRunner import PrivatizationRunner
import cv2
from sklearn.cluster import KMeans
from scipy.signal import convolve2d, wiener

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

def blur(image, blur_ksize = (7, 7)):
    blurred = cv2.GaussianBlur(image, blur_ksize, 1)
    return blurred

def wiener_deblur(blurred, ksize=(3, 3)):
    if len(blurred.shape) == 2:  # grayscale
        channels = [blurred]
    else:  # color image
        channels = cv2.split(blurred)  # Split into B, G, R

    deblurred_channels = []
    for ch in channels:
        deblurred = np.clip(wiener(ch, ksize), 0, 255)
        deblurred = np.nan_to_num(deblurred, nan=0.0).astype(np.uint8)
        deblurred_channels.append(deblurred)

    # Merge channels back to BGR image
    return cv2.merge(deblurred_channels)

def quantize(blurred, k=16):
    data = blurred.reshape((-1,3))
    # init_vals = np.linspace(0, 255, k)
    # result = [(i, value) for i, value in enumerate(init_vals)]
    kmeans = KMeans(n_clusters=k, random_state=42, n_init = 'auto').fit(data)
    quant = kmeans.cluster_centers_[kmeans.labels_].reshape(blurred.shape).astype(np.uint8)
    return quant

def blur(image, blur_ksize = (5, 5)):
    blurred = cv2.GaussianBlur(image, blur_ksize, 1)
    return blurred
# Load image
# img = np.array(Image.open('data/inference/images.jpeg'))
# quantized = quantize(img, 3)  # 3-bit quantization
# show_images([img, quantized], titles=['Original', 'Quantized'])


# def quantize_3bit(img):
#     #print("preclip", quantize(img, 3))
#     img_out = np.clip(quantize(img, 3), 0, 255).astype(np.uint8)
#     #print("postclip",img_out)
#     return Image.fromarray(img_out)

out_dir = "CelebA_Quantized"

runner = PrivatizationRunner(celebA_dir, out_dir, [blur, quantize], useCV = True)

runner.run()
