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


def gaussian_derivative(image, ksize=7, sigma=1.0):
    # Ensure image is grayscale for derivative computation
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Apply Gaussian blur first
    blurred = cv2.GaussianBlur(gray, (ksize, ksize), sigma)

    # Compute gradients in x and y directions (derivatives)
    grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=ksize)
    grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=ksize)

    # Combine gradients to form edge magnitude
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    magnitude = np.clip(magnitude, 0, 255).astype(np.uint8)

    # Optional: convert back to BGR for compatibility with color pipeline
    return cv2.cvtColor(magnitude, cv2.COLOR_GRAY2BGR)

def quantize(blurred, k=16):
    data = blurred.reshape((-1,3))
    # init_vals = np.linspace(0, 255, k)
    # result = [(i, value) for i, value in enumerate(init_vals)]
    kmeans = KMeans(n_clusters=k, random_state=42, n_init = 'auto').fit(data)
    quant = kmeans.cluster_centers_[kmeans.labels_].reshape(blurred.shape).astype(np.uint8)
    return quant

# Load image
# img = np.array(Image.open('data/inference/images.jpeg'))
# quantized = quantize(img, 3)  # 3-bit quantization
# show_images([img, quantized], titles=['Original', 'Quantized'])


# def quantize_3bit(img):
#     #print("preclip", quantize(img, 3))
#     img_out = np.clip(quantize(img, 3), 0, 255).astype(np.uint8)
#     #print("postclip",img_out)
#     return Image.fromarray(img_out)

out_dir = "CelebA_Gaussian+Quantized"

runner = PrivatizationRunner(celebA_dir, out_dir, [gaussian_derivative], useCV = True)

runner.run()
