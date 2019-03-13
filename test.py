import numpy as np
import cv2
from scipy.signal import convolve2d
import matplotlib.pyplot as plt

# Classic Edge Detection Masks
def euler_filter():
	print("Euler Filter")

	h_dx = np.array([[0, 0, 0], [-1, 1, 0], [0, 0, 0]])
	h_dy = np.array([[0, -1, 0], [0, 1, 0], [0, 0, 0]])

	return h_dx, h_dy

def bilinear_filter():
	print('Bilinear Filter')

	h_dx = np.array([[0, 0, 0], [-1, 0, 1], [0, 0, 0]])
	h_dy = np.array([[0, -1, 0], [0, 0, 0], [0, 1, 0]])

	return h_dx, h_dy

def prewitt_filter():
	print('Prewitt Filter')

	h_dx = (1 / 6) * np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
	h_dy = np.transpose(h_dx)

	return h_dx, h_dy

def sobel_filter():
	print('Sobel Filter')

	h_dx = (1 / 8) * np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
	h_dy = np.transpose(h_dx)

	return h_dx, h_dy


# img = cv2.imread("Lenna_image.png")
img = cv2.imread("flower.jpeg", 1)

plt.imshow(img)
plt.title('Original Image')
plt.show()
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

print(type(img))
print(img.shape)
plt.imshow(img)
plt.title('Grayscale Image')
plt.show()

# h_dx, h_dy = euler_filter()
# h_dx, h_dy = bilinear_filter()
# h_dx, h_dy = prewitt_filter()
h_dx, h_dy = sobel_filter()

print(type(h_dy), '\tShape : ', h_dx.shape)
print(h_dy)

new_image_x = convolve2d(h_dx, img, mode = 'valid')
new_image_y = convolve2d(h_dy, img, mode = 'valid')
new_image = np.add(new_image_x, new_image_y)
# plt.subplot(5, 3, 1)
plt.imshow(new_image_x)
plt.title('H_DX Convolution')
plt.show()
# plt.subplot(5, 3, 3)
plt.imshow(new_image_y)
plt.title('H_DY Convolution')
plt.show()
# plt.subplot(5, 3, 2)
plt.imshow(new_image)
print(new_image.shape)

plt.title('Final Image')
plt.show()