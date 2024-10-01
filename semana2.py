import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

### Actividad 1 :)

img = cv2.imread('imagen_prueba.jpg', cv2.IMREAD_GRAYSCALE)

_, otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
adaptive = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

plt.figure(figsize=(13, 5))
plt.subplot(1, 3, 1)
plt.imshow(img, cmap='gray')
plt.title('Imagen original: ')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(otsu, cmap='gray')
plt.title('Umbralizacion otsu: ')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(adaptive, cmap='gray')
plt.title('Umbralizacion adaptativa')
plt.axis('off')

plt. show()

### Actividad 2 :|

img = cv2.imread('imagen_prueba.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def region_growing(img, seed, threshold):
    rows, cols, _ = img.shape
    segmented = np.zeros((rows, cols), np.uint8)
    segmented[seed[0], seed[1]] == 255
    pixel_list = [seed]
    seed_color = img[seed[0], seed[1]]
    while len(pixel_list) > 0 :
        pix = pixel_list.pop(0)
        for i in range(pix[0]-1, pix[0]+2):
            for j in range(pix[1]-1, pix[1]+2):
                if i>=0 and i<rows and j >= 0 and j<cols:
                    if segmented[i, j] == 0 and np.sum(np.abs(img[i, j] - seed_color)) <threshold:
                        segmented[i, j] = 255
                        pixel_list.append([i, j])
    return segmented

seed = (img.shape[0]//2, img.shape[1]//2)
reg_grown = region_growing(img, seed, 30)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

sure_bg = cv2.dilate(opening, kernel, iterations=3)
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

sure_fg = np.uint8(sure_fg)
uknown = cv2.subtract(sure_bg, sure_fg)

ret, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1
markers[uknown ==255] = 0

makers = cv2.watershed(img, markers)
watershed_result = img.copy()
watershed_result[markers == -1] = [255, 0, 0]

plt.figure(figsize=(13, 5))
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Imagen original: ')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(reg_grown, cmap='gray')
plt.title('Crecimiento de regiones: ')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(watershed_result, cv2.COLOR_BGR2RGB))
plt.title('Watershed')
plt.axis('off')
plt.show()


### Actividad 3 :)

iimg = cv2.imread('imagen_prueba.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create()

keypoints, descriptions = sift.detectAndCompute(gray, None)
img_keypoints = cv2.drawKeypoints(img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

plt.figure(figsize=(10, 6))
plt.subplot(1,2, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Imagen original: ')
plt.axis('off')


plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(img_keypoints, cv2.COLOR_BGR2RGB))
plt.title('Puntos de interes detectados con SIFT: ')
plt.axis('off')


plt.show()




