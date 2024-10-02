import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# De igual manera ingresé las 3 actividades en el archivo
### Actividad 1 :)

# se lee la imagen en escala de grises
img = cv2.imread('imagen_prueba.jpg', cv2.IMREAD_GRAYSCALE)

#se aplica la unbralizacion global con el metodo de Otsu y la adaptativa
_, otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
adaptive = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

# primero se crea el area designada para mostrar las imagenes
plt.figure(figsize=(13, 5))

#se crea el campo de la imagen original
plt.subplot(1, 3, 1)
plt.imshow(img, cmap='gray')
plt.title('Imagen original: ')
plt.axis('off')

# se crea el campo para la umbralizacion por el metodo de Otsu
plt.subplot(1, 3, 2)
plt.imshow(otsu, cmap='gray')
plt.title('Umbralizacion otsu: ')
plt.axis('off')

#se crea el campo para la umbralizacion adaptativa
plt.subplot(1, 3, 3)
plt.imshow(adaptive, cmap='gray')
plt.title('Umbralizacion adaptativa')
plt.axis('off')

#se muestra el area con las imagenes
plt.show()


### Actividad 2 :|

#se lee nuevamente la imagen y se transforma de  formato bgr a rgb
img = cv2.imread('imagen_prueba.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#se crea la funcion para el crecimiento de las regiones
def region_growing(img, seed, threshold):  
    rows, cols, _ = img.shape   #inicializamos las variables
    segmented = np.zeros((rows, cols), np.uint8)
    segmented[seed[0], seed[1]] == 255
    pixel_list = [seed]
    seed_color = img[seed[0], seed[1]]
    #en el ciclo while se utiliza una estructura de lista para procesar los pixeles de la region
    while len(pixel_list) > 0 :    
        pix = pixel_list.pop(0) 
        #se itera sobre los vecinos conectados de cada pixel
        for i in range(pix[0]-1, pix[0]+2):         
            for j in range(pix[1]-1, pix[1]+2):
                if i>=0 and i<rows and j >= 0 and j<cols:
                    #condiciones para agregar un pixel a la region
                    if segmented[i, j] == 0 and np.sum(np.abs(img[i, j] - seed_color)) <threshold:
                        segmented[i, j] = 255
                        pixel_list.append([i, j])
    return segmented # se retorna la imagen sgmentada

#se crea la semilla y le enviamos los parametros a la funcion 
seed = (img.shape[0]//2, img.shape[1]//2)
reg_grown = region_growing(img, seed, 30)

#se convierte la imagen a escala de grises
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
kernel = np.ones((3, 3), np.uint8) #transformaciones morfologicas de apertura y dilatacion
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

sure_bg = cv2.dilate(opening, kernel, iterations=3)

# se calcula la transformacion de ladistancia y deteccion del area segura
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

# se detectan las areas desconocidas
sure_fg = np.uint8(sure_fg)
uknown = cv2.subtract(sure_bg, sure_fg)

#se encuentran los componentes conectados y se prepara para la transformacion Watershed
ret, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1
markers[uknown ==255] = 0

#se aplica el algoritmo Watershed
makers = cv2.watershed(img, markers)
#se vicualiza el resultado final
watershed_result = img.copy()
watershed_result[markers == -1] = [255, 0, 0]

#se crea el area designada para mostrar las imagenes
plt.figure(figsize=(13, 5))
#se crea el campo para mostrar la imagen original
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Imagen original: ')
plt.axis('off')

#se crea el campo para mostrar la imagen con crecimiento de regiones
plt.subplot(1, 3, 2)
plt.imshow(reg_grown, cmap='gray')
plt.title('Crecimiento de regiones: ')
plt.axis('off')

#se crea el campo para mostrar la imagen con la transformacion watershed
plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(watershed_result, cv2.COLOR_BGR2RGB))
plt.title('Watershed')
plt.axis('off')
plt.show() # se muestran todas las imagenes


### Actividad 3 :)

#se lee la imagen y se transforma de  formato bgr a gray
img = cv2.imread('imagen_prueba.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#se inicializa e detector SIFT
sift = cv2.SIFT_create()
#se detectan los puntos clave y los descriptores
keypoints, descriptions = sift.detectAndCompute(gray, None)
#se dibujan los puntos claves
img_keypoints = cv2.drawKeypoints(img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

#se crea el area designada para mostrar las imagenes
plt.figure(figsize=(10, 6))
#se crea el campo para mostrar la imagen original
plt.subplot(1,2, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Imagen original: ')
plt.axis('off')

#se crea el campo para mostrar la imagen con los puntos detectados con SIFT
plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(img_keypoints, cv2.COLOR_BGR2RGB))
plt.title('Puntos de interes detectados con SIFT: ')
plt.axis('off')

#se muestran las imagenes
plt.show()




