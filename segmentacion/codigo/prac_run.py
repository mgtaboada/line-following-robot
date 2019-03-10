
import  cv2
import numpy as np

#import clasif as cl
import clasificador as c

cl = c.Clasificador()
# Inicio la captura de imagenes
capture = cv2.VideoCapture("../videos/video1.mp4")

# Ahora clasifico el video
n = -1
while (capture.isOpened()):
    ret,img = capture.read()
    n+=1
    if not ret:
        break
    # Clasifico una imagen de cada 25
    if (n%25)==0:
        n = 0
        # obtengo la matriz de categorias
        cats = cl.classif_img(img)
        paleta = np.array([[255,0,0],[0,255,0],[0,0,255],[0,0,0]],dtype=np.uint8)
        # ahora pinto la imagen
        cv2.imshow("Segmentacion Euclid",cv2.cvtColor(paleta[cats],cv2.COLOR_RGB2BGR))
    cv2.imshow("Original",img)
    cv2.waitKey(10)

