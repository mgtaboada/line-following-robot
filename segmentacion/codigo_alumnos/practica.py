####################################################
# Esqueleto de programa para ejecutar el algoritmo de segmentacion.
# Este programa primero entrena el clasificador con los datos de
#  entrenamiento y luego segmenta el video (este entrenamiento podria
#  hacerse en "prac_ent.py" y aqui recuperar los parametros del clasificador
###################################################


import  cv2
import numpy as np

#import clasif as cl
import clasificador as c

cl = c.Clasificador()
# Inicio la captura de imagenes
capture = cv2.VideoCapture("../videos/video1.mp4")

# Ahora clasifico el video
#n = -1
while (capture.isOpened()):
    ret,img = capture.read()
    #n+=1
    if not ret:
        break
    #if ret:# and(n%15)==0:
        
    # obtengo la matriz de categorias
    #imgn = cv2.resize(img,(120,160))
    #img = imgn
    #if (n%10)==0:
    cats = cl.classif_img(img)
    paleta = np.array([[255,0,0],[0,255,0],[0,0,255],[0,0,0]],dtype=np.uint8)
    # ahora pinto la imagen
    cv2.imshow("Segmentacion Euclid",cv2.cvtColor(paleta[cats],cv2.COLOR_RGB2BGR))
    cv2.imshow("Original",img)
    cv2.waitKey(1)

    #else:
    #    break
