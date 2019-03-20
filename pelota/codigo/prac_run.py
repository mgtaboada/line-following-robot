#!/usr/bin/env python
#coding: utf-8
import  cv2
import numpy as np

#import clasif as cl
import clasificador as c # clasificador euclideo
import ball # funciones para tratar la pelota

cl = c.Clasificador()

# Calibración de la cámara: obtener distancia focal a partir de foto con distancia conocida
img = cv2.imread("../imagenes/30cm.png")
cats = cl.classif_img(img)
focal_length = ball.get_focal_length(cats,30.0) # sabemos que la foto se hizo desde 30cm

# Inicio la captura de imagenes
capture = cv2.VideoCapture("../videos/video0.mp4")
_,img = capture.read()

# creo un video nuevo como resultado
# no he conseguido crear un .mp4
video = cv2.VideoWriter("../videos/resultado.avi",cv2.cv.CV_FOURCC(*"XVID"),24,(img.shape[1],img.shape[0]))
# Ahora clasifico el video
n = -1
while (capture.isOpened()):
    ret,img = capture.read()

    if not ret:
        break

    # obtengo la matriz de categorias
    cats = cl.classif_img(img)
    paleta = np.array([[255,0,0],[0,0,255],[255,255,0],[0,255,0]],dtype=np.uint8)
    # ahora pinto la imagen
    #cv2.imshow("Segmentacion Euclid",cv2.cvtColor(paleta[cats],cv2.COLOR_RGB2BGR))
    # Obtengo los parámetros de la pelota en la imagen
    ul,dr,center = ball.ball_square(cats)
    # calculo la distancia de la pelota a la camara
    dist = ball.distance_to_camera(dr,ul,focal_length)

    #Dibujar un rectángulo alrededor de la pelota y un punto en su centro
    cv2.rectangle(img,ul,dr,(255,0,0),2)
    cv2.circle(img,center,3,(0,0,255),-1)
    
    cv2.putText(img,"{:06.3f}cm".format(dist),(img.shape[1]-250,img.shape[0]-20),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,0,0),3)
    # Guardar el video 
    video.write(img)
    
    # Mostrar el video
    # cv2.imshow("Original",img)
    # cv2.waitKey(50)
cv2.destroyAllWindows()
video.release()
print("Terminado")
