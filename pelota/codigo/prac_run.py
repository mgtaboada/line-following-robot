#!/usr/bin/env python
#coding: utf-8
import  cv2
import numpy as np

#import clasif as cl
import clasificador as c # clasificador euclideo
import ball # funciones para tratar la pelota

cl = c.Clasificador()
# Inicio la captura de imagenes
capture = cv2.VideoCapture("../videos/video0.mp4")

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
    ul,dr,center = size.ball_square(cats)
    cv2.rectangle(img,ul,dr,(255,0,0),2)
    cv2.circle(img,center,3,(0,0,255),2)
    cv2.imshow("Original",img)
    cv2.waitKey(50)

