#coding:utf-8
import  cv2
import numpy as np
import time
import sys
#import clasif as cl
import clasificador as c

import analisis as a

thresh_turn = 100

texto = "Linea recta,Curva derecha,Curva izquierda,Bifurcacion o T,Cruce en X".split (",")
cl = c.Clasificador()
# Inicio la captura de imagenes
capture = cv2.VideoCapture("../videos/line0.mp4")

_,img = capture.read()
# Ahora clasifico el video
n = -1

fourcc = cv2.VideoWriter_fourcc(*"XVID")
video = cv2.VideoWriter("resultado.avi",fourcc,24,(img.shape[1],img.shape[0]))
half = float(img.shape[1]/2)
#  entrada: puntode entrada
#  salida_final: salida que va a tomar el robot en esta iteración
#  salida_mantener: salida que marcaba la última flecha que se encontró el robot
#  cnt_una: contador de frames seguidos en los que ha aparecido una sola linea
#  salidas_flecha: lista de todas las salidas que ha marcado la flecha

entrada=None
salida_mantener = None
salidas_flecha = []
cnt_una = thresh_turn
estado = "una linea"
primero = True
while (capture.isOpened()):
    ret,img = capture.read()
    n+=1
    if not ret:
        break
    h = int(img.shape[0]*0.3)
    # obtengo la matriz de categorias
    cats = cl.classif_img(img [h:,:])
    # preparo imagen binaria de la linea
    lin = (cats ==2).astype (np.uint8)
    mar = a.encontrar_icono((cats ==0).astype (np.uint8))
    paleta = np.array([[0,0,0],[255,255,255],[0,0,255],[0,0,0]],dtype=np.uint8)

    # detecto tipo de linea
    tipo= a.tipo_linea (lin)
    if (tipo < a.DOS_SALIDAS): # una linea
        sys.stdout.write("Veo una sola linea")
        cnt_una +=1
        salidas_flecha = []
        if cnt_una < thresh_turn: # lleva poco rato viendo una sola line
            sys.stdout.write(", pero no la puedo seguir todavia\n")
            tcolor = (255,0,0) # color azul
            salida_final = salida_mantener
        elif np.any(mar): # ve una flecha
            sys.stdout.write(" y además un icono, que debe ser una marca\n")
            # reconocer icono
            pass
        else: # lleva suficiente sin ver una flecha como para seguir solo la linea
            sys.stdout.write(" y no hay cruces\n")
            tcolor = (0,255,0) # color verde
            entrada,salida_final = a.entrada_salida(cats,entrada)
    else: # dos lineas
        sys.stdout.write("Veo un cruce")
        
        if np.any(mar): # ve una flecha -> toma nota de su salida
            sys.stdout.write(" y sigo la flecha")
            tcolor = (255,0,255) # color morado
            entrada,nueva_salida_flecha=a.entrada_salida(cats,entrada)
            salidas_flecha.append(nueva_salida_flecha)
        else:
            salidas_flecha = []
        if salidas_flecha != []: # llevamos un rato viendo flecha
            cnt_una = 0
            sys.stdout.write(". Además veo la flecha desde hace rato {}\n".format(salidas_flecha))
            salida_mantener =tuple(np.median(np.array(salidas_flecha),axis=0).astype(int))
            salida_final = salida_mantener
        else: # hemos perdido la flecha (¿Es posible?) -> seguimos como si fuera una linea
            sys.stdout.write(", pero he perdido la flecha así que sigo como si nada\n")
            entrada,salida_final = a.entrada_salida(cats,entrada)
    if salida_final != None:
        salida = salida_final
    lineDistance = ((half-salida[0]))/half
    cv2.putText(img,str(lineDistance),(0,img.shape[0]-20),cv2.FONT_HERSHEY_SIMPLEX,2,tcolor,3)
    cv2.putText(img,"{}({})".format(estado,cnt_una),(0,40),cv2.FONT_HERSHEY_SIMPLEX,1,tcolor,1)
    cv2.circle(img,tuple(salida),3,(0,255,0),-1)
    #cv2.circle(img,(self_entrada[0],img[h:,:].shape[1]),3,(0,255,0),-1)
    if np.any(mar):
        p1,p2,salida = a.direccion_flecha(mar)
        cv2.arrowedLine(img,p1,p2,(255,255,255),3)

    #cv2.imshow("segmentacion",paleta[lin])
    #cv2.imshow("video",img)
    cv2.imshow("imagen",img)
    video.write(img)
    primero = False
    cv2.waitKey(1)
cv2.destroyAllWindows ()
video.release ()

