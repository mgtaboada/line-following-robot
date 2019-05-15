import  cv2
import numpy as np
import time

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

entrada=None
salida_mantener = None
self_salidas_flecha = []
cnt_una = 0
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
    #cv2.imshow("Segmentacion Euclid",cv2.cvtColor(paleta[cats],cv2.COLOR_RGB2BGR))
    #cv2.waitKey (10)
    # detecto tipo de linea
    tipo= a.tipo_linea (lin)

    if tipo < a.DOS_SALIDAS:
        cnt_una +=1
        tcolor = (255,0,0)
        if cnt_una > thresh_turn:
            estado = "una linea"
            tcolor = (0,255,0)

    if (tipo >= a.DOS_SALIDAS and np.any(mar)):
          if estado != "flecha":
             self_salidas_flecha = []
          if tipo >= a.DOS_SALIDAS:
             cnt_una = 0
             tcolor = (255,0,255)
          estado="flecha"
    half = float(img.shape[1]/2)
    entrada, salida = a.entrada_salida(cats,self_entrada)
    if primero:
        self_salida = salida
    if estado == "flecha":
        entrada, salida = a.entrada_salida(cats,self_entrada)
        self_salidas_flecha.append(salida)
        salida = self_salidas_flecha[-1]

        if salida is not None:
            self_salida = salida
            self_entrada = entrada
        else:
            salida = self_salida
    if estado == 'una linea' and self_salidas_flecha != []:
        self_salida = np.median(np.array(self_salidas_flecha),axis=0).astype(int)
        self_salidas_flecha = []
        print(self_salida)
        salida = self_salida
    if estado =="una linea" and cnt_una < thresh_turn:
        salida = self_salida
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

