
import  cv2
import numpy as np

#import clasif as cl
import clasificador as c

import analisis as a
texto = "Linea recta,Curva derecha,Curva izquierda,Bifurcacion o T,Cruce en X".split (",")
cl = c.Clasificador()
# Inicio la captura de imagenes
capture = cv2.VideoCapture("../videos/line0.mp4")

_,img = capture.read()
# Ahora clasifico el video
n = -1
h = int(img.shape[0]*0.3)

fourcc = cv2.VideoWriter_fourcc(*"XVID")
analisis = cv2.VideoWriter("resultado_analisis.avi",fourcc,24,(img.shape[1],img.shape[0]))
segmentacion = cv2.VideoWriter("resultado_segmentacion.avi",fourcc,24,(img.shape[1],img.shape[0]-h))
entrada=None
while (capture.isOpened()):
    ret,img = capture.read()
    n+=1
    if not ret:
        break


    # Clasifico una imagen de cada 25
    n = 0
    # obtengo la matriz de categorias
    cats = cl.classif_img(img [h:,:])
    # preparo imagen binaria de la linea
    lin = (cats ==2).astype (np.uint8)
    paleta = np.array([[0,0,255],[0,0,0],[255,0,0],[0,0,0]],dtype=np.uint8)
    #cv2.imshow("Segmentacion Euclid",cv2.cvtColor(paleta[cats],cv2.COLOR_RGB2BGR))
    segmentacion.write (paleta [cats])
    #cv2.waitKey (10)
    # detecto tipo de linea
    tipo= a.tipo_linea (lin)
    
    flecha = cats == 0
    if np.any (flecha):
        f1,f2,_,_ = a.direccion_flecha ((flecha).astype (np.uint8))
        cv2.arrowedLine(img [h:,:],f1,f2,(0,255,0),4)
    if tipo is not None:
        cv2.putText (img [h:,: ],texto [tipo],(0,img [h:,:].shape[0]-20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1)
    else:
        cv2.putText (img [h:,: ],"Sin linea",(0,img [h:,:].shape[0]-20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1)
    entrada,salida = a.entrada_salida (cats,entrada)
    cv2.circle(img,(entrada [0],entrada [1]+h),3,(0,255,0),-1)
    cv2.circle(img,(salida [0],salida [1]+h),3,(0,0,255),-1)
    analisis.write (img)
    

cv2.destroyAllWindows ()
segmentacion.release ()
analisis.release () 
