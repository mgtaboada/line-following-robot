
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

fourcc = cv2.VideoWriter_fourcc(*"XVID")
video = cv2.VideoWriter("resultado.avi",fourcc,24,(img.shape[1],img.shape[0]))

while (capture.isOpened()):
    ret,img = capture.read()
    n+=1
    if not ret:
        break
    

    h = int(img.shape[0]*0.3)
    # Clasifico una imagen de cada 25
    n = 0
    # obtengo la matriz de categorias
    cats = cl.classif_img(img [h:,:])
    # preparo imagen binaria de la linea
    cats = (cats ==2).astype (np.uint8)
    paleta = np.array([[0,0,0],[255,255,255],[0,0,255],[0,0,0]],dtype=np.uint8)
    cv2.imshow("Segmentacion Euclid",cv2.cvtColor(paleta[cats],cv2.COLOR_RGB2BGR))
    cv2.waitKey (10)
    # detecto tipo de linea
    tipo,img [h:,:] = a.tipo_linea (cats,img [h:,:])
    if tipo is not None:
        cv2.putText (img [h:,: ],texto [tipo],(0,img [h:,:].shape[0]-20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1)
    video.write (img)

cv2.destroyAllWindows ()
video.release ()
 
