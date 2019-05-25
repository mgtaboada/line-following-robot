import  cv2
import numpy as np
import time

#import clasif as cl
import clasificador as c

import analisis as a
import pickle #Guardar los datos

cl = c.Clasificador()


n = -1


entrada=None

videos = ["../videos/persona.mp4","../videos/escalera.mp4","../videos/cruz.mp4","../videos/telefono.mp4", "../videos/flecha1.mp4", "../videos/flecha2.mp4"]
etiquetas = ["persona","escalera","cruz","telefono","flecha","flecha"]

dataset = []
dataset_etiquetas = []

for i in range(len(videos)):
    # Inicio la captura de imagenes
    capture = cv2.VideoCapture(videos[i])
    longitud = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    _,img = capture.read()
    modulador = longitud//100
    po = 0
    while (capture.isOpened()):
        n+=1
        ret,img = capture.read()
        
        if n % modulador != 0:
            continue
        if not ret:
            break
        
        po += 1
        h = int(img.shape[0]*0.3)

        cats = cl.classif_img(img)
        # preparo imagen binaria de la marca
        sym = (cats ==0).astype (np.uint8)
        sym = a.limpiar_img(sym)
        paleta = np.array([[0,0,0],[255,255,255],[0,0,255],[0,0,0]],dtype=np.uint8)
        grayscale_img = cv2.cvtColor(cv2.cvtColor(paleta[sym],cv2.COLOR_RGB2BGR),cv2.COLOR_BGR2GRAY)
        _, binary_img = cv2.threshold(grayscale_img, 20, 255, cv2.THRESH_BINARY)
        moments = cv2.HuMoments(cv2.moments(binary_img)).flatten()
        dataset.append(moments)
        dataset_etiquetas.append(etiquetas[i])
        cv2.imshow ("imagen", grayscale_img)
        cv2.waitKey (10)
    print(n)
    print(po)
with open("datasetFiguras", 'wb') as f:
        pickle.dump(dataset,f)    
with open("datasetEtiquetasFiguras", 'wb') as f:
        pickle.dump(dataset_etiquetas,f)    
cv2.destroyAllWindows ()
#video.release ()

