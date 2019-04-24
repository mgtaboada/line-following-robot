import cv2
import numpy as np
import pickle #Cargar los datos

import sklearn.neighbors as n

import clasificador as c
import analisis as a

etiquetas = {"telefono":"Telefono","cruz":"Botiquin","persona":"Caballeros","escalera":"Escalera"}

with open("datasetFiguras", 'rb') as f:
    dataset = pickle.load(f)
with open("datasetEtiquetasFiguras", 'rb') as f:
    dataset_etiquetas = pickle.load(f)

dataset = np.array(dataset)
dataset_etiquetas = np.array(dataset_etiquetas)

skmaha = n.KNeighborsClassifier(1,algorithm='brute',metric='mahalanobis',metric_params={'V':np.cov(dataset)})

skmaha.fit(dataset,dataset_etiquetas)

cl = c.Clasificador()
video = "../videos/recorrido.avi"
capture = cv2.VideoCapture(video)
_,img = capture.read()

fourcc = cv2.VideoWriter_fourcc(*"XVID")
video = cv2.VideoWriter("resultado.avi",fourcc,24,(img.shape[1],img.shape[0]))

while (capture.isOpened()):
    ret,img = capture.read()
    if not ret:
        break


    h = 0#int(img.shape[0]*0.3)
    # Clasifico una imagen de cada 25

    # obtengo la matriz de categorias
    cats = cl.classif_img(img [h:,:])
    # preparo imagen binaria de la marca
    sym = (cats ==0).astype (np.uint8)
    if np.any(sym==1):
        sym = a.limpiar_img(sym)
        paleta = np.array([[0,0,0],[255,255,255],[0,0,255],[0,0,0]],dtype=np.uint8)
        grayscale_img = cv2.cvtColor(cv2.cvtColor(paleta[sym],cv2.COLOR_RGB2BGR),cv2.COLOR_BGR2GRAY)
        _, binary_img = cv2.threshold(grayscale_img, 20, 255, cv2.THRESH_BINARY)
        
        conts,hier = cv2.findContours(binary_img,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
        
        if len(conts) > 0:
            moments = cv2.HuMoments(cv2.moments(binary_img)).flatten()
            mark = skmaha.predict([moments])[0]
            cont = conts[0] 
            cv2.drawContours(img,[cont],0,(0,255,0),3)
            cv2.putText (img [h:,: ],etiquetas[mark],(0,img [h:,:].shape[0]-20),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
    # detecto tipo de linea
    video.write (img)
    cv2.imshow ("imagen", img)
    cv2.waitKey (10)

cv2.destroyAllWindows ()
video.release ()
