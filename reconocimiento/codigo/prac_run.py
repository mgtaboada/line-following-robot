import  cv2
import numpy as np
import time

#import clasif as cl
import clasificador as c

import analisis as a
import pickle #Guardar los datos
texto = "Linea recta,Curva derecha,Curva izquierda,Bifurcacion o T,Cruce en X".split (",")
cl = c.Clasificador()
# Inicio la captura de imagenes

# Ahora clasifico el video
n = -1

#fourcc = cv2.VideoWriter_fourcc(*"XVID")
#video = cv2.VideoWriter("resultado.avi",fourcc,24,(img.shape[1],img.shape[0]))

entrada=None

videos = ["../videos/persona.mp4","../videos/escalera.mp4","../videos/cruz.mp4","../videos/telefono.mp4"]
etiquetas = ["persona","escalera","cruz","telefono"]

dataset = []
dataset_etiquetas = []

for i in range(len(videos)):
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
        # Clasifico una imagen de cada 25
        #n = 0
        # obtengo la matriz de categorias
        #cats = cl.classif_img(img [h-20:,:])
        cats = cl.classif_img(img)
        # preparo imagen binaria de la linea
        lin = (cats ==2).astype (np.uint8)
        paleta = np.array([[0,0,0],[255,255,255],[0,0,255],[0,0,0]],dtype=np.uint8)
        #cv2.imshow("Segmentacion Euclid",cv2.cvtColor(paleta[cats],cv2.COLOR_RGB2BGR))
        #cv2.waitKey (10)
        # detecto tipo de linea
        #tipo = a.tipo_linea (lin)
    
        grayscale_img = cv2.cvtColor(cv2.cvtColor(paleta[cats],cv2.COLOR_RGB2BGR),cv2.COLOR_BGR2GRAY)
        _, binary_img = cv2.threshold(grayscale_img, 20, 255, cv2.THRESH_BINARY)
        moments = cv2.HuMoments(cv2.moments(binary_img)).flatten()
        dataset.append(moments)
        dataset_etiquetas.append(etiquetas[i])

    
        #flecha = cats == 0
        #if np.any (flecha):
        #    f1,f2,_,_ = a.direccion_flecha ((flecha).astype (np.uint8))
        #    cv2.arrowedLine(img [h:,:],f1,f2,(0,0,255),4)
        #if tipo is not None:
        #    cv2.putText (img [h:,: ],texto [tipo],(0,img [h:,:].shape[0]-20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1)
    
        #st = time.time()
        #entrada,salida = a.entrada_salida (cats,entrada)
        #print(time.time()-st)
        #cv2.circle(img,(entrada [0],entrada [1]+h),3,(0,255,0),-1)
        #cv2.circle(img,(salida [0],salida [1]+h),3,(0,0,255),-1)
        #video.write (img)
        #cv2.imshow ("imagen", img)
        cv2.waitKey (10)
    print(n)
    print(po)
with open("datasetFiguras", 'wb') as f:
        pickle.dump(dataset,f)    
with open("datasetEtiquetasFiguras", 'wb') as f:
        pickle.dump(dataset_etiquetas,f)    
cv2.destroyAllWindows ()
#video.release ()

