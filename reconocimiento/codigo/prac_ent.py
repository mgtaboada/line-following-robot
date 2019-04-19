#coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

data_marca = []
data_fondo = []
data_linea = []


# Generar dataset si no existe
if not os.path.isfile("../imagenes/dataset.txt"):
    for i in range(10):
        imname = "linea{}.png".format(i)
        markname = "linea{}_m.png".format(i)

        im = cv2.imread(imname)
        mark = cv2.imread(markname)

        imrgb = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
        im =imrgb


        markrgb = cv2.cvtColor(mark,cv2.COLOR_BGR2RGB)
        mark = markrgb

        imn = np.rollaxis((np.rollaxis(im,2)+0.0)/np.sum(im,2),0,3)[:,:,:2]

        data_marca+=imn[np.where(np.all(np.equal(mark,(255,60,0)),2))].tolist()
        data_fondo+=imn[np.where(np.all(np.equal(mark,(0,255,0)),2))].tolist()
        data_linea+=imn[np.where(np.all(np.equal(mark,(0,0,255)),2))].tolist()



        with open("dataset.txt","w") as f:
            for r,g in data_marca:
                f.write("{},{},0\n".format(r,g))
            for r,g in data_fondo:
                f.write("{},{},1\n".format(r,g))
            for r,g in data_linea:
                f.write("{},{},2\n".format(r,g))
# Si ya existía, leerlo
else:
    dataset = np.genfromtxt("../imagenes/dataset.txt",delimiter=",")
    data = dataset[:,:2]
    labels = dataset[:,2]

    data_marca = data[labels == 0]
    data_fondo = data[labels == 1]
    data_linea = data[labels == 2]


# calcular los centroides <--> entrenar el clasificador
c_marca = np.mean(data_marca,0)
c_fondo = np.mean(data_fondo,0)
c_linea = np.mean(data_linea,0)


# Dibujar los datos obtenidos

plt.figure()
plt.plot(data_marca[:,1],data_marca[:,0],"r.")
plt.plot(data_fondo[:,1],data_fondo[:,0],"g.")
plt.plot(data_linea[:,1],data_linea[:,0],"b.")

plt.show()


# Generar codigo python para los parametros del clasificador
cod = """import numpy as np
c_marca = np.array([{0}])
c_fondo = np.array([{1}])
c_linea = np.array([{2}])
"""
cod_marca = ""
cod_fondo = ""
cod_linea = ""
for i in range(len(c_marca)):
    if i != 0:
        cod_marca +=', '
        cod_fondo +=', '
        cod_linea +=', '
    cod_marca += str(c_marca[i])
    cod_fondo += str(c_fondo[i])
    cod_linea += str(c_linea[i])

        # Escribir los datos en los parámetros del clasificador
with open("parametros.py",'w') as f:
    f.write(cod.format(cod_marca,cod_fondo,cod_linea))
