#coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

data_pelota = []
data_fondo = []
data_mano = []
data_ropa = []

# Generar dataset si no existe
if not os.path.isfile("../imagenes/dataset.txt"):
    for i in range(4):
        imname = "../imagenes/imagen_{}.png".format(i)
        markname = "../imagenes/imagen_{}_m.png".format(i)

        im = cv2.imread(imname)
        mark = cv2.imread(markname)

        imrgb = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
        im =imrgb


        markrgb = cv2.cvtColor(mark,cv2.COLOR_BGR2RGB)
        mark = markrgb
        suma = np.sum(im,2)
        suma[suma == 0]=1
        imn = np.rollaxis((np.rollaxis(im,2)+0.0)/suma,0,3)[:,:,:2]

        data_fondo+=imn[np.where(np.all(np.equal(mark,(0,255,0)),2))].tolist()
        data_pelota+=imn[np.where(np.all(np.equal(mark,(0,0,255)),2))].tolist()
        data_mano+=imn[np.where(np.all(np.equal(mark,(255,255,0)),2))].tolist()
        data_ropa+=imn[np.where(np.all(np.equal(mark,(255,0,0)),2))].tolist()


        with open("../imagenes/dataset.txt","w") as f:
            for r,g in data_fondo:
                f.write("{},{},0\n".format(r,g))
            for r,g in data_pelota:
                f.write("{},{},1\n".format(r,g))
            for r,g in data_mano:
                f.write("{},{},2\n".format(r,g))
            for r,g in data_ropa:
                f.write("{},{},3\n".format(r,g))

    data_fondo_prev = np.array(data_fondo)
    data_pelota_prev = np.array(data_pelota)
    data_mano_prev = np.array(data_mano)
    data_ropa_prev = np.array(data_ropa)

    data_fondo = data_fondo_prev
    data_pelota = data_pelota_prev
    data_mano = data_mano_prev
    data_ropa = data_ropa_prev

# Si ya existía, leerlo
else:
    dataset = np.genfromtxt("../imagenes/dataset.txt",delimiter=",")
    data = dataset[:,:2]
    labels = dataset[:,2]

    data_fondo = data[labels == 0]
    data_pelota = data[labels == 1]
    data_mano = data[labels == 2]
    data_ropa = data[labels == 3]
# calcular los centroides <--> entrenar el clasificador
c_fondo = np.mean(data_fondo,0)
c_pelota = np.mean(data_pelota,0)
c_mano = np.mean(data_mano,0)
c_ropa = np.mean(data_ropa,0)

# Dibujar los datos obtenidos

plt.figure()


plt.plot(data_pelota[:,1],data_pelota[:,0],"b.")
plt.plot(data_mano[:,1],data_mano[:,0],"m.")
plt.plot(data_ropa[:,1],data_ropa[:,0],"r.")
plt.plot(data_fondo[:,1],data_fondo[:,0],"g.")
plt.show()


# Generar codigo python para los parametros del clasificador
cod = """import numpy as np
c_fondo = np.array([{0}])
c_pelota = np.array([{1}])
c_mano = np.array([{2}])
c_ropa = np.array([{3}])
"""
cod_fondo = ""
cod_pelota = ""
cod_mano = ""
cod_ropa = ""
for i in range(len(c_fondo)):
    if i != 0:
        cod_fondo +=', '
        cod_pelota +=', '
        cod_ropa +=', '
        cod_mano +=', '
    cod_fondo += str(c_fondo[i])
    cod_pelota += str(c_pelota[i])
    cod_ropa += str(c_ropa[i])
    cod_mano += str(c_mano[i])
        # Escribir los datos en los parámetros del clasificador
with open("parametros.py",'w') as f:
    f.write(cod.format(cod_fondo,cod_pelota,cod_mano,cod_ropa))
