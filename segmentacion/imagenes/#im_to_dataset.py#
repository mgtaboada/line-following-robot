import numpy as np
import cv2

data_marca = []
data_fondo = []
data_linea = []

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
