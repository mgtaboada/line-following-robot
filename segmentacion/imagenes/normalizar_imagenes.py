import cv2
import numpy as np

for i in range(10):
    filename = "linea{}.png".format(i)
    img_pre = cv2.imread(filename)
    norm=np.rollaxis((np.rollaxis(img_pre,2)+0.0)/np.sum(img_pre,2),0,3)[:,:,:]
    cv2.imwrite("norm_linea{}.png".format(i),norm)
