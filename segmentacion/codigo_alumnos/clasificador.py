import numpy as np
import cv2
#from scipy.spatial import distance_matrix
def distance_matrix (x,y):
    return -2*np.dot (x,y.T) + np.sum (y**2,axis=1)# + np.sum (x**2,axis=1)
class Clasificador:
    def __init__(self):
        self.c_marca = np.array([0.45112993, 0.2954761])
        self.c_fondo = np.array([ 0.33413146,  0.36349363])
        self.c_linea = np.array([ 0.21104495,  0.34013866])
        self.centroids = np.array([self.c_marca,self.c_fondo,self.c_linea])
        self.W = np.c_[np.apply_along_axis(lambda zi: zi.dot(zi)/-2,1,self.centroids),self.centroids]
    def classif_px (self,px):
        x = np.insert (px,0,1,axis=1)
        return np.argmax(x.dot (self.centroids.T))

    def classif_img(self,bgr):
        '''Devuelve una matriz con los valores 0,1 o 2 para cada pixel (marca,fondo,linea)'''
        imrgb = cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
        im =imrgb
        imn = np.rollaxis((np.rollaxis(im,2)+0.0)/np.sum(im,2),0,3)[:,:,:2]
        x,y,z = imn.shape
        imr = np.reshape (imn,(-1,z))
        # cats = np.apply_along_axis (self.classif_px,1,imr)
        d = self.distance_matrix (imr,self.centroids)
        #cats = np.argmin (d,axis=1)
        cats = np.argmax(d,axis=1)
        res = np.reshape (cats,(x,y))
        return res
    def distance_matrix (self,x,y):
        #return -2*np.dot (x,y.T) + np.sum (y**2,axis=1)# + np.sum (x**2,axis=1)
        X = np.insert(x,0,1,axis=1)
        return X.dot(self.W.T)

