import  cv2
import numpy as np

#import clasif as cl
import clasificador as c
import analisis as a

def test_image(path):

    cl = c.Clasificador()

    img = cv2.imread(path)
    h = img.shape[0]
    print(path)
    cats =cl.classif_img(img)

    paleta = np.array([[0,0,0],[255,255,255],[0,0,255],[0,0,0]],dtype=np.uint8)


    f1,f2= a.entrada_salida (cats)
    #cv2.arrowedLine(img ,f1,f2,(0,0,255),4)
    print(f1)
    cv2.circle(img,tuple(f1),3,(0,255,0))
    cv2.circle(img,tuple(f2),3,(0,0,255))

    cv2.imshow(path,img)


base = "../imagenes/"
paths = ["linea2.png"]


for path in paths:
    test_image(base+path)

cv2.waitKey()
