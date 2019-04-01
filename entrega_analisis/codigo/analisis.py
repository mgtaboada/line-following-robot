#coding: utf-8
import cv2
import numpy as np

LINEA_RECTA = 0
CURVA_DERECHA = 1
CURVA_IZQUIERDA = 2
DOS_SALIDAS = 3
TRES_SALIDAS= 4

def chull_area(chull):
    x = chull[:,0,0]
    y = chull[:,0,1]
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def en_borde(cont,shape):
    y,x = shape
    return np.any(cont == 0) or np.any(cont[:,0,0] == y-1) or np.any(cont[:,0,1] == x-1)

def limpiar_img(img):
    """ Eliminar aquellos pixeles que estaban mal segmentados como linea en la imagen

    img: imagen binaria en la que los 1 son pixeles de linea y los 0 de otra cosa
    devuelve otra imagen
    """

    paleta = np.array([[0,0,0],[255,255,255],[0,0,255],[0,0,0]],dtype=np.uint8)
    median = cv2.medianBlur(img,11)
    return (median == 1).astype(np.uint8)
    '''
    umbral_area = 0.1 # Porcentaje minimo que tiene que ocupar el contorno para considerarse linea
    _, conts, hier = cv2.findContours((img== 1).astype(np.uint8)*255,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    for cont in conts:
        if cv2.contourArea(cont) > img.shape[0]*img.shape[1]*umbral_area:
            img[cont[:,0,1],cont[:,0,0]] = 0

    _, conts, hier = cv2.findContours((img != 1).astype(np.uint8)*255,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)

    # Si no tiene puntos en el borde de la imagen, no es una linea
    v,h = img.shape
    for cont in conts:
        if cv2.contourArea(cont) > img.shape[0]*img.shape[1]*umbral_area:
        #if not en_borde(cont,img.shape):
            img[cont[:,0,1],cont[:,0,0]] = 1
    return img
'''


def tipo_linea(img,orig):
    """
    img: imagen binaria en la que los 1 son pixeles de linea y los 0 de otra cosa

    """
    thres = 0.2 # Porcentaje de cierre convexo que no es linea para considerar linea recta
    img = limpiar_img(img)

    paleta = np.array([[0,0,0],[255,255,255],[0,0,255],[0,0,0]],dtype=np.uint8)
    cv2.imshow("Imagen limpiada",cv2.cvtColor(paleta[img],cv2.COLOR_RGB2BGR))

    # Contamos los contornos de no linea: Si hay más de dos, hay más de una salida

    _, conts, hier = cv2.findContours((img == 0).astype(np.uint8)*255,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(orig, conts, -1, (0,255,0), 3)
    if len(conts) == 4:
        return TRES_SALIDAS,orig
    if len(conts) == 3:
        return DOS_SALIDAS,orig
    _, conts, hier = cv2.findContours((img == 1).astype(np.uint8)*255,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)

    if len(conts) == 0:
        return None,orig
    #suponemos que la linea es el contorno con la mayor area
    line =conts[0]
    area = cv2.contourArea(line)
    for cont in conts:
        new_area = cv2.contourArea(cont)
        if new_area > area:
            line = cont
            area = new_area

    chull = cv2.convexHull(line)

    cv2.drawContours(orig,[chull],0,(0,0,255),1)
    # Si el contorno y el chull tienen más o menos (+- 5%)la misma área,podemos considerar que es una linea recta
    charea = chull_area(chull)
    if area > charea * (1-thres) and area < charea*(1+thres):
        return LINEA_RECTA,orig
    # La linea es curva
    # Para averiguar hacia donde gira, comparamos la cantidad de pixeles de linea en las dos mitades
    # de la imagen
    medio = int(img.shape[1]/2)
    derecha = img[:,medio:]
    izquierda = img[:,:medio]
    if np.size(derecha[derecha == 1]) > np.size(izquierda[izquierda == 1]):
        # hay mas linea a la derecha que a la izquierda -> gira a la derecha
        return CURVA_DERECHA,orig
    return CURVA_IZQUIERDA,orig

def direccion_flecha(img):
    """
    img: imagen binaria en la que los 1 son pixeles de flecha y los 0 de otra cosa

    """
    img, conts, hier = cv2.findContours(img[img == 1],cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)

    #suponemos que la flecha es el contorno con la mayor area
    flecha = None
    area = 0
    for cont in conts:
        new_area = cv2.contourArea(cont)
        if new_area > area:
            flecha = cont
            area = new_area

    center,shape,angle = cv2.minAreaRect(flecha)
