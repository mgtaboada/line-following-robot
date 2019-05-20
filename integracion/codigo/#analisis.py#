#coding: utf-8
import cv2
import numpy as np
import time

LINEA_RECTA = 0
CURVA_DERECHA = 1
CURVA_IZQUIERDA = 2
DOS_SALIDAS = 3
TRES_SALIDAS= 4


def in_border (img):
    #np.savetxt ("img.txt",img)
    #st = time.time()
    mask = np.zeros (img.shape)
    mask [1:-1,1:-1]=1
    mask = mask.astype (bool)
    img [mask] = 0
    #print(time.time()-st)
    return np.array (np.where (img == 1))

def chull_area(chull):
    x = chull[:,0,0]
    y = chull[:,0,1]
    return carea(x,y)

def carea(x,y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def en_borde(cont,shape):
    x,y = shape
    return np.any(cont[:,0,:] == 0) or np.any(cont[:,0,0] == y-1) or np.any(cont[:,0,1] == x-1)

def limpiar_img(img):
    """ Eliminar aquellos pixeles que estaban mal segmentados como linea en la imagen

   img: imagen binaria en la que los 1 son pixeles de linea y los 0 de otra cosa
    devuelve otra imagen binaria en la que solo aparece el contorno más grande
    """
    res = np.zeros(img.shape)
    _,conts, hier = cv2.findContours((img== 1).astype(np.uint8)*255,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    biggest = None
    area = 30*30 
    #print("Min area: {}".format(area))
    for cont in conts:
        new_area = cv2.contourArea(cont)
        if new_area > area:
            biggest = cont
            area = new_area

    #res[biggest] = 1
    if biggest is not None: # and not en_borde(biggest,img.shape):
            cv2.drawContours(res, [biggest], 0, (1), thickness=cv2.FILLED)
    return res.astype(np.uint8)

def encontrar_icono(img):
    """ Eliminar aquellos pixeles que estaban mal segmentados como linea en la imagen

   img: imagen binaria en la que los 1 son pixeles de linea y los 0 de otra cosa
    devuelve otra imagen binaria en la que solo aparece el contorno más grande
    """
    res = np.zeros(img.shape)
    _,conts, hier = cv2.findContours((img== 1).astype(np.uint8)*255,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    biggest = None
    area = 10 
    for cont in conts:
        new_area = cv2.contourArea(cont)
        if new_area > area:
            biggest = cont
            area = new_area

    #res[biggest] = 1
    if biggest is not None and not en_borde(biggest,img.shape):
            cv2.drawContours(res, [biggest], 0, (1), thickness=cv2.FILLED)
    return res.astype(np.uint8)




def tipo_linea(img):
    """
    img: imagen binaria en la que los 1 son pixeles de linea y los 0 de otra cosa

    """
    thres = 0.2 # Porcentaje de cierre convexo que no es linea para considerar linea recta
    #img = limpiar_img(img)

    # Contamos los contornos de no linea: Si hay más de dos, hay más de una salida


    _,conts, hier = cv2.findContours((img == 0).astype(np.uint8)*255,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    n_conts = 0
    for cont in conts:
        if cv2.contourArea (cont) > 100: # 10*10
            n_conts +=1
    if n_conts == 4:
        return TRES_SALIDAS
    if n_conts == 3:
        return DOS_SALIDAS
    _, conts, hier = cv2.findContours((img == 1).astype(np.uint8)*255,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)

    if len (conts) == 0:
        print("No conts")
        return None
    print("Algun cont")
    #suponemos que la linea es el contorno con la mayor area
    line =conts[0]
    area = cv2.contourArea(line)
    for cont in conts:
        new_area = cv2.contourArea(cont)
        if new_area > area:
            line = cont
            area = new_area

    chull = cv2.convexHull(line)


    # Si el contorno y el chull tienen más o menos (+- 5%)la misma área,podemos considerar que es una linea recta
    charea = chull_area(chull)
    if area > charea * (1-thres) and area < charea*(1+thres):
        return LINEA_RECTA
    # La linea es curva
    # Para averiguar hacia donde gira, comparamos la cantidad de pixeles de linea en las dos mitades
    # de la imagen
    medio = int(img.shape[1]/2)
    derecha = img[:,medio:]
    izquierda = img[:,:medio]
    if np.size(derecha[derecha == 1]) > np.size(izquierda[izquierda == 1]):
        # hay mas linea a la derecha que a la izquierda -> gira a la derecha
        return CURVA_DERECHA
    return CURVA_IZQUIERDA

def direccion_flecha(im):
    '''Devuelve la direccion de la flecha
    im: imagen binaria donde los 1's son pixeles de la flecha y los 0's de otra cosa

    devuelve: p1, p2: puntos por los que pasa la flecha. En sentido p1->p2
              m: pendiente de la flecha, para no tener que volver a calcularla
              c: ordenada en el origen de la recta que contiene a la  flecha, para no tener que volver a calcularla
'''
    bi = encontrar_icono(im)
    _,conts,hier = cv2.findContours(bi*255,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    #conts,hier = cv2.findContours(bi*255,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    if len(conts) == 0:
       return None,None,None,None
    arrow =conts[0]
    area = cv2.contourArea(arrow)
    for cont in conts:
        new_area = cv2.contourArea(cont)
        if new_area > area:
            arrow = cont
            area = new_area


    [vx,vy,x,y] = cv2.fitLine(arrow,cv2.DIST_L2,0,0.01,0.01)
    h,_,w = arrow.shape
    #points = np.array(np.where(bi==1))
    points = np.reshape(arrow,(h,w)).T

    m = vy/vx
    mm = -1/m if m != 0 else 0
    c = y - (m*x)
    cc = y-((mm*x))

    p1 = (x,y)
    p2 = (x+30, m*(x+30)+c) #positive direction
    p3 = (x+30, mm*(x+30)+cc)
    if m < 0:
    #     m=-m
         aux = p1
         p1 = p2
         p2 = aux

    pos = points.T[points[1] > cc+(mm*points[0])]
    neg = points.T[points[1] <= cc+(mm*points[0])]


    #print("M: {}\nC: {}\nMM:{}\nCC: {}\np1: {}".format(m[0],c,mm,cc,np.array(p1).T))
    if carea(pos[:,0],pos[:,1]) > carea(neg[:,0],neg[:,1]):
        #print('{}>{}'.format(pos,neg))
        return p1,p2,m,c
    else:
        #print('{}<{}'.format(pos,neg))
        return p2,p1,m,c


def entrada_salida (img,anterior_entrada=None):
    linea = (img == 2).astype (np.uint8)
    flecha = (img == 0).astype (np.uint8)
    h,w = linea.shape
    # Suponemos que está cerca del centro
    if anterior_entrada is None:
        anterior_entrada = np.array ((h,w/2))
    bordes = in_border (linea).T
    distancias = np.sum((bordes - anterior_entrada)**2, axis=1)
    if np.size (distancias)> 0:
        cercano = np.argmin (distancias)
        entrada = bordes [cercano]
    else:
        entrada = (0,0)
    salida = (0,0)

    # buscar la salida
    tipo =tipo_linea (linea)
    if tipo is None:
        #print("Tipo None")
        return None,None
    if tipo < DOS_SALIDAS:      # es una sola linea
        # La salida tiene que estar separada de la entrada
        print("Una linea")
        lejano = np.argmax (distancias)
        salida = bordes [lejano]
    else: # debería haber una flecha
        print("Cruce")
        if np.any (flecha==1):
            p1,p2,m,c = direccion_flecha (flecha)
            if p1 is None:
                return None,None
            xo = 0
            yo = 0
            if p1 [0] < p2 [0]:
                # sentido positivo en las x -> buscamos el final
                xo = w
            elif p1 [0] == p2 [0]:
                #vertical, usamos la misma x
                xo = p1 [0]
            if p1 [1] < p2 [1]:
                # sentido positivo en las y -> buscamos el final
                yo=h
            elif p1 [1] == p2 [1]:
                #horizontal, usamos la misma y
                yo = p1 [1]
            #calculamos por donde sale la recta
            y = m*xo + c
            if y >= h or y < 0:
                xo = (yo-c)/m
            else:
                yo = y
            #encontramos el punto mas cercano
            distancias = np.sum((bordes - (yo,xo))**2, axis=1)
            if np.size (distancias)> 0:
                cercano = np.argmin (distancias)
                salida = bordes [cercano]
    return (entrada [1],entrada [0]),(salida [1],salida [0])
