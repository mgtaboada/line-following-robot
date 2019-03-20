#coding: utf-8
import numpy as np
import clasificador as c
# Tamaño de la pelota (conocido)
KNOWN_WIDTH = 7.0
KNOWN_HEIGTH = 7.0

def ball_square(im):
    """Devuelve las esquinas del rectángulo que inscribe a la pelota y su centro"""
    # Encontrar las coordenadas de la imagen que pertenezcan a la categoría 1 (pelota)
    dr = (0,0)
    ul = (0,0)
    center = (0,0)

    ball = np.array(np.where(im==1))
    if not ball.any():
        return ul,dr,center
    # El centro es la media de las coordenadas
    center = np.uint(np.mean(ball,1)).tolist()
    # la esquina superior izquierda son las coordenadas de menor valor, la inferior derecha
    # son las coordenadas mayores
    dr = np.max(ball,1).tolist()
    ul = np.min(ball,1).tolist()

    # Por alguna razón, están en orden inverso
    center.reverse()
    ul.reverse()
    dr.reverse()
    return tuple(ul), tuple(dr), tuple(center)

def get_focal_length(im,known_distance):
    """Calcula y devuelve la distancia focal de la camara"""
    # obtiene las esquinas del rectángulo que modela la pelota
    ul,dr,_ = ball_square(im)

    # calcula la altura y anchura aparente de la pelota en la imagen
    apparent_width = abs(ul[0]-dr[0])
    apparent_heigth = abs(ul[1]-dr[1])

    # calcula dos posibilidades de la focal, ya que conocemos dos medidas
    focal_width = float(apparent_width * known_distance) / KNOWN_WIDTH
    focal_heigth = float(apparent_heigth * known_distance) / KNOWN_HEIGTH

    # devuelve la media por redundancia
    return np.mean([focal_heigth, focal_width])

def distance_to_camera(dr,ul,focal):
    # calcula la altura y anchura aparente de la pelota en la imagen
    apparent_width = abs(ul[0]-dr[0])
    apparent_heigth = abs(ul[1]-dr[1])

    # calcula dos posibilidades de la focal, ya que conocemos dos medidas
    distance_width =(KNOWN_WIDTH * focal) / apparent_width
    distance_heigth =(KNOWN_HEIGTH * focal) / apparent_heigth

    #devuelve la media por redundancia
    return np.mean([distance_width, distance_heigth])
