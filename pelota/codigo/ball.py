#coding: utf-8
import numpy as np
import clasificador as c
# Tamaño de la pelota (conocido)
KNOWN_WIDTH = 7.0
KNOWN_HEIGTH = 7.0

def ball_square(im):
    """Devuelve las esquinas del rectángulo que inscribe a la pelota y su centro"""
    # Encontrar las coordenadas de la imagen que pertenezcan a la categoría 1 (pelota)
    ball = np.array(np.where(im==1))
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
    ul,dr,_ = ball_square(im)
    apparent_width = abs(ul[0]-dr[0])
    apparent_heigth = abs(ul[1]-dr[1])


    focal_width = float(apparent_width * known_distance) / KNOWN_WIDTH
    focal_heigth = float(apparent_heigth * known_distance) / KNOWN_HEIGTH

    return np.mean([focal_heigth, focal_width])

def distance_to_camera(dr,ul,focal):

    apparent_width = abs(ul[0]-dr[0])
    apparent_heigth = abs(ul[1]-dr[1])

    distance_width =(KNOWN_WIDTH * focal) / apparent_width
    distance_heigth =(KNOWN_HEIGTH * focal) / apparent_heigth

    return np.mean([distance_width, distance_heigth])