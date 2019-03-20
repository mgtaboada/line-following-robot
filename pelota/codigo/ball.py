#coding: utf-8
import numpy as np
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
