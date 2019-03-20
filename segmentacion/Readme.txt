    ###########################################
    # Segmentación de vídeo basada en colores #
    ###########################################

Autores:Daniel Fernández Pérez y  Miguel García-Mauriño Taboada

Para la segmentación del vídeo utilizamos un clasificador basado
en la distancia euclídea, que clasifica cada pixel en una de las
siguiente categorías en función de su color:

  - Línea
  - Marca
  - Fondo

Para lograrlo, hemos hecho varias capturas de dos de los vídeos
que se nos daban y hemos utilizado un editor de imágenes para
colorear de forma diferenciada los pixeles según la categoría
a la que pertenecen.

A continuación, la parte de entrenamiento de nuestro programa
(codigo/prac_ent.py) lee tanto las imágenes originales como
las marcadas y genera un dataset con los valores normalizados
de cada pixel y la categoría a la que pertenece. Calcula la
media y crea una representación gráfica de los datos. Gracias
a la gran cantidad de datos de la que disponemos, se observa que
las categorías no tienen fronteras claramente definidas, por lo
que en algunos casos, poco frecuentes en comparación con el
resto, el clasificador fallará.

Tras calcular las medias, el programa genera el fichero
codigo/parametros.py en el vuelca los centroides para las tres
categorías de forma que el fichero codigo/clasificador.py los
puede recuperar para utilizarlos en la clasificación.

Finalmente, al ejecutar el fichero codigo/prac_run.py se lee
uno de los vídeos de la carpeta videos y se muestra junto a
la segmentación realizada. Esta última se hace utilizando el
clasificador del que ya hemos hablado, y segmentando una de
cada veinticinco imágenes.
