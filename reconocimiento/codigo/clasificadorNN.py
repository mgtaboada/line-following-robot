import numpy as np
class clasificadorNN:
    def __init__(self,labels):
        """Constructor de la clase
        labels: lista de etiquetas de esta clase; Obligatorio, sin etiquetas no tiene sentido"""
        self.labels=np.unique(labels)[np.newaxis,:]
        self.W = None #El vector con los centroides y ZiT * Zi * -1/2
        pass

    def fit(self,X,y):
        """Entrena el clasificador
        X: matriz numpy cada fila es un dato, cada columna una medida
        y: vector de etiquetas, tantos elementos como filas en X
        retorna objeto clasificador"""
        
        #Por cada clase que tenemos se calcula el centroide, calculando la media de cada clase. Y se transpone
        centroids = np.apply_along_axis(lambda label: X[y == label[0]].mean(axis=0),0,self.labels).T
        
        #Para cada centroide se calcula ZiT * Zi * -1/2 y se pone en la primera columna
        self.W = np.c_[np.apply_along_axis(lambda zi: zi.dot(zi)/-2,1,centroids),centroids]
        
        return self

    def predict(self,X):
        """Estima el grado de pertenencia de cada dato a las clases 
        X: matriz numpy cada fila es un dato, cada columna una medida
        retorna una matriz, cada fila almacena los valores pertenencia"""
        
        #añade un 1 a cada fila, al principio.
        X = np.insert(X,0,1,axis=1)
        
        #Hace la multiplicacion de matrices por cada fila (vector) de X con W.
        #Mejora tras la retroalimentacion aportada por el profesor:
        belonging = X.dot(self.W.T)
        
        #belonging = np.apply_along_axis(lambda x: self.W.dot(x),1,X) #Antiguo codigo
        
        
       
        return belonging
    
    def predLabel(self,X):
        """Estima la etiqueta de cada dato
        X: matriz numpy cada fila es un dato, cada columna una medida
        retorna un vector con las etiquetas de cada dato"""
        
        #Calcula el grado de pertenencia
        belonging = self.predict(X)
       
        
        #Almacena la etiqueta predicha para cada X
        predictedLabel = self.labels[:,belonging.argmax(axis=1)]
        
        
        return predictedLabel
