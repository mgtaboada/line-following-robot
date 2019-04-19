import numpy as np
import pickle #Cargar los datos
import clasificadorNN as c
import clasificadorEstadistico as m


with open("datasetFiguras", 'rb') as f:
    dataset = pickle.load(f)
with open("datasetEtiquetasFiguras", 'rb') as f:
    dataset_etiquetas = pickle.load(f)

dataset = np.array(dataset)
dataset_etiquetas = np.array(dataset_etiquetas)
nn = c.clasificadorNN(dataset_etiquetas)
nn.fit(dataset,dataset_etiquetas)

maha = m.clasificadorEstadistico()

#print(nn.predLabel(dataset))
def evalClassifier(classif, Xtrain, ytrain, Xtest, ytest,printOutput=True):
    """ Funcion que evalua un clasificador.
    classif: clasificador que se desea evaluar
    Xtrain: matriz numpy que almacena los datos de entrenamiento
            cada fila es un dato, cada columna una medida
    ytrain: vector que almacena las etiquetas de entrenamiento, 
            tiene tantos elementos como filas hay en X
    Xtest: matriz numpy que almacena los datos de test
           cada fila es un dato, cada columna una medida
    ytest: vector que almacena las etiquetas de test, 
           tiene tantos elementos como filas hay en X
    retorna el numero de errores total y desglosado por clase"""
    # Entreno clasificador con datos de train y clasifico con datos de test
    ypred = classif.fit(Xtrain,ytrain).predLabel(Xtest) 
    # Creo un diccionario de los indices
    
    #Se aplanan las matrices. Yo los utilizo en 2D
    ypred = ypred.flatten()
    labels = classif.labels.flatten()
    
    dicEtq = dict([(label,index) for index,label in enumerate(labels)])
    yReIdx = [dicEtq[x] for x in list(ytest)]
    yPrIdx = [dicEtq[x] for x in list(ypred)]
    errors = np.zeros((len(labels),len(labels)),dtype=np.uint32)
    for predict,real in zip(yPrIdx,yReIdx):
        errors[predict,real] += 1
    if(printOutput):
        print("Errores ",np.sum(errors)-np.trace(errors)) # total number of errors
        print("Matriz de confusion \n",errors) # confusion matrix
        print("\n")
    return errors

evalClassifier(nn,dataset,dataset_etiquetas,dataset,dataset_etiquetas)
evalClassifier(maha,dataset,dataset_etiquetas,dataset,dataset_etiquetas)

