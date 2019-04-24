import numpy as np
import pickle #Cargar los datos
import clasificadorNN as c
import clasificadorEstadistico as m
import sklearn.neighbors as n


with open("datasetFiguras", 'rb') as f:
    dataset = pickle.load(f)
with open("datasetEtiquetasFiguras", 'rb') as f:
    dataset_etiquetas = pickle.load(f)

dataset = np.array(dataset)
dataset_etiquetas = np.array(dataset_etiquetas)

n1n= n.KNeighborsClassifier(1)
skmaha = n.KNeighborsClassifier(1,algorithm='brute',metric='mahalanobis',metric_params={'V':np.cov(dataset)})

n1n.fit(dataset,dataset_etiquetas)
skmaha.fit(dataset,dataset_etiquetas)

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

def leave_one_out_sk(classif,dataset,dataset_etiquetas):
    err = 0
    for k in range(len(dataset)):
        if k == 0:
            test = dataset[0]
            testl = dataset_etiquetas[0]
            train = dataset[1:]
            trainl = dataset_etiquetas[1:]
        elif k == len(dataset)-1:
            test = dataset[-1]
            testl = dataset_etiquetas[-1]
            train = dataset[:-1]
            trainl = dataset_etiquetas[:-1]

        else:
            test = dataset[k]
            testl = dataset_etiquetas[k]
            train = np.concatenate((dataset[:k],dataset[k+1:]))
            trainl = np.concatenate((dataset_etiquetas[:k],dataset_etiquetas[k+1:]))
        classif.fit(train,trainl)
        pred = classif.predict([test])
       
        if pred[0] == testl:
            err +=1
    print("Aciertos: {} sobre un total de {} ({:.02f}%)".format(err,len(dataset),(float(err)/len(dataset))*100))

def leave_one_out_our(classif,dataset,dataset_etiquetas):
    err = 0
    for k in range(len(dataset)):
        if k == 0:
            test = dataset[0]
            testl = dataset_etiquetas[0]
            train = dataset[1:]
            trainl = dataset_etiquetas[1:]
        elif k == len(dataset)-1:
            test = dataset[-1]
            testl = dataset_etiquetas[-1]
            train = dataset[:-1]
            trainl = dataset_etiquetas[:-1]

        else:
            test = dataset[k]
            testl = dataset_etiquetas[k]
            train = np.concatenate((dataset[:k],dataset[k:]))
            trainl = np.concatenate((dataset_etiquetas[:k],dataset_etiquetas[k:]))
        classif.fit(train,trainl)
        pred = classif.predLabel([test])
       
        if pred[0] == testl:
            err +=1
    print("Aciertos: {} sobre un total de {} ({:.02f}%)".format(err,len(dataset),(float(err)/len(dataset))*100))    

#evalClassifier(nn,dataset,dataset_etiquetas,dataset,dataset_etiquetas)
#evalClassifier(maha,dataset,dataset_etiquetas,dataset,dataset_etiquetas)

print("""###########################
# Sklearn nearest neighbor #
###########################""")
leave_one_out_sk(n1n,dataset,dataset_etiquetas)
print("""
################################
# Sklearn mahalanobis distance #
################################""")
leave_one_out_sk(skmaha,dataset,dataset_etiquetas)

print("""
########################
# Our nearest neighbor #
########################""")
leave_one_out_our(nn,dataset,dataset_etiquetas)
print("""
############################
# Our mahalanobis distance #
############################""")
leave_one_out_our(maha,dataset,dataset_etiquetas)

