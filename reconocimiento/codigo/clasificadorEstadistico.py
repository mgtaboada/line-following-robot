import numpy as np

class clasificadorEstadistico:
    def __init__(self,labels=[],grIg=0.0,grDiag=0.0):
        """Constructor de la clase
        labels: lista de etiquetas de esta clase
        priors: class priors
        grIg: lambda-regularized parameter
        grDiag: gamma-regularized parameter"""
        self.labels=np.unique(labels)[np.newaxis,:]
        #self.priors=priors
        self.grIg=grIg
        self.grDiag=grDiag
        pass

    def __lndetcov(self,cvmat):
        """ Computes the ln of the determinant of matrix cvmat
        cvmat: covariance matrix"""
        # compute the sum of log eigenvalues of cvmat
        return sum(np.log(np.real(np.linalg.eigvals(cvmat))))
    
    def fit(self,X,y):
        """Entrena el clasificador
        X: matriz numpy cada fila es un dato, cada columna una medida
        y: vector de etiquetas, tantos elementos como filas en X
        retorna objeto clasificador"""
       
        if(len(self.labels[0]) == 0):
            self.labels=np.unique(y)[np.newaxis,:]
        #Por cada clase que tenemos se calcula el centroide, calculando la media de cada clase. Y se transpone
        self.means = np.apply_along_axis(lambda label: X[y == label[0]].mean(axis=0),0,self.labels).T
        #Se calculan las matrices de covariancia
        self.covs = np.apply_along_axis(lambda label: np.cov(X[y == label[0]],rowvar=False,ddof=0),0,self.labels).T
        #Se calculan las cardinalidades de cada clase
        self.cardinalidades = np.apply_along_axis(lambda label: np.count_nonzero(y[y==label[0]]),0,self.labels)[np.newaxis,:][np.newaxis,:].T
        #Se calcula la cardinalidad total del conjunto
        self.total_cardinalidad = self.cardinalidades.sum()
        
        #Se calcula la matriz de covariancia conjunta
        tmp = self.cardinalidades / self.total_cardinalidad
        self.cov_total = (self.covs * tmp).sum(axis=0)
        self.cov_total = self.cov_total * (self.total_cardinalidad/(self.total_cardinalidad-len(self.labels)))
        
        #Se calcula la matriz de covarianza de cada clase con el hiperparametro lambda
        numerador = (1-self.grIg)*self.cardinalidades*self.covs+self.grIg*self.total_cardinalidad*self.cov_total
        denominador = (1-self.grIg)*self.cardinalidades+self.grIg*self.total_cardinalidad
        self.covs_lambda = numerador / denominador
        
        #Se calcula ai
        traza = (np.identity(self.covs_lambda.shape[1]) * self.covs_lambda).sum(axis=1).sum(axis=1)[np.newaxis,:]
        a = traza/self.covs_lambda.shape[1]
        #Se calcula ai*I
        aI = np.apply_along_axis(lambda a_i: a_i*np.identity(self.covs_lambda.shape[1]),0,a).T

        #Se calcula la matriz de covarianza de cada clase con el hiperparametro lambda y gamma
        self.covs_l_g = (1-self.grDiag)*self.covs_lambda + self.grDiag*aI
        
        #Calculo de las probabilidades a priori y se hace el logaritmo para usarlo mas adelante
        self.priors = np.log(np.apply_along_axis(lambda label: X[y == label[0]].shape[0],0,self.labels)/X.shape[0]) 
        self.priors2 = np.apply_along_axis(lambda label: X[y == label[0]].shape[0],0,self.labels)/X.shape[0]
        
        return self

    def predict(self,X):
        """Estima el grado de pertenencia de cada dato a las clases 
        X: matriz numpy cada fila es un dato, cada columna una medida
        retorna una matriz, cada fila almacena los valores pertenencia""" 
        
        #Calcula el grado de pertenencia. Por cada clase con todos los objetos.
        i = 0 #Indice por la clase que toca
        def di(w):
            
            nonlocal i
            #p1 => primera parte de la ecuacion -1/2 * ln(det(E_ag))
            p1 = self.__lndetcov(self.covs_l_g[i])/-2
            
            #P3 => calcula el log(priori)
            #p3 = np.log(self.priors[i])
            p3 = self.priors[i]
            p13 = np.sum([p1,p3])
            
            #g => (X-mu_i)
            g = X-self.means[i]
            
            #p2 calcula (X-mu_i)^T * E_ag^-1 * (X-mu_i)
            p2 = g.dot(np.linalg.inv(self.covs_l_g[i]))
            p2 = (p2*g).sum(1) #Esta ultima parte se hace asi ya que dot no es capaz de aplicarlo adecuadamente 
            p2 = p2/-2
            i = i + 1
            
            #Se suma todo
            
            return p1+p2+p3
            
        
        belonging  = np.apply_along_axis(di,0,self.labels)
        
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
