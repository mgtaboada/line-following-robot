import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D as ax3d

dataset = np.genfromtxt("dataset.txt",delimiter=",")
data = dataset[:,:2]
labels = dataset[:,2]

data_marca = data[labels == 0]
data_fondo = data[labels == 1]
data_linea = data[labels == 2]

c_marca = np.mean(data_marca,0)
c_fondo = np.mean(data_fondo,0)
c_linea = np.mean(data_linea,0)


# fig = plt.figure()
# ax = fig.gca(projection='3d')

# ax.plot(data_marca[:,2],data_marca[:,1],data_marca[:,0],'r.',label='marca')
# ax.plot(data_fondo[:,2],data_fondo[:,1],data_fondo[:,0],'g.',label='fondo')
# ax.plot(data_linea[:,2],data_linea[:,1],data_linea[:,0],'b.',label='linea')
# ax.set_xlabel('R Label')
# ax.set_ylabel('G Label')
# ax.set_zlabel('B Label')
# ax.legend()
# plt.title('Espacio RGB')

print("Marca: {}\nFondo: {}\nLinea: {}".format(c_marca,c_fondo,c_linea))
plt.figure()
plt.plot(data_marca[:,1],data_marca[:,0],"r.")
plt.plot(data_fondo[:,1],data_fondo[:,0],"g.")
plt.plot(data_linea[:,1],data_linea[:,0],"b.")

plt.show()
