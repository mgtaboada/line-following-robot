import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import numpy as np
import sys

style.use('fivethirtyeight')
fig = plt.figure()
ax = fig.add_subplot(1,1,1)

fin = False

def animate(i):
    data = open("error",'r').read()
    lines = data.split("\n")
    xs = []
    ys = []
    x = 0
#    print("#############")
    for line in lines:
        if len(line)>0:
         ys.append(float(line))
         xs.append(x)
 #        print(line)
         x+=0.1
    ax.clear()
    ax.plot(xs,ys)

ani = animation.FuncAnimation(fig,animate,interval=100)
plt.show()
