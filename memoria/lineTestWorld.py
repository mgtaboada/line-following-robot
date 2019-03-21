"""
A simulation with a simple line on the floor.
"""

from lineSimulation import LineSimulation
from pyrobot.simulators.pysim import TkPioneer, \
     PioneerFrontSonars, PioneerFrontLightSensors

def INIT():
    # (width, height), (offset x, offset y), scale
    sim = LineSimulation((450,675), (20,650), 32,
                         background="line-images/lineBackground-1.png")  
                         #background="line-images/lineBackground-2.png")
                         #background="line-images/square.png")
                         #background="line-images/hand.png")
                         #background="line-images/camino_recto.png")
                         #background="line-images/ellipse_2.png")
                         #background="line-images/ellipse.png")
                         #background="line-images/circle.png")
                         #background="line-images/cloud.png")
    # an example of an obstacle on the line
      # x1, y1, x2, y2
    sim.addBox(5, 12, 6, 11)

    sim.addRobot(60000, 
		 # name, x, y, th, boundingBox
                 TkPioneer("RedErratic",
                           #position for cloud
                           #3,18,4,
                           #position for hand
                           # 1.4,18,3,
                           #position for square
                           #1.4,18,5,
                           #position for circle
                           #1.4,18,5,
                           # position for ellipse_2
                           #1,19.5,4.0,
                           # position for ellipse
		           #1, 19.1, 4.0,
			   # position for lineBackground-1
		            1, 18.9, 3.8,
			   # position for lineBackground-2
		           #8.5, 2.35, 1.57,
                           ((.185, .185, -.185, -.185),
                            (.2, -.2, -.2, .2))))

    # add some sensors:
    sim.robots[0].addDevice(PioneerFrontSonars())

    # to create a trail
    sim.robots[0].display["trail"] = 1
    
    return sim
