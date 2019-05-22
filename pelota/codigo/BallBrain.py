#coding: utf-8
import sklearn.neighbors as n
from pyrobot.brain import Brain

import math
import pid

import  cv2
import numpy as np
import clasificador as c
import ball

#CAPTURE ="/dev/video1"
CAPTURE =0
thresh_turn = 10

class BrainTestNavigator(Brain):
  Kp_v =0.9
  Ki_v = 0.0
  Kd_v = 0.13

  Kp_r = 1
  Ki_r = 0.00
  Kd_r = 0.1332

  def setup_ball(self):
    self.pid_velocidad = pid.PID(self.Kp_v,self.Ki_v,self.Kd_v)
    self.pid_rotacion = pid.PID(self.Kp_r,self.Ki_r,self.Kd_r)

  def setup_capture(self):
    global CAPTURE
    self.capture = cv2.VideoCapture(CAPTURE)
    self.capture.set(3,360)
    self.capture.set(4,270)
    self.cl = c.Clasificador()

    self.objetivo = 30.0

    _,img = self.capture.read()
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    self.video = cv2.VideoWriter("resultado.avi",fourcc,6,(img.shape[1],img.shape[0]))
    self.segmentacion = cv2.VideoWriter("segmentacion.avi",fourcc,6,(img.shape[1],img.shape[0]))
    self.half =  float(img.shape[1]/2)
    cats = self.cl.classif_img(img)
    self.focal_length = ball.get_focal_length(cats,self.objetivo) # sabemos que la foto se hizo desde 30cm

  def setup(self):
      self.setup_ball()
      self.robot.range.units = "ROBOTS"
      self.setup_capture()


  def step_ball(self):

    global hasBall
    global ballX
    global ballY
    if hasBall:
        self.rotacion = self.pid_rotacion.compute(ballX)

        self.velocidad = self.pid_velocidad.compute(ballY)

    else:
        self.velocidad = 0
        self.rotacion = 0

    print('v:{} r: {}'.format(self.velocidad,self.rotacion))

  def step_capture(self):

    hasBall,ballX,ballY= False,None,None

    if self.capture.isOpened():
        ret,img = self.capture.read()
    if not ret:
        return hasBall,ballX,ballY

    # obtengo la matriz de categorias
    cats = self.cl.classif_img(img)
    paleta = np.array([[255,0,0],[0,0,255],[255,255,0],[0,255,0]],dtype=np.uint8)
    # ahora pinto la imagen
    #cv2.imshow("Segmentacion Euclid",cv2.cvtColor(paleta[cats],cv2.COLOR_RGB2BGR))
    # Obtengo los parámetros de la pelota en la imagen
    ul,dr,center = ball.ball_square(cats)

    if ul != (0,0) and dr != (0,0) and center != (0,0):
        hasBall = True
        # Solo si se ha encontrado una pelota
        # calculo la distancia de la pelota a la camara
        dist = ball.distance_to_camera(dr,ul,self.focal_length)

        #Dibujar un rectángulo alrededor de la pelota y un punto en su centro
        cv2.rectangle(img,ul,dr,(255,0,0),2)
        cv2.circle(img,center,3,(0,0,255),-1)

        cv2.putText(img,"{:06.3f}cm".format(dist),(img.shape[1]-250,img.shape[0]-20),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,0,0),3)
        ballY = (dist - self.objetivo)
        if ballY < 0:
          ballY /= float(self.objetivo) # como máximo la tiene justo delante
        else:
          ballY /= 1.0  # como maximo está a cuatro metros
        h, w = cats.shape
        ballX = ((w/2)-center[0])/w 
    # Guardar el video
    self.video.write(img)
    self.segmentacion.write(paleta[cats])
    return hasBall, ballX, ballY



  def step(self):
    global hasBall
    global ballX,ballY
    hasBall,ballX,ballY = self.step_capture()
    print("ErrX= {}, Erry = {}".format(ballX,ballY))
    #print "I got from the simulation",hasLine,lineDistance,icon
    #print self.estado
    #    print("Distance: {}".format(lineDistance))
    self.step_ball()
    self.move(self.velocidad,self.rotacion)

def INIT(engine):
  assert (engine.robot.requires("range-sensor") and
          engine.robot.requires("continuous-movement"))

  # If we are allowed (for example you can't in a simulation), enable
  # the motors.
  try:
    engine.robot.position[0]._dev.enable(1)
  except AttributeError:
    pass
  return BrainTestNavigator('BrainTestNavigator', engine)
