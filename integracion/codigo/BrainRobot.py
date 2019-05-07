from pyrobot.brain import Brain

import math
import pid

import  cv2
import numpy as np
import clasificador as c
import analisis as a
import pickle #Cargar los datos
import sklearn.neighbors as n


CAPTURE = 0

class BrainTestNavigator(Brain):
  #estados
  SPIRAL = 0
  LINEA = 1
  GIRO = 2
  NOVENTA = 3
  SEGUIR_OBJETO = 4
  GIRAR_ESQUINA = 5
  
  NO_FORWARD = 0
  SLOW_FORWARD = 0.1
  MED_FORWARD = 0.5
  FULL_FORWARD = 1.0

  NO_TURN = 0
  MED_LEFT = 0.5
  HARD_LEFT = 1.0
  MED_RIGHT = -0.5
  HARD_RIGHT = -1.0
  NO_ERROR = 0

  Kp_v = 0.9
  Ki_v = 0.0
  Kd_v = 0.13
  
  Kp_r = 0.19
  Ki_r = 0.00
  Kd_r = 0.1332
  total_error = 0.0
  n_cycles = 0.0
  
  def setup_line(self):
    self.pid_velocidad = pid.PID(self.Kp_v,self.Ki_v,self.Kd_v)
    self.pid_rotacion = pid.PID(self.Kp_r,self.Ki_r,self.Kd_r)
    self.logfile = "error"
    with open(self.logfile,'w') as f:
      f.write("")
    self.linepos = 0 # -1 si izquierda, 1 si derecha

  def setup_spiral(self):
    self.velocidad = 0.5
    self.rotacion = 1.0
    self.ticks_en_espiral = 0
    self.target = self._n_cycles()
  def setup_avoid(self):
    pass
  def setup_capture(self):
    global CAPTURE
    self.capture = cv2.VideoCapture(CAPTURE)
    self.cl = c.Clasificador()
    self.entrada = None
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    self.video = cv2.VideoWriter("resultado.avi",fourcc,1,(img.shape[1],img.shape[0]))
    self.segmentancion = cv2.VideoWriter("segmentacion.avi",fourcc,1,(img.shape[1],img.shape[0]))
    
  def setup_recog(self):
    with open("datasetFiguras", 'rb') as f:
      dataset = pickle.load(f)
    with open("datasetEtiquetasFiguras", 'rb') as f:
      dataset_etiquetas = pickle.load(f)
    self.skmaha = n.KNeighborsClassifier(1,algorithm='brute',metric='mahalanobis',metric_params={'V':np.cov(dataset)})

    self.skmaha.fit(dataset,dataset_etiquetas)

  def setup(self):
    self.robot.range.units = "ROBOTS"
    self.ticks_en_linea = 0
    self.estado = self.SPIRAL
    self.setup_line()
    self.setup_spiral()
    self.setup_avoid()
    self.setup_capture()
    self.estados = {self.SPIRAL:self.step_spiral,self.LINEA:self.step_line, self.GIRO: self.step_giro, self.NOVENTA: self.step_noventa,
            self.SEGUIR_OBJETO:self.seguir_objeto, self.GIRAR_ESQUINA:self.girar_esquina}
  def _n_cycles(self):
        '''Devuelve el numero de ciclos necesarios para dar media vuelta con la configuracion actual'''
        return int(((3/self.rotacion)**2)/2.0)

  def step_noventa(self):
    print "NOVENTA"
    global hasLine
    if hasLine:
      self.estado = self.LINEA
      #self.step_line()
    else:
      self.velocidad = 0
      if abs(self.robot.th - self.orientacion_0) >= 90:
        self.linepos *= -1 # cambiar de sentido para hacer el otro giro
        self.estado = self.GIRO
        self.step_giro()
      else:
        if self.linepos == 1:
          self.rotacion = -0.5
        else:
          self.rotacion = 0.5
    
  def step_giro(self):
    print "GIRO"
    global hasLine
    if hasLine:
      self.estado = self.LINEA
      self.step_line()
    else:
      self.velocidad = 0
      if self.linepos == 1:
        self.rotacion = -0.5
      else:
        self.rotacion = 0.5

  def step_line(self):

    global hasLine
    global lineDistance

    #Si encuentra un objeto se para y pasa a seguirlo
    if self.front <= 1:
        self.velocidad = 0
        self.rotacion = 0
        self.estado = self.SEGUIR_OBJETO
    elif (hasLine):
      self.ticks_en_linea +=1
      print("LINEA")
      with open(self.logfile,'a') as f:
        f.write(str(lineDistance) + "\n")
      if lineDistance> 0: # derecha
        self.linepos = 1
      elif lineDistance < 0:
        self.linepos = -1
        
      
      self.rotacion = self.pid_rotacion.compute(lineDistance)

      self.velocidad = max([0,1- abs(self.pid_velocidad.compute(self.rotacion))])
      self.total_error += abs(lineDistance)

    else:
      if self.ticks_en_linea >= 5:
        self.estado = self.NOVENTA
        self.orientacion_0 = self.robot.th
        self.step_noventa()
      else:
        self.ticks_en_linea = 0
        self.estado = self.SPIRAL
        #self.setup_spiral()
        self.step_spiral()
      # if self.linepos == -1: # la linea esta a la izquierda
      #   self.rotacion =- 0.5 # girar a la derecha
      # elif self.linepos == 1:
      #   self.rotacion = 0.5 # girar a la izquierda
      # else:
      #   self.rotacion = 0

      # self.velocidad = 0.1# self.NO_FORWARD  # moverse despacito

      
    print('v:{} r: {}'.format(self.velocidad,self.rotacion))

  def step_capture(self):
    hasLine,lineDistance,icon = None,None,None
    if self.capture.isOpened():
      ret,img = self.capture.read()
      if not ret:
        return hasLine,lineDistance,icon
      cv2.waitKey(1)
      h = int(img.shape[0]*0.3)
      cats = self.cl.classif_img(img[h:,:])
      lin = (cats ==2).astype (np.uint8)
      mar = (cats == 0).astype(np.uint8)
      tipo = a.tipo_linea(lin)
      paleta = np.array([[0,0,0],[255,255,255],[0,0,255],[0,0,0]],dtype=np.uint8)
      self.segmentancion.write(paleta[cats])
      if (tipo <a.DOS_SALIDAS) and np.any(mar): #  una sola linea
        # si hay marca es un icono
        sym = a.limpiar_img(mar)
      
        grayscale_img = cv2.cvtColor(cv2.cvtColor(paleta[sym],cv2.COLOR_RGB2BGR),cv2.COLOR_BGR2GRAY)
        _, binary_img = cv2.threshold(grayscale_img, 20, 255, cv2.THRESH_BINARY)
        
        _,conts,hier = cv2.findContours(binary_img,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
        
        if len(conts) > 0:
            moments = cv2.HuMoments(cv2.moments(binary_img)).flatten()
            icon = self.skmaha.predict([moments])[0]
            self.entrada, salida = a.entrada_salida(lin,self.entrada)
            half = img.shape[1]/2
            lineDistance = (abs(salida[0]-half))
            cv2.circle(img,(0,salida),10,(0,255,0),-1)
        if not np.any(lin):
          hasLine = False
        else:
          hasLine = True
    self.video.write(img)
    return hasLine,lineDistance,icon
  
  def step_spiral(self):
    print "SPIRAL"
    global hasLine
    if not(hasLine) or self.ticks_en_espiral < 2:
      self.ticks_en_espiral +=1
      self.n_cycles +=1
      if self.n_cycles >= self.target :
        if self.rotacion > 0:
          self.rotacion -=0.05
          self.target =self._n_cycles()
        self.n_cycles = 0
    else:
      self.ticks_en_espiral = 0
      self.estado = self.LINEA
      self.step_line()

  def distancias_ultrasonidos(self):
      #Toma las distancias de los sensores de ultrasonidos
      self.front = min([s.distance() for s in self.robot.range["front"]])
      self.left = min([s.distance() for s in self.robot.range["left"] + [self.robot.range[1]] + [self.robot.range[2]] ])
      self.right = min([s.distance() for s in self.robot.range["right"]])

  def girar_esquina(self):
      #Cuando esta siguiendo un objeto y se encuentra una esquina
      ret = (0,0)
      #Si esta cerca de la pared y no hay nada delante se cambia de estado
      if self.front > 1.5:
          self.estado = self.SEGUIR_OBJETO
      else:
          #Si esta muy lejos de la pared se acerca
          ret = (0, -.2)
      #else:
      #    ret = (0, -.3)
      self.velocidad = ret[0]
      self.rotacion = ret[1]


  def seguir_objeto(self):
      #Sigue un objeto
      global hasLine
      self.velocidad = 0.5
      self.rotacion  = 0
      #Si esta cerca de la pared se aleja
      if self.left < 0.5:
          self.velocidad = 0.2
          self.rotacion  = -.3
      elif self.left > 0.8:
          #Si esta lejos se acerca
          self.velocidad = 0.2
          self.rotacion  = .3
      if self.front <= 1:
          #Si hay un objeto delante gira
          self.velocidad = 0
          self.rotacion  = 0
          self.estado = self.GIRAR_ESQUINA
      if self.front > 2 and hasLine:
          #Si no hay nada y hay linea sigue la linea
          self.estado = self.LINEA



  def step(self):
    global hasLine
    global lineDistance
    self.distancias_ultrasonidos()
    hasLine,lineDistance,icon = self.step_capture()
    print "I got from the simulation",hasLine,lineDistance,icon
    print self.estado
    self.estados[self.estado]()
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
