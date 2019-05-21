#coding: utf-8
from pyrobot.brain import Brain

import math
import pid

import  cv2
import numpy as np
import clasificador as c
import analisis as a
import pickle #Cargar los datos
import sklearn.neighbors as n


#CAPTURE ="/dev/video1"
CAPTURE =0
thresh_turn = 10
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

  Kp_v =0.9
  Ki_v = 0.0
  Kd_v = 0.13
  
  Kp_r = 1 
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
    self.capture.set(3,360)
    self.capture.set(4,270)
    self.cl = c.Clasificador()

    #  entrada: punto de entrada
    #  salida_mantener: salida que marcaba la última flecha que se encontró el robot
    #  cnt_una: contador de frames seguidos en los que ha aparecido una sola linea
    #  salidas_flecha: lista de todas las salidas que ha marcado la flecha
   
    self.salida = None
    self.salida_mantener = None
    self.salidas_flecha = []
    self.entrada = None
    self.estado_capture="una linea"
    self.cnt_una = thresh_turn # contador para cuantos frames lleva de una linea
    _,img = self.capture.read()
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    self.video = cv2.VideoWriter("resultado.avi",fourcc,6,(img.shape[1],img.shape[0]))
    self.segmentacion = cv2.VideoWriter("segmentacion.avi",fourcc,6,(img.shape[1],img.shape[0]))
    self.half =  float(img.shape[1]/2)

    
  def setup_recog(self):
    pass
    # with open("datasetFiguras", 'rb') as f:
    #   dataset = pickle.load(f)
    # with open("datasetEtiquetasFiguras", 'rb') as f:
    #   dataset_etiquetas = pickle.load(f)
    # self.skmaha = n.KNeighborsClassifier(1,algorithm='brute',metric='mahalanobis',metric_params={'V':np.cov(dataset)})

    # self.skmaha.fit(dataset,dataset_etiquetas)

  def setup(self):
    self.robot.range.units = "ROBOTS"
    self.ticks_en_linea = 0
    self.estado = self.SPIRAL
    self.setup_line()
    self.setup_spiral()
    self.setup_avoid()
    self.setup_capture()
    self.setup_recog()
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
      print("NO LINEA")
      if self.ticks_en_linea >= 5:
        self.estado = self.NOVENTA
        self.orientacion_0 = self.robot.th
        ##self.step_noventa()
      else:
        self.ticks_en_linea = 0
        self.estado = self.SPIRAL
        #self.setup_spiral()
        ##self.step_spiral()
      # if self.linepos == -1: # la linea esta a la izquierda
      #   self.rotacion =- 0.5 # girar a la derecha
      # elif self.linepos == 1:
      #   self.rotacion = 0.5 # girar a la izquierda
      # else:
      #   self.rotacion = 0

      # self.velocidad = 0.1# self.NO_FORWARD  # moverse despacito

      
    print('v:{} r: {}'.format(self.velocidad,self.rotacion))

  def step_capture(self):
    hasLine,lineDistance,icon,turn = None,None,None,None
    self.entrada = None
    if self.capture.isOpened():
      ret,img = self.capture.read()
      if not ret:
        return hasLine,lineDistance,icon
      h = int(img.shape[0]*0)
      cats = self.cl.classif_img(img[h:,:])
      lin = (cats ==2).astype (np.uint8)
      estado=0
      mar = a.encontrar_icono((cats == 0).astype(np.uint8))
      paleta = np.array([[0,0,255],[0,0,0],[255,0,0],[0,0,0]],dtype=np.uint8)
      tcolor = (255,255,255)
      salida_final = self.salida
      #Detecto tipo del inea
      tipo = a.tipo_linea(lin)
      if tipo != None: 
         hasLine = True
      if tipo == None:
        hasLine = False
	lineDistance = 0
	return hasLine, lineDistance, None
      elif (tipo < a.DOS_SALIDAS): # una linea
        self.cnt_una +=1 # incremento contador
        self.salidas_flecha = [] # vacío lista de la ultima flecha
        if False:#self.cnt_una < thresh_turn:
          # si llevo poco rato viendo una sola linea
          # mantengo la direccion de la ultima flecha
          tcolor = (255,0,0) # color azul
          salida_final = self.salida_mantener
        else:
    	   if np.any(mar):
              # si hay marca es un icono
              
              sym = mar
              grayscale_img = cv2.cvtColor(cv2.cvtColor(paleta[sym],cv2.COLOR_RGB2BGR),cv2.COLOR_BGR2GRAY)
              _, binary_img = cv2.threshold(grayscale_img, 20, 255, cv2.THRESH_BINARY)
              _,conts,hier = cv2.findContours(binary_img,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
              #conts,hier = cv2.findContours(binary_img,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
              if len(conts) > 0:
                moments = cv2.HuMoments(cv2.moments(binary_img)).flatten()
                #icon = self.skmaha.predict([moments])[0]
                icon = None
              # como los pixeles segmentados como flecha de cats son de una marca, los elimino
              cats[cats==0]=1
    
           # lleva suficiente tiempo sin ver una flecha como para
           # volver a seguir la linea
           tcolor = (0,255,0) # color verde
           self.entrada,salida_final,estado= a.entrada_salida(cats,self.entrada, self.salida)
           
	   print("Salida obtenida ",salida_final)
      else: # dos lineas

        if np.any(mar):
          # veo una flecha -> la sigo y apunto para donde apunta
          tcolor = (255,0,255) # color morado
          self.entrada,nueva_salida_flecha,estado=a.entrada_salida(cats,self.entrada,self.salida)
	  print("Salida para la flecha ",nueva_salida_flecha)
	
          self.salidas_flecha.append(nueva_salida_flecha)
          salida_final = nueva_salida_flecha
          
          cv2.putText(img,str(salida_final),(0,img.shape[0]-60),cv2.FONT_HERSHEY_SIMPLEX,1,tcolor,1)
        else:
          # no hay flecha -> vacío la lista
          self.salidas_flecha = []

        if self.salidas_flecha != []: # ya hemos visto la flecha
          tcolor = (0,0,0) # color negro
          self.cnt_una = 0
          self.salida_mantener =tuple(np.median(np.array(self.salidas_flecha),axis=0).astype(int))
          #########################################################################################
          #salida_final por esta rama no adquiere valor?
          #salida_final = self.salida_mantener
        else:
          if self.cnt_una == 0: # hemos perdido la flecha pero seguimos viendo el cruce
             tcolor = (125,0,125) # color morado oscuro
             salida_final = self.salida_mantener
          else:
             tcolor = (125,125,125)#Gris
	     self.entrada, salida_final,estado = a.entrada_salida(cats,self.entrada,self.salida)
      if salida_final != None:
          self.salida = salida_final
      print("Salida final final final: {}".format(self.salida))
      lineDistance = ((self.half-self.salida[0]))/self.half
      cv2.putText(img,str(self.salida),(0,img.shape[0]-20),cv2.FONT_HERSHEY_SIMPLEX,1,tcolor,1)
      cv2.putText(img,"{}({})".format("",self.cnt_una),(0,40),cv2.FONT_HERSHEY_SIMPLEX,1,tcolor,1)
      cv2.circle(img[h:,:],tuple(self.salida),3,(0,255,0),-1)
      #cv2.circle(img,(self_entrada[0],img[h:,:].shape[1]),3,(0,255,0),-1)
      if np.any(mar):
          p1,p2,salida = a.direccion_flecha(mar)
          cv2.arrowedLine(img[h:,:],p1,p2,(255,255,255),3)
          cv2.circle(img[h:,:],tuple(salida),3,(0,0,0),-1)
          cv2.putText(img,str(salida),(0,img.shape[0]-80),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),1)
          ########################################################################################
          #Pintar la salida de la flecha.

    #cv2.imshow("segmentacion",paleta[lin])
    #cv2.imshow("video",img)
    self.video.write(img)
    cv2.waitKey(1)
    return hasLine,lineDistance,icon,estado

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
    hasLine,lineDistance,icon,estado = self.step_capture()
    #print "I got from the simulation",hasLine,lineDistance,icon
    #print self.estado
    print("Distance: {}".format(lineDistance))
    if estado != 0:
        self.velocidad = 0
        self.rotacion = estado
    else:
       self.estados[self.LINEA]()
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
