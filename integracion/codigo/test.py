import cv2
cap = 1
c = cv2.VideoCapture(cap)

ret,_ = c.read()
try:
 while ret:
  ret,img = c.read()
  cv2.imshow("cosas",img)
  cv2.waitKey(1)
except KeyboardInterrupt:
 cv2.destroyAllWindows()
