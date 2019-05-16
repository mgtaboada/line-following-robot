import cv2

captura = cv2.videoCapture(0)
_,img = captura.read()

fourcc = cv2.VideoWriter_fourcc(*"XVID")
video = cv2.VideoWriter("video.avi",fourcc,24,(img.shape[1],img.shape[0]))
def captura_frame(capture,video):
    ret,img = capture.read()
    if not ret:
        return False
    video.write(img)
    cv2.waitKey(1000/24)
    return True

try:
    while captura_frame(captura,video):
        pass
except KeyboardInterrupt:
    video.release()
