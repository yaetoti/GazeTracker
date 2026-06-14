import threading
import cv2

class CameraStream:
  def __init__(self, src=0):
    self.cap = cv2.VideoCapture(src, cv2.CAP_ANY)
    self.ret, self.frame = self.cap.read()
    self.frameNumber = 0
    self.lock = threading.Lock()
    self.running = True
    threading.Thread(target=self.update, daemon=True).start()

  def update(self):
    while self.running:
      ret, frame = self.cap.read()
      with self.lock:
        self.ret = ret
        self.frame = frame
        self.frameNumber += 1

  def read(self):
    with self.lock:
      return self.ret, self.frame, self.frameNumber

  def stop(self):
    self.running = False
    self.cap.release()