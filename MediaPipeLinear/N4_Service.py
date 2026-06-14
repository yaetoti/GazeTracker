from pathlib import Path
import time
from Utils.CameraStream import CameraStream
from N3_Calibrate import CalibrationModel
from Common import *
from Utils.SharedMemoryWriter import SharedMemoryWriter

# Params
assetsDir = Path("../Assets/Model0")
imagesDir = assetsDir / "Images"
weightsDir = assetsDir / "Weights"
weightFile = weightsDir / "model.pt"

def run_main_loop():
  # API
  gazeWriter = SharedMemoryWriter()
  gazeWriter.Write(0, 0)


  # Model
  cap = CameraStream(0)
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  calibrator = CalibrationModel()
  calibrator.load_state_dict(torch.load(weightFile, map_location=device))
  calibrator.to(device)
  calibrator.eval()

  preprocess = Preprocess()

  running = True

  last_frame_number = -1
  x = 0
  y = 0
  start_time = time.time()
  elapsed_time = 0.0
  width = 1920
  height = 1080

  while running:
    # Timer here
    current_time = time.time()
    elapsed_time = current_time - start_time
    start_time = current_time


    # Face detection
    status, frame, frame_number = cap.read()
    if not status:
      continue

    if frame_number - last_frame_number > 0:
      # Collect features
      process_start = time.time()
      frameRgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

      preprocess_start = time.time()
      features = preprocess.Handle(frameRgb)
      preprocess_time = time.time() - preprocess_start

      if features is not None:
        last_frame_number = frame_number

        # Extract tensor for specified indices
        featuresTensor = torch.tensor(features, dtype=torch.float32, device=device).unsqueeze(0)

        calibrator_start = time.time()
        outputX, outputY = calibrator(featuresTensor)
        calibrator_time = time.time() - calibrator_start

        # Update coords
        gazeWriter.Write(outputX, outputY)

        #x, y = int(outputX * width), int(outputY * height)
        #total_process_time = time.time() - process_start
        #print(f"Preprocess time: {preprocess_time * 1000:.2f}ms, Calibrator time: {calibrator_time * 1000:.2f}ms, Total process time: {total_process_time * 1000:.2f}ms")


if __name__ == '__main__':
  # Loop
  run_main_loop()
