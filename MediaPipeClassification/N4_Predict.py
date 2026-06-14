from pathlib import Path
import pygame
from Utils.CameraStream import CameraStream
from N3_Calibrate import CalibrationModel
from Common import *
from Utils.Stats import append_sample_data, calculate_and_print_stats
from Utils.Timer import Timer

# Params
assetsDir = Path("../Assets/Model0")
imagesDir = assetsDir / "Images"
weightsDir = assetsDir / "Weights"
weightFile = weightsDir / "model.pt"

def run_main_loop():
  # Model
  cap = CameraStream(0)
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  calibrator = CalibrationModel()
  calibrator.load_state_dict(torch.load(weightFile, map_location=device))
  calibrator.to(device)
  calibrator.eval()

  preprocess = Preprocess()

  # Pygame
  info = pygame.display.Info()
  width = info.current_w
  height = info.current_h

  surface = pygame.display.set_mode((1920, 1080), pygame.FULLSCREEN)
  pygame.display.set_caption("PyGame Stand")

  clock = pygame.time.Clock()
  running = True
  fps = 144
  clock.tick(fps)

  last_frame_number = -1
  x = 0
  y = 0
  time = 0.0

  # Stats
  stats = []

  while running:
    dt = clock.tick(fps)
    time += dt

    # events
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        running = False
      if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
        running = False

    # update
    # Get mouse position
    mouse_x, mouse_y = pygame.mouse.get_pos()

    # Face detection
    status, frame, frame_number = cap.read()
    if not status:
      continue

    if frame_number - last_frame_number > 0:
      processTimer = Timer()

      # Collect features
      frameRgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      features = preprocess.Handle(frameRgb)

      processTime = processTimer.GetElapsedTimeMs()
      calibrateTimer = Timer()

      if features is not None:
        last_frame_number = frame_number

        # Extract tensor for specified indices
        featuresTensor = torch.tensor(features, dtype = torch.float32, device = device).unsqueeze(0)

        outputX, outputY = calibrator(featuresTensor)
        outputX, outputY = torch.softmax(outputX.squeeze(), 0).cpu(), torch.softmax(outputY.squeeze(), 0).cpu()


        def SoftArgmax(inputX: torch.Tensor, inputY: torch.Tensor):
          xId = torch.tensor([x for x in range(X_CLASSES)], dtype=torch.float32)
          yId = torch.tensor([y for y in range(Y_CLASSES)], dtype=torch.float32)
          return (inputX * xId).sum().item() / X_CLASSES, (inputY * yId).sum().item() / Y_CLASSES

        def DeclassifyArgmax(inputX: torch.Tensor, inputY: torch.Tensor):
          indexX, indexY = torch.argmax(outputX).item(), torch.argmax(outputY).item()
          x, y = Declassify(indexX, 0, 1, X_CLASSES), Declassify(indexY, 0, 1, Y_CLASSES)
          return x, y

        x, y = SoftArgmax(outputX, outputY)
        #x, y = DeclassifyArgmax(outputX, outputY)

        # indexX, indexY = torch.argmax(outputX).item(), torch.argmax(outputY).item()
        # x, y = Declassify(indexX, 0, 1, X_CLASSES), Declassify(indexY, 0, 1, Y_CLASSES)
        x = x * width + (width / X_CLASSES) / 2
        y = y * height + (height / Y_CLASSES) / 2

        calibrateTime = calibrateTimer.GetElapsedTimeMs()

        if time > 2.0:
          append_sample_data(stats, mouse_x, mouse_y, x, y, processTime, calibrateTime)
          calculate_and_print_stats(stats)

          if len(stats) > 1000:
            exit()

    # render
    surface.fill((55, 55, 55))

    # Draw a red circle at the mouse position
    pygame.draw.circle(surface, (70, 110, 255), (mouse_x, mouse_y), 20)
    pygame.draw.circle(surface, (255, 0, 0), (x, y), 20)

    pygame.display.flip()

  pygame.quit()



if __name__ == '__main__':
  pygame.init()

  # Loop
  run_main_loop()

  pygame.quit()
