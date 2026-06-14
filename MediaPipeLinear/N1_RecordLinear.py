import pygame
import sys
import cv2
from pathlib import Path
from Utils.CameraStream import CameraStream
from Utils.Mover import Mover

# Params
cWhite = (240, 240, 240)
cGray = (150, 150, 150)
cDark = (20, 20, 20)

delay = 0
color = cGray
radius = 20
speed = 500

assetsDir = Path("../Assets/Model0")
imagesDir = assetsDir / "Images"
metadataStage0File = assetsDir / "metadataStage0.txt"

file = open(metadataStage0File, "w")

# Pygame
pygame.init()
pygame.font.init()
running = True
paused = True

def waitKeyHandler(ctx, deltaTime):
  global shouldSwitch

  if shouldSwitch:
    shouldSwitch = False
    return True

  return False

# Data
info = pygame.display.Info()
width = info.current_w
height = info.current_h

coordIndex = 0
images = 10000
imagesGathered = 0

chain = (
  Mover((width, height), (0, 0), speed)
)

patternCount = 5
for i in range(patternCount + 1):
  if i % 2 == 0:  # Move up in zig-zag
    chain = chain.moveN((i) / float(patternCount), 0.0)
  else:  # Move down in zig-zag
    chain = chain.moveN((i) / float(patternCount), 1.0)

for i in range(patternCount + 1):
  if i % 2 == 0:  # Move up in zig-zag
    chain = chain.moveN(0.0, (i) / float(patternCount))
  else:  # Move down in zig-zag
    chain = chain.moveN(1.0, (i) / float(patternCount))


# PyGame data
if __name__ == "__main__":
  surface = pygame.display.set_mode((width, height), pygame.FULLSCREEN)
  pygame.display.set_caption("My first game")
  pygame.display.toggle_fullscreen()
  clock = pygame.time.Clock()

  # Write original resolution
  file.write(f"{width} {height}\n")

  # Text font setup
  font = pygame.font.Font(None, 24)

  # Main loop
  cap = CameraStream(0)
  lastFrameNumber = -1
  imageNumber = 0

  coords = (0, 0)
  coordsN = (coords[0] / width, coords[1] / height)

  while running:
    dt = clock.tick(144) / 1000

    # Events
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        running = False

      if event.type == pygame.KEYDOWN:
        if event.key == pygame.K_ESCAPE:
          running = False

        if event.key == pygame.K_SPACE:
          paused = not paused

    # Update

    if not paused:
      coords = chain.tick(dt)
      coordsN = (coords[0] / width, coords[1] / height)

      success, frame, frameNumber = cap.read()
      if success:
        if lastFrameNumber < frameNumber - delay:
          lastFrameNumber = frameNumber

          # Write image and metadata (name + normalized coords)
          imageName = f"image{imageNumber}.png"
          cv2.imwrite(str(imagesDir / imageName), frame)
          file.write(f"{imageName} {coordsN[0]} {coordsN[1]}\n")

          imageNumber += 1
          imagesGathered += 1
          if imagesGathered > images:
            paused = True

    # Render
    surface.fill(color)

    # Render images gathered text
    text_surface = font.render(f"Images gathered: {imagesGathered} / {images}", True, cWhite)
    surface.blit(text_surface, (10, 10))

    circleColor = (150, 0, 0) if paused else (255, 0, 0)
    pygame.draw.circle(surface, circleColor, (coords[0], coords[1]), radius)

    # Show
    pygame.display.update()

  file.close()
  cap.stop()
  pygame.quit()
  sys.exit()