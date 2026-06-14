import pygame
import sys
from pathlib import Path
from Utils.CameraStream import CameraStream
from Common import *



# Params
cWhite = (240, 240, 240)
cGray = (150, 150, 150)
cDark = (20, 20, 20)

images = 50
delay = 0
color = cDark
radius = 20

assetsDir = Path("../Assets/Model0")
imagesDir = assetsDir / "Images"
metadataStage0File = assetsDir / "metadataStage0.txt"

# Pygame
pygame.init()
pygame.font.init()
running = True
paused = True

# Data
info = pygame.display.Info()
width = info.current_w
height = info.current_h

imagesGathered = 0
currentClass = 0
classes = X_CLASSES * Y_CLASSES

def DrawSector(surface, classX, classY, classesX, classesY, width, height, color):
  minX, minY, maxX, maxY = classAABB(classX, classY, classesX, classesY, width, height)
  pygame.draw.rect(surface, color, (minX, minY, maxX - minX, maxY - minY), width=1)
  pygame.draw.line(surface, color, (minX, minY), (maxX, maxY), width=1)
  pygame.draw.line(surface, color, (maxX, minY), (minX, maxY), width=1)

def DecomposeClass(cls):
  return (cls - (cls // Y_CLASSES * X_CLASSES)), (cls // Y_CLASSES)

# PyGame data
if __name__ == "__main__":
  imageNumber = 0
  append = input("New file? ") != "+"

  if (append):
    file = open(metadataStage0File, "r+")
    # lines - metadata line
    imageNumber = len(file.readlines()) - 1
  else:
    file = open(metadataStage0File, "w")

  surface = pygame.display.set_mode((width, height), pygame.FULLSCREEN)
  pygame.display.set_caption("My first game")
  pygame.display.toggle_fullscreen()
  clock = pygame.time.Clock()

  # Write original resolution
  if not append:
    file.write(f"{width} {height}\n")

  # Text font setup
  font = pygame.font.Font(None, 24)

  # Main loop
  cap = CameraStream(0)
  lastFrameNumber = -1

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

        if event.key == pygame.K_TAB:
          currentClass = (currentClass + 1) % classes
          imagesGathered = 0

    # Update
    classX, classY = DecomposeClass(currentClass)
    coordsN = Declassify(classX, 0.0, 1.0, X_CLASSES), Declassify(classY, 0.0, 1.0, Y_CLASSES)

    if imagesGathered >= images:
      paused = True

    if not paused:
      success, frame, frameNumber = cap.read()
      if success:
        if lastFrameNumber < frameNumber - delay:
          lastFrameNumber = frameNumber

          # Write image and metadata (name + normalized coords)
          imageName = f"image{imageNumber}.png"
          cv2.imwrite(str(imagesDir / imageName), frame)
          #file.write(f"{imageName} {coordIndex}\n")
          file.write(f"{imageName} {coordsN[0]} {coordsN[1]}\n")

          imageNumber += 1
          imagesGathered += 1

    # Render
    surface.fill(color)

    # Render images gathered text
    text_surface = font.render(f"Images gathered: {imagesGathered} / {images}", True, cWhite)
    surface.blit(text_surface, (10, 10))

    circleColor = (150, 0, 0) if paused else (255, 0, 0)
    DrawSector(surface, classX, classY, X_CLASSES, Y_CLASSES, width, height, circleColor)

    # Show
    pygame.display.update()

  file.close()
  cap.stop()
  pygame.quit()
  sys.exit()