import cv2
from pathlib import Path
from Common import *

assetsDir = Path("../Assets/Model0")
imagesDir = assetsDir / "Images"
metadataStage0File = assetsDir / "metadataStage0.txt"
metadataStage1File = assetsDir / "metadataStage1.txt"

if __name__ == "__main__":
  preprocess = Preprocess()
  imageNumber = 0

  append = input("New? ") != "+"

  fileStage0 = open(metadataStage0File, "r")
  fileStage0Lines = len(fileStage0.readlines())
  fileStage0.seek(0)

  if append:
    fileStage1 = open(metadataStage1File, "r+")
    fileStage1Lines = len(fileStage1.readlines())
    fileStage1.seek(0)
  else:
    fileStage1 = open(metadataStage1File, "w")
    fileStage1Lines = 0

  if append:
    assert fileStage0Lines > fileStage1Lines

    # Goto end of file1
    fileStage1.seek(0, 2)

    # Skip N lines from file0
    for _ in range(fileStage1Lines):
      fileStage0.readline()
  else:
    # Skip resolution
    line = fileStage0.readline()
    fileStage1.write(line)


  # Postprocess line by line
  for line in fileStage0:
    data = line.strip().split()
    imageName = data[0]

    # Extract features
    frame = cv2.imread(str(imagesDir / imageName), cv2.IMREAD_COLOR_RGB)
    height, width, _ = frame.shape

    features = preprocess.Handle(frame)
    if features is None:
      print(f"Image {imageName} was skipped")
      continue

    features = [str(x) for x in features]
    data += features

    # Write data to the file
    fileStage1.write(' '.join(data))
    fileStage1.write('\n')

    imageNumber += 1
    if imageNumber % 50 == 0:
      print(f"Processed {imageNumber} images.")

  fileStage0.close()
  fileStage1.close()
