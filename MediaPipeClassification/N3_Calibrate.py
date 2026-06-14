import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from Common import *

epochs = 2000
lossXCoef = 3.0
lossYCoef = 1.0

# Model
class CalibrationModel(nn.Module):
  def __init__(self, depth = 3, width = 1, layerSize = 128):
    super().__init__()

    self.layers = nn.ModuleList([
      nn.Sequential(
        nn.Linear(INPUT_SIZE, layerSize),
        *[nn.Sequential(
          #nn.BatchNorm1d(layerSize),
          nn.LayerNorm(layerSize),
          nn.ReLU(),
          #nn.Sigmoid(),
          nn.Linear(layerSize, layerSize)
        ) for _ in range(depth)]
      ) for _ in range(width)
    ])

    self.x = nn.Sequential(
      nn.Linear(layerSize * width, 128),
      nn.Dropout(0.2),
      nn.Linear(128, X_CLASSES),
    )
    self.y = nn.Sequential(
      nn.Linear(layerSize * width, 128),
      nn.Dropout(0.2),
      nn.Linear(128, Y_CLASSES),
    )

  def forward(self, input):
    outputs = [layer(input) for layer in self.layers]
    concatenated = torch.cat(outputs, dim = 1)
    x = self.x(concatenated)
    y = self.y(concatenated)
    return x, y

# Dataset
class CalibrationDataset(Dataset):
  '''
  Loads Stage1 data from a file.

  Data format:

  <imagePath> <xNormalized> <yNormalized> <label0> <label1> ...
  '''

  def __init__(self, file, device = torch.device('cpu')):
    self.samples = []

    with open(file, 'r') as f:
      # Read resolution
      line = f.readline()
      self.width, self.height = (float(item) for item in line.split())

      # Read data
      for line in f:
        # Prepare data
        data = line.strip().split()
        expected = [float(x) for x in data[1:3]]
        labels = [float(x) for x in data[3:]]

        # Prepare expected
        expected = PreprocessExpected(expected, self.width, self.height)

        result = torch.tensor(expected + labels, dtype = torch.float32)
        self.samples.append(result)

  def __len__(self):
    return len(self.samples)

  def __getitem__(self, id):
    return self.samples[id]


# Params
assetsDir = Path("../Assets/Model0")
imagesDir = assetsDir / "Images"
weightsDir = assetsDir / "Weights"
metadataStage1File = assetsDir / "metadataStage1.txt"
weightFile = weightsDir / "model.pt"

if __name__ == "__main__":
  try:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #loss_fn = nn.MSELoss()
    #loss_fn = PointDistanceLoss()
    loss_fn = nn.CrossEntropyLoss()
    model = CalibrationModel().to(device)
    optim = optim.AdamW(model.parameters(), lr=0.001)

    if input("Enter + to load weights: ") == '+':
      model.load_state_dict(torch.load(weightFile, map_location=device))
      print("Weights loaded")

    dataset = CalibrationDataset(metadataStage1File, device)
    dataloader = DataLoader(dataset, batch_size=4096, shuffle=True)

    model.train()

    for i in range(epochs):
      avgLoss = torch.tensor(0.0, device=device)

      for labels in dataloader:
        labels = labels.to(device)

        expected = labels[:, :2]
        modelInput = labels[:, 2:]

        # transform expected output
        #expectedX = torch.nn.functional.one_hot(expected[:, 0].to(torch.long), num_classes = X_CLASSES).to(torch.float32)
        #expectedY = torch.nn.functional.one_hot(expected[:, 1].to(torch.long), num_classes = Y_CLASSES).to(torch.float32)

        expectedX = expected[:, 0].to(torch.long)
        expectedY = expected[:, 1].to(torch.long)

        # estimate
        resultX, resultY = model(modelInput)

        # Calculate accuracy
        predX = torch.argmax(resultX, dim = 1)
        predY = torch.argmax(resultY, dim = 1)
        #trueX = torch.argmax(expectedX, dim = 1)
        #trueY = torch.argmax(expectedY, dim = 1)
        trueX = expectedX
        trueY = expectedY
        correctX = (predX == trueX).sum().item()
        correctY = (predY == trueY).sum().item()
        accuracyX = correctX / len(trueX) * 100
        accuracyY = correctY / len(trueY) * 100

        # backpropagate
        lossX = loss_fn(resultX, expectedX)
        lossY = loss_fn(resultY, expectedY)
        loss = lossX * lossXCoef + lossY * lossYCoef
        avgLoss += loss
        optim.zero_grad()
        loss.backward()
        optim.step()

        print(f"Batch Accuracy - X: {accuracyX:.2f}%, Y: {accuracyY:.2f}%")

      avgLoss /= len(dataloader)

      if i % 1 == 0:
        print(f"Epoch {i+1}/{epochs}, Loss: {avgLoss.item()}")

  except KeyboardInterrupt:
    print("\nTraining interrupted")


  if input("Enter + to save weights: ") == '+':
    torch.save(model.state_dict(), weightFile)
    print("Weights saved")
