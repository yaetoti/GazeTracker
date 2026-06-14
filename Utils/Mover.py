import math

class Mover:
  def __init__(self, resolution=(800, 600), position=(0, 0), speed=100):
    self.resolution = resolution
    self.x, self.y = position
    self.speed = speed

    self.targets = []
    self.current = 0

  def move(self, x, y):
    self.targets.append((x, y))
    return self

  def moveN(self, nx, ny):
    w, h = self.resolution
    self.targets.append((nx * w, ny * h))
    return self

  def tick(self, deltaTime):
    if not self.targets:
      return (self.x, self.y)

    tx, ty = self.targets[self.current]

    dx = tx - self.x
    dy = ty - self.y
    dist = math.sqrt(dx**2 + dy**2)

    step = self.speed * deltaTime

    if dist <= step:
      self.x, self.y = tx, ty
      self.current = (self.current + 1) % len(self.targets)
    else:
      self.x += dx / dist * step
      self.y += dy / dist * step

    return (int(self.x), int(self.y))