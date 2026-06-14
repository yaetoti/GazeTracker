class EMA:
  def __init__(self, alpha, value=0):
    self.alpha = alpha
    self.value = value

  def update(self, newValue):
    self.value = newValue * self.alpha + (1 - self.alpha) * self.value
    return self.value