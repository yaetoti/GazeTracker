import time

class Timer:
  def __init__(self):
    self.startTime = time.perf_counter_ns()

  def Start(self):
    self.startTime = time.perf_counter_ns()

  def GetElapsedTime(self):
    return time.perf_counter_ns() - self.startTime

  def GetElapsedTimeMs(self):
    return self.GetElapsedTime() / 1000000

  def PrintElapsedTimeMs(self):
    print(f"Elapsed time: {self.GetElapsedTimeMs()} ms")

class ScopedTimer:
  def __enter__(self):
    self.timer = Timer()
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    self.timer.PrintElapsedTimeMs()