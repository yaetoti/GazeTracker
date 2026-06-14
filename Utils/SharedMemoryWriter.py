import mmap
import struct

class SharedMemoryWriter:
  STRUCT_FORMAT = "<ff"
  STRUCT_SIZE = struct.calcsize(STRUCT_FORMAT)

  def __init__(self, name="Local\\GazeDetectorIPC"):
    self.name = name
    self.mm = mmap.mmap(
      -1,
      self.STRUCT_SIZE,
      self.name,
      mmap.ACCESS_WRITE
    )

    self.version = 0

  def Write(self, x: float, y: float):
    packed = struct.pack(
      self.STRUCT_FORMAT,
      x, y
    )

    self.mm.seek(0)
    self.mm.write(packed)

  def __del__(self):
    self.Close()

  def Close(self):
    self.mm.close()