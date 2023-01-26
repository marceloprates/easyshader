
import taichi as ti
import numpy as np
from matplotlib.colors import hex2color

def hex2numpy(c):
    return np.array(hex2color(c))

@ti.pyfunc
def get_palette(palette):
    palette_field = ti.Vector.field(3, dtype=ti.f32, shape=(len(palette),1))
    palette_field.from_numpy(np.array([[hex2numpy(p)] for p in palette], dtype = np.float32))
    return palette_field

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

class Infix:
    def __init__(self, function):
        self.function = function
    def __ror__(self, other):
        return Infix(lambda x, self=self, other=other: self.function(other, x))
    def __or__(self, other):
        return self.function(other)
    def __rlshift__(self, other):
        return Infix(lambda x, self=self, other=other: self.function(other, x))
    def __rshift__(self, other):
        return self.function(other)
    def __call__(self, value1, value2):
        return self.function(value1, value2)

