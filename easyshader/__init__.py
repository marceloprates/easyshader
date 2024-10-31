import taichi as ti

ti.init(arch=ti.gpu, verbose=False)

from .rendering import Rendering, Light
from .shape import *
from .primitives import *
