
import taichi as ti
import numpy as np
from matplotlib.colors import hex2color
import cv2

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

def load_texture(input):

    if type(input) == str:
        img = np.array(Image.open(input).convert("RGB"), dtype=np.float32) / 255
    elif isinstance(input, Image.Image):
        img = np.array(input).astype(np.float32)
        if np.max(img) > 1:
            img /= 255
    else:
        img = input.astype(np.float32)

    if len(img.shape) > 2:
        if len(img.shape) == 4:
            img = img[:, :, :3]
        img = np.transpose(img[:, ::-1, :], (1, 0, 2))
        w, h, k = img.shape
        texture = ti.Vector.field(3, dtype=ti.f32, shape=(w, h))
    else:
        img = np.transpose(img[:, ::-1], (1, 0))
        w, h = img.shape
        texture = ti.field(ti.f32, shape=(w, h))

    texture.from_numpy(img)
    return texture

def img2ascii(img, n=40):
    def color(char, style=0, fg=30, bg=41):
        format = ";".join([str(style), str(fg), str(bg)])
        s1 = "\x1b[%sm %s \x1b[0m" % (format, char)
        return s1

    img = cv2.resize(img, (n, n))
    img = np.array(img)[:, :, :3].mean(axis=-1)
    img = np.digitize(img, [np.quantile(img, i) for i in np.linspace(0.1, 0.9, 8)])
    # img = palette[img-1]

    img = "\n".join(
        [
            "".join([color(" ", bg=np.arange(40, 48)[c - 1]) for c in line])
            for line in img
        ]
    )

    return img

def hex2numpy(c: str) -> np.ndarray:
    return np.array(hex2color(c))

@ti.pyfunc
def get_palette(palette) -> ti.Vector.field:
    palette_field = ti.Vector.field(3, dtype=ti.f32, shape=(len(palette), 1))
    palette_field.from_numpy(
        np.array([[hex2numpy(p)] for p in palette], dtype=np.float32)
    )
    return palette_field
