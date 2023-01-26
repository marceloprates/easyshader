# Taichi and Taichi-GLSL imports
import ast
from ast import arguments
import taichi as ti
from taichi import *
import taichi_glsl as tg
from taichi_glsl import mix, clamp

# Easyshader imports
from .utils import *
from .transformations import *
from easyshader import Rendering

# Other imports
import os
import re
import sys
import inspect
import IPython
import subprocess
from PIL import Image
from copy import deepcopy
from numbers import Number
from functools import reduce
from inspect import signature
from typing import Iterable, Callable, Union

# Numerical computing imports
import cv2
from numpy import pi as π
from numpy.linalg import norm
from sklearn.cluster import KMeans
from skimage.measure import find_contours
from shapely.geometry import Polygon, MultiPolygon

# Vsketch (for pen plotting)
import vsketch
from dataclasses import dataclass


def load_texture(input):

    if type(input) == str:
        img = np.array(Image.open(input).convert(
            "RGB"), dtype=np.float32) / 255
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

@dataclass
class Transform:
    transform: Union[Callable, str]

@ti.data_oriented
class Shape:
    """_summary_"""

    def __init__(
        self,
        sdf: Union[Callable, str] = "z",
        color: Union[Callable, str] = "#fff",
        transform: Union[Callable, str] = "(x,y,z)",
        textures: Iterable[Union[str, np.array]] = [],
        palette: Iterable[str] = ["#000", "#fff"],
        rendering_kwargs: dict = {},
        background_color=None,  #: str = "#fff",
        widgets: dict = {},
        gui: bool = False,
        text=None,
        mask_background=False,
        **kwargs,
    ):
        # Set attributes
        self.__dict__.update(locals())
        self.__dict__.update(kwargs)
        del self.__dict__["self"]

        # Load textures
        if "texture_fields" not in self.__dict__:
            self.texture_fields = []
            for texture in textures:
                self.texture_fields.append(load_texture(texture))

        # Evaluate sdf string representation
        if type(sdf) == str:
            self.sdf = self.augment_locals(sdf.strip())

        # Evaluate color string representation
        if type(color) == str:
            if color.startswith("#"):
                # Parse hex color
                self.color = lambda p, t: ti.Vector(hex2numpy(color))
            else:
                # Evaluate color function string representation
                self.color = self.augment_locals(color.strip())

        # Get palette field
        self.palette_field = get_palette(palette)

        # Render text
        if text is not None:

            if type(text) == str:
                text = dict(
                    s=text,
                    x=-0.06,
                    y=0,
                    size=27,
                    horizontalalignment="left",
                    verticalalignment="center",
                )

            h, w = Rendering(**self.rendering_kwargs).resolution
            dpi = 300
            fig, ax = plt.subplots(figsize=(20, 20), dpi=dpi)
            fig.patch.set_facecolor("#000")

            ax.text(**text, color="#fff")
            ax.axis("off")
            ax.autoscale()
            plt.savefig(".tmp/text.png")
            plt.close()
            self.text_field = load_texture(".tmp/text.png")

    def __call__(self, p: Iterable, t: float = 0) -> float:
        """
        Evaluate SDF

        Args:
            p (Iterable): (x,y,z) coordinates
            t (float, optional): Time. Defaults to 0.

        Returns:
            float: SDF evaluation at point p
        """
        return self.sdf(p, t)

    def _ipython_display_(self):
        IPython.display.display(self.render())
    
    def step(self):
        pass

    def augment_locals(self, func):

        def inner(p, t):
            # access p (x,y,z) coordinates directly
            x, y, z = p
            # 'palette' function for indexing palette
            def palette(i):
                return tg.bilerp(
                    self.palette_field,
                    ti.Vector([round(i) % self.palette_field.shape[0], 0]),
                )
            # 'texture' function for indexing textures
            def texture(i, x, y):
                h, w = self.texture_fields[i].shape
                return tg.bilerp(
                    self.texture_fields[i], ti.Vector(
                        [(-x + 0.5) * w, (-y + 0.5) * h])
                )
            # 'text' function for indexing text texture
            def text(x, y):
                h, w = self.text_field.shape
                return tg.bilerp(
                    self.text_field, ti.Vector(
                        [(-x + 0.5) * w, (-y + 0.5) * h])
                )
            
            for attr in self.__dict__.keys():
                attr = self.__dict__[attr]

            return eval(func) if type(func) == str else func(p,t)

        return inner

    def paint(
        self,
        color=None,
        palette=None,
        textures=None,
        inplace = False,
        **kwargs
    ):
        shape = Shape(
            sdf=self.sdf,
            **(dict(color = color) if color is not None else {}),
            **(dict(palette = palette) if palette is not None else {}),
            **(dict(textures = textures) if textures is not None else {}),
            **{k: v for k, v in kwargs.items()},
            **{
                k: v
                for k, v in self.__dict__.items()
                if (k not in kwargs)
                and k not in ["sdf", "color", "palette", "textures", "texture_fields"]
            },
        )
        
        if inplace:
            self.color = shape.color
            self.palette = shape.palette
            self.textures = shape.textures
        else:
            return shape
            

    # Rendering-related functions

    def GUI(
        self,
        frames=None,
        **rendering_kwargs
    ):

        rendering_kwargs = rendering_kwargs

        # Create rendering object
        rendering = Rendering(
            scene=[self],
            **rendering_kwargs,
        )

        gui_ = ti.GUI("", rendering.resolution)

        t = 0.0
        while gui_.running:
            # Render and get result
            self.step()
            rendering.color_buffer.fill(0.0)
            rendering.iteration = 0
            rendering.render(t)
            # Set GUI image and show
            gui_.set_image(rendering.result())
            gui_.show()
            # Update time & frame counter
            t += 2 * π / rendering.frames
        gui_.close()

    def rendering(self, **kwargs):

        background_color = (
            kwargs.pop(
                "background_color") if "background_color" in kwargs else None
        )

        rendering_kwargs = deepcopy(self.rendering_kwargs)
        rendering_kwargs.update(kwargs)

        return Shape(
            self.sdf,
            self.color,
            background_color=self.background_color if background_color is None else background_color,
            rendering_kwargs=rendering_kwargs,
            **{
                k: v
                for k, v in self.__dict__.items()
                if k not in ["sdf", "color", "rendering_kwargs", "background_color"]
            },
        )

    def render(self, **kwargs):
        return self.render_static(**kwargs)

    def render_static(self, **rendering_kwargs):

        background_color = None
        if "background_color" in rendering_kwargs:
            background_color = rendering_kwargs.pop("background_color")
        elif "background_color" in self.__dict__:
            background_color = self.background_color

        # Create rendering object
        rendering = Rendering(
            #scene=[Shape("z+1", color=background_color), self]
            #if background_color is not None
            #else [self],
            [self],
            **(
                {
                    k: v
                    for k, v in self.rendering_kwargs.items()
                    if k not in rendering_kwargs
                }
                if (self.rendering_kwargs is not None)
                else {}
            ),
            **rendering_kwargs
            # **(self.rendering_kwargs),
        )

        # Create & return static image
        rendering.render(0)
        result = (
            (255 * rendering.result()).transpose(1,
                                                 0, 2)[::-1, :, :].astype(np.uint8)
        )
        result = Image.fromarray(result, mode="RGBA")

        return result

    def animate(
        self,
        path=".tmp/output.gif",
        **rendering_kwargs,
    ):

        # Create folder
        directory = "/".join(path.split("/")[:-1])
        if not os.path.exists(directory):
            os.makedirs(directory)
        # Clean folder
        for f in os.listdir(directory):
            if f.endswith(".png"):
                os.remove(f"{directory}/{f}")

        background_color = (
            rendering_kwargs.pop("background_color")
            if "background_color" in rendering_kwargs
            else self.background_color
        )
        frames = rendering_kwargs.pop(
            "frames") if "frames" in rendering_kwargs else 20
        framerate = (
            rendering_kwargs.pop(
                "framerate") if "framerate" in rendering_kwargs else 30
        )

        # Create rendering object
        rendering = Rendering(
            scene=[self],
            **rendering_kwargs,
        )

        # Create animation
        for i, t in enumerate(np.linspace(0, 2 * np.pi, frames)[:-1]):
            self.step()
            rendering.color_buffer.fill(0.0)
            rendering.iteration = 0
            rendering.render(t)
            result = (
                (255 * rendering.result())
                .transpose(1, 0, 2)[::-1, :, :]
                .astype(np.uint8)
            )
            Image.fromarray(result).save(f".tmp/{i}.png")

        if path.split(".")[-1] == "gif":
            # Create & display GIF
            subprocess.call(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    f".tmp/%d.png",
                    path,
                    "-y",
                    "-framerate",
                    str(framerate),
                ]
            )
        else:
            # Create MP4
            subprocess.call(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    f".tmp/%d.png",
                    path,
                    "-y",
                    "-framerate",
                    str(framerate),
                ]
            )

        return IPython.display.Image(
            data=open(f".tmp/output.gif", "rb").read(), format="png"
        )

    def save(self, path, display=False):

        # Render static scene and save as temporary file
        result = self.render()
        Image.fromarray(result).save(path)

        if display:
            IPython.display.display(
                IPython.display.Image(
                    data=open(path, "rb").read(), format="png")
            )

    def vectorize(self, n_colors=2, **kwargs):

        img = np.array(self.render())[:, :, :3]

        h, w, _ = img.shape

        img = img.transpose(1, 0, 2).reshape(-1, 3)

        # Cluster image pixels by similarity into 'n_color' color classes
        kmeans = KMeans(n_clusters=n_colors + 1, random_state=0)
        kmeans.fit(img)
        labels = kmeans.predict(img).reshape(h, w)

        labels = np.pad(labels, [(4, 4), (4, 4)], mode="edge")

        # Create a list of "color layers" from KMeans output
        color_layers = [
            MultiPolygon([Polygon(c)
                         for c in find_contours(labels, level=level + 0.5)])
            for level in sorted(
                range(n_colors),
                key=lambda i: norm(kmeans.cluster_centers_[i]),
                reverse=True,
            )
        ]

        return color_layers

    def pen_plotter(self, n_colors=2, scale=1, save=None, rendering_kwargs=None):

        self.rendering_kwargs = rendering_kwargs

        vecs = self.vectorize(n_colors)

        class MySketch(vsketch.SketchClass):
            def draw(self, vsk: vsketch.Vsketch) -> None:
                vsk.size("a4", landscape=True)
                vsk.scale(scale, scale)

                for i, vec in enumerate(vecs):
                    vsk.stroke(i + 1)  # vsk.fill(i+1)
                    # if i > 0: vsk.fill(i+1);
                    vsk.geometry(vec)

                if save is not None:
                    vsk.save(save)

            def finalize(self, vsk: vsketch.Vsketch) -> None:
                vsk.vpype("linemerge linesimplify reloop linesort")

        MySketch().display()

    # Binary operators

    def __and__(self, o):
        """

        Computes the intersection between two shapes, f and g, expressed as Signed Distance Functions (SDF).

        The intersection is computed as:

        .. math::
            (f \& g)(p) = \min(f(p),g(p))

        :param other: A Shape expressed as a SDF
        :type other: Shape
        :return: Intersection between 'self' and 'other'
        :rtype: Shape

        .. code-block::
            :caption: Example: Intersection between sphere and a half-plane

                input: Sphere(.15) & Shape('x')
                output:

        .. image:: ../../docs/source/_static/and.png
            :scale: 60 %
            :alt: alternate text
            :align: center

        """

        # sdf,color = self.sdf, self.color

        if isinstance(o, str):
            sdf = self.augment_locals(f"self(p,t) + {o}")
            color = self.color
        else:

            def sdf(p, t):
                return max(self.sdf(p, t), o.sdf(p, t))

            def color(p, t):
                return self.color(p, t) if (self(p, t) < o(p, t)) else o.color(p, t)

        return Shape(
            sdf,
            color,
            **{k: v for k, v in self.__dict__.items() if k not in ["sdf", "color"]},
        )

    def __add__(self, o):

        if isinstance(o, str):
            if o.startswith("(") and o.endswith(")"):
                sdf = self.augment_locals(f"self(p+ti.Vector(list({o})),t)")
                color = self.augment_locals(
                    f"self.color(p+ti.Vector(list({o})),t)")
            else:
                sdf = self.augment_locals(f"self(p,t) - {o}")
                color = self.augment_locals(f"self.color(p,t) - {o}")
        elif isinstance(o, Union[float, int]):

            @ti.func
            def sdf(p, t):
                return self(p, t) - o

        elif isinstance(o, Iterable):
            if type(o) != ti.Vector:
                o = ti.Vector(o)

            @ti.func
            def sdf(p, t):
                return self(p - o, t)

            @ti.func
            def color(p, t):
                return self.color(p - o, t)

        else:

            @ti.func
            def sdf(p, t):
                return min(self(p, t), o(p, t))

            @ti.func
            def color(p, t):
                return self.color(p, t) if (self(p, t) < o(p, t)) else o.color(p, t)

        return Shape(
            sdf,
            color,
            **{k: v for k, v in self.__dict__.items() if k not in ["sdf", "color"]},
        )

    def __sub__(self, o):

        if isinstance(o, str):
            sdf = self.augment_locals(f"self(p,t) + {o}")
            color = self.color
        elif isinstance(o, Number):

            # @ti.func
            def sdf(p, t):
                return self(p, t) + o

        elif isinstance(o, Iterable):
            if type(o) != ti.Vector:
                o = ti.Vector(o)

            # @ti.func
            def sdf(p, t):
                return self(p + o, t)

            # @ti.func
            def color(p, t):
                return self.color(p + o, t)

        else:

            # @ti.func
            def sdf(p, t):
                return max(self(p, t), -o(p, t))

            # @ti.func
            def color(p, t):
                # if (self(p, t) < o(p, t)) else o.color(p, t)
                return self.color(p, t)

        return Shape(
            sdf,
            color,
            **{k: v for k, v in self.__dict__.items() if k not in ["sdf", "color"]},
        )

    def __mul__(self, o):

        if isinstance(o, str):
            sdf = self.augment_locals(f"self(p/{o},t)")
            color = self.color
        elif isinstance(o, Union[float, int]):

            @ti.func
            def sdf(p, t):
                return self(p / o, t)

            color = self.color
        elif isinstance(o, Iterable):
            if type(o) != ti.Vector:
                o = ti.Vector(o)

            @ti.func
            def sdf(p, t):
                return self(p / o, t)

            @ti.func
            def color(p, t):
                return self.color(p / o, t)

        return Shape(
            sdf,
            color,
            **{k: v for k, v in self.__dict__.items() if k not in ["sdf", "color"]},
        )

    def __pow__(self, o):

        @ti.func
        def sdf(p, t):
            return self(p, t)**o

        color = self.color

        return Shape(
            sdf,
            color,
            **{k: v for k, v in self.__dict__.items() if k not in ["sdf", "color"]},
        )

    def __truediv__(self, o: float):
        @ti.func
        def sdf(p, t):
            return self(p * o, t) / o

        return Shape(sdf, color=self.color)

    def __or__(self, o):
        """

        Computes the intersection between two shapes, f and g, expressed as Signed Distance Functions (SDF).

        The intersection is computed as:

        .. math::
            (f \& g)(p) = \min(f(p),g(p))

        :param other: A Shape expressed as a SDF
        :type other: Shape
        :return: Intersection between 'self' and 'other'
        :rtype: Shape

        .. code-block::
            :caption: Example: Intersection between sphere and a half-plane

                input: Sphere(.15) & Shape('x')
                output:

        .. image:: ../../docs/source/_static/and.png
            :scale: 60 %
            :alt: alternate text
            :align: center

        """

        # sdf,color = self.sdf, self.color

        if isinstance(o, str):
            sdf = self.augment_locals(f"{o.replace('dist','self(p,t)')}")
            color = self.color
        #else:
        #    def sdf(p, t):
        #        return max(self.sdf(p, t), o.sdf(p, t))
        #    def color(p, t):
        #        return self.color(p, t) if (self(p, t) < o(p, t)) else o.color(p, t)

        return Shape(
            sdf,
            color,
            **{k: v for k, v in self.__dict__.items() if k not in ["sdf", "color"]},
        )

    def __ror__(self, o):
        """

        Computes the intersection between two shapes, f and g, expressed as Signed Distance Functions (SDF).

        The intersection is computed as:

        .. math::
            (f \& g)(p) = \min(f(p),g(p))

        :param other: A Shape expressed as a SDF
        :type other: Shape
        :return: Intersection between 'self' and 'other'
        :rtype: Shape

        .. code-block::
            :caption: Example: Intersection between sphere and a half-plane

                input: Sphere(.15) & Shape('x')
                output:

        .. image:: ../../docs/source/_static/and.png
            :scale: 60 %
            :alt: alternate text
            :align: center

        """

        # sdf,color = self.sdf, self.color

        if isinstance(o, str):
            #sdf = self.augment_locals(f"")
            def sdf_(p,t):
                x,y,z = p
                p = ti.Vector(eval(o))
                return self.sdf(p,t)
            sdf = sdf_
            color = self.color
        #else:
        #    def sdf(p, t):
        #        return max(self.sdf(p, t), o.sdf(p, t))
        #    def color(p, t):
        #        return self.color(p, t) if (self(p, t) < o(p, t)) else o.color(p, t)

        return Shape(
            sdf,
            color,
            **{k: v for k, v in self.__dict__.items() if k not in ["sdf", "color"]},
        )

    # SDF operations
    
    def smooth_union(self, o, k=0.5):

        if type(k) != str:
            k = str(k)

        def sdf(p, t):
            d2, d1 = self(p, t), o(p, t)
            h = clamp(0.5 + 0.5 * (d2 - d1) / eval(k), 0, 1)
            return mix(d2, d1, h) - eval(k) * h * (1 - h)

        @ti.func
        def color(p, t):
            return self.color(p, t) if (self(p, t) < o(p, t)) else o.color(p, t)

        return Shape(sdf, color)

    def smooth_difference(self, o, k=0.5):

        if type(k) != str:
            k = str(k)

        def sdf(p, t):
            d2, d1 = self(p, t), o(p, t)
            h = clamp(0.5 - 0.5 * (d2 + d1) / eval(k), 0, 1)
            return mix(d2, -d1, h) + eval(k) * h * (1 - h)

        return Shape(
            sdf,
            self.color,
            **{k: v for k, v in self.__dict__.items() if k not in ["sdf", "color"]},
        )

    def smooth_intersection(self, o, k=0.5):

        if type(k) != str:
            k = str(k)

        def sdf(p, t):
            d2, d1 = self(p, t), o(p, t)
            h = clamp(0.5 - 0.5 * (d2 - d1) / eval(k), 0, 1)
            return mix(d2, d1, h) + eval(k) * h * (1 - h)

        return Shape(sdf, color=self.color)

    def twist(self, k, axis="y"):

        if type(k) != str:
            k = str(k)

        if axis == "x":
            # @ti.func
            def sdf(p, t):
                c_ = ti.cos(eval(k) * p.x)
                s_ = ti.sin(eval(k) * p.x)
                q = ti.Vector([c_ * p.y - s_ * p.z, s_ * p.y + c_ * p.z, p.x])
                return self(q, t)

            # @ti.func
            def color(p, t):
                c_ = ti.cos(eval(k) * p.x)
                s_ = ti.sin(eval(k) * p.x)
                q = ti.Vector([c_ * p.y - s_ * p.z, s_ * p.y + c_ * p.z, p.x])
                return self.color(q, t)

        elif axis == "y":
            # @ti.func
            def sdf(p, t):
                c_ = ti.cos(eval(k) * p.y)
                s_ = ti.sin(eval(k) * p.y)
                q = ti.Vector([c_ * p.x - s_ * p.z, s_ * p.x + c_ * p.z, p.y])
                return self(q, t)

            # @ti.func
            def color(p, t):
                c_ = ti.cos(eval(k) * p.y)
                s_ = ti.sin(eval(k) * p.y)
                q = ti.Vector([c_ * p.x - s_ * p.z, s_ * p.x + c_ * p.z, p.y])
                return self.color(q, t)

        if axis == "z":
            # @ti.func
            def sdf(p, t):
                c_ = ti.cos(eval(k) * p.z)
                s_ = ti.sin(eval(k) * p.z)
                q = ti.Vector([c_ * p.x - s_ * p.y, s_ * p.x + c_ * p.y, p.z])
                return self(q, t)

            # @ti.func
            def color(p, t):
                c_ = ti.cos(eval(k) * p.z)
                s_ = ti.sin(eval(k) * p.z)
                q = ti.Vector([c_ * p.x - s_ * p.y, s_ * p.x + c_ * p.y, p.z])
                return self.color(q, t)

        return Shape(
            sdf,
            color,
            **{k: v for k, v in self.__dict__.items() if k not in ["sdf", "color"]},
        )

    def bend(self, k=10.0, axis="x"):

        if isinstance(k, Number):
            k = str(k)

        if axis == "x":
            # @ti.func
            def sdf(p, t):
                x, y, z = p
                c = ti.cos(eval(k) * x)
                s = ti.sin(eval(k) * x)
                q = ti.Vector([c * x - s * y, s * x + c * y, p.z])
                return self(q, t)

            # @ti.func
            def color(p, t):
                x, y, z = p
                c = ti.cos(eval(k) * x)
                s = ti.sin(eval(k) * x)
                q = ti.Vector([c * x - s * y, s * x + c * y, p.z])
                return self.color(q, t)

        elif axis == "y":
            # @ti.func
            def sdf(p, t):
                x, y, z = p
                c = ti.cos(eval(k) * y)
                s = ti.sin(eval(k) * y)
                q = ti.Vector([c * x + s * y, -s * x + c * y, p.z])
                return self(q, t)

            # @ti.func
            def color(p, t):
                x, y, z = p
                c = ti.cos(eval(k) * y)
                s = ti.sin(eval(k) * y)
                q = ti.Vector([c * x - s * y, s * x + c * y, p.z])
                return self.color(q, t)

        elif axis == "z":
            # @ti.func
            def sdf(p, t):
                x, y, z = p
                c = ti.cos(eval(k) * z)
                s = ti.sin(eval(k) * z)
                q = ti.Vector([c * x - s * y, s * x + c * y, z])
                return self(q, t)

            # @ti.func
            def color(p, t):
                x, y, z = p
                c = ti.cos(eval(k) * y)
                s = ti.sin(eval(k) * y)
                q = ti.Vector([c * x - s * y, s * x + c * y, p.z])
                return self.color(q, t)

        return Shape(sdf, color)

    def repeat(self, c, l):

        # if type(c) != ti.Vector: c = ti.Vector(c);
        # if type(l) != ti.Vector: l = ti.Vector(l);

        if type(c) != str:
            c = str(c)
        if type(l) != str:
            l = str(l)

        def sdf(p, t):
            c_ = ti.Vector(list(eval(c)))
            l_ = ti.Vector(list(eval(l)))
            return self.sdf(p - c_ * tg.scalar.clamp(ti.round(p / c_), -l_, l_), t)

        return Shape(
            sdf,
            self.color,
            **{k: v for k, v in self.__dict__.items() if k not in ["sdf", "color"]},
        )

    def rotate(self, angle, axis="y"):

        if type(angle) != str:
            angle = str(angle)

        rot = {"x": "rotate_x", "y": "rotate_y", "z": "rotate_z"}[axis]

        return Shape(
            sdf=self.augment_locals(f'self.sdf({rot}(p, eval("{angle}")), t)'),
            color=self.augment_locals(
                f'self.color({rot}(p, eval("{angle}")), t)'),
            texture=lambda i, x, y: self.texture(i, *rot(p, eval(angle)).xy),
            **{
                k: v
                for k, v in self.__dict__.items()
                if k not in ["sdf", "color", "texture"]
            },
        )

    def isometric(self):
        return self.rotate(pi / 4, "x").rotate(atan2(1, sqrt(2)), "y")

    def onion(self, n=60, thickness=0.4, n2 = None):

        if n2 is None:
            n2 = n

        if type(n) != str:
            n = str(n)
        if type(n2) != str:
            n2 = str(n2)
        if type(thickness) != str:
            thickness = str(thickness)

        @ti.func
        def sdf(p, t):
            f = self(p, t) * eval(n2)
            i = int(f)
            if f < 0:
                if i % 2 == 1:
                    f -= ti.floor(f)
                else:
                    f = ti.floor(f) + 1 - f
            f = (f - eval(thickness)) / eval(n)
            return f

        return Shape(
            sdf,
            self.color,
            **{k: v for k, v in self.__dict__.items() if k not in ["sdf", "color"]},
        )

    def make_nested(self):
        
        @ti.func
        def sdf(p, t):
            f = self.sdf(p,t)
            f = f**2 * 10
            i = int(f)
            if f < 0:
                if i % 2 == 1:
                    f -= ti.floor(f)
                else:
                    f = ti.floor(f) + 1 - f
            f = (f - 0.4) / 60
            return f
        
        return Shape(
            sdf,
            self.color,
            **{k: v for k, v in self.__dict__.items() if k not in ["sdf", "color"]},
        )
    