"""
.. module:: shape
   :synopsis: Implements the easyshaper.Shape class
.. moduleauthor:: Marcelo Prates <github.com/marceloprates>
"""

import os
import re
import subprocess
from copy import deepcopy
from numbers import Number
from typing import Callable, Iterable, List, Union

import trimesh
import cv2
import IPython
import numpy as np

# import open3d as o3d
import PIL
import taichi as ti
from numpy import pi
import taichi_glsl as tg

# import vsketch
from infix import shift_infix as infix
from IPython.display import clear_output
from matplotlib import pyplot as plt
from numpy import pi as π
from numpy.linalg import norm
from PIL import Image
from shapely.geometry import MultiPolygon, Polygon
from skimage.measure import find_contours, marching_cubes
from sklearn.cluster import KMeans
from taichi import *
from taichi_glsl import clamp, mix

from easyshader import Light, Rendering

from .transformations import *
from .utils import *
from scipy.ndimage import gaussian_filter
import open3d as o3d
from .camera import Camera


@ti.data_oriented
class Shape:
    def __init__(
        self,
        sdf: Union[Callable, str] = "z",
        color: Union[Callable, str] = "#fff",
        textures: List[Union[str, PIL.Image.Image, np.ndarray]] = [],
        palette: List[Union[str, Iterable[Number]]] = ["#000", "#fff"],
        background_color: any = None,
        text: any = None,
        verbose: bool = False,
        **kwargs,
    ):
        """
        Generic "Shape" class.

        :param sdf: A Signed Distance Function (SDF). Can be either a function or a string. defaults to "z"
        :type sdf: Union[Callable, str], optional
        :param color: _description_, defaults to "#fff"
        :type color: Union[Callable, str], optional
        :param textures: _description_, defaults to []
        :type textures: List[Union[str, PIL.Image.Image, np.ndarray]], optional
        :param palette: _description_, defaults to ["#000", "#fff"]
        :type palette: List[Union[str, Iterable[Number]]], optional
        :param rendering_kwargs: _description_, defaults to {}
        :type rendering_kwargs: dict, optional
        :param background_color: _description_, defaults to None
        :type background_color: any, optional
        :param animated: _description_, defaults to None
        :type animated: bool, optional
        :param text: _description_, defaults to None
        :type text: any, optional
        """

        # Set attributes
        self.__name__ = ""
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

        self.depth_resolution = 2000
        resolution = [self.depth_resolution] * 2
        self.depth = ti.field(ti.f32, shape=resolution)

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

            # h, w = Rendering(scene=Shape(), **self.rendering_kwargs).resolution
            h, w = Camera().rendering_kwargs["resolution"]
            dpi = 300
            fig, ax = plt.subplots(figsize=(h / dpi, w / dpi), dpi=dpi)
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
                    ti.Vector(
                        [round(ti.cast(i, ti.f32)) % self.palette_field.shape[0], 0]
                    ),
                )

            # 'texture' function for indexing textures
            def texture(i, x, y):
                h, w = self.texture_fields[i].shape
                return tg.bilerp(
                    self.texture_fields[i], ti.Vector([(-x + 0.5) * w, (-y + 0.5) * h])
                )

            # 'text' function for indexing text texture
            def text(x, y):
                h, w = self.text_field.shape
                return tg.bilerp(
                    self.text_field, ti.Vector([(-x + 0.5 * h / w) * w, (-y + 0.5) * w])
                )

            for attr in self.__dict__.keys():
                exec(f'{attr} = self.__dict__["{attr}"]')

            return eval(func) if type(func) == str else func(p, t)

        return inner

    def paint(self, color=None, palette=None, textures=None, inplace=False, **kwargs):

        shape = Shape(
            sdf=self.sdf,
            **(dict(color=color) if color is not None else {}),
            **(dict(palette=palette) if palette is not None else {}),
            **(dict(textures=textures) if textures is not None else {}),
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

    def GUI(self, **rendering_kwargs):

        rendering_kwargs = rendering_kwargs

        background = Shape(
            f"z+{global_background['distance']}", global_background["color"]
        )

        # Create rendering object
        rendering = Rendering(
            scene=(background + self),
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

    def render(
        self,
        t=0,
        save=False,
        depth=False,
        background="#aaa",
        background_distance=2,
        verbose=None,
        **kwargs,
    ):
        img = self.render_static(
            t=t,
            depth=depth,
            verbose=verbose,
            background=background,
            background_distance=background_distance,
            **kwargs,
        )
        if save:
            img.save(save)
        return img

    def render_static(
        self,
        t=0,
        save=False,
        lights=[Light()],
        depth=False,
        verbose=None,
        background_distance=2,
        background=None,
        **camera_kwargs,
    ):

        camera = Camera(camera_pos=(0, 0, 8))

        # Overwrite camera rendering_kwargs with "camera_kwargs" keyword arguments
        for arg, value in camera_kwargs.items():
            camera.rendering_kwargs[arg] = value

        return camera.snap(
            (
                (Shape(f"z+{background_distance}", background) + self)
                if background is not None
                else self
            ),
            lights=lights,
            t=t,
            depth=depth,
            verbose=verbose if verbose else self.verbose,
        )

    def animate(
        self,
        camera=Camera(),
        lights=[Light()],
        background="#fff",
        background_distance=2,
        save=False,
        resume=False,
        frames=None,
        framerate=30,
        **camera_kwargs,
    ):

        # Overwrite camera rendering_kwargs with "camera_kwargs" keyword arguments
        for arg, value in camera_kwargs.items():
            camera.rendering_kwargs[arg] = value
        # Overwrite frames and framerate parameters (if not None)
        for arg in ["frames", "framerate"]:
            if eval(arg) is not None:
                camera.rendering_kwargs[arg] = eval(arg)
        background = Shape(f"z+{background_distance}", background)
        return camera.record(
            (background + self) if background is not None else self,
            lights=lights,
            resume=resume,
            **(dict(path=save) if save else dict()),
        )

    def animate_ascii(
        self,
        path=".tmp/output.gif",
        **rendering_kwargs,
    ):

        if "resolution" not in rendering_kwargs:
            rendering_kwargs["resolution"] = (80, 80)

        (
            rendering_kwargs.pop("background_color")
            if "background_color" in rendering_kwargs
            else self.background_color
        )
        frames = rendering_kwargs.pop("frames") if "frames" in rendering_kwargs else 20
        (rendering_kwargs.pop("framerate") if "framerate" in rendering_kwargs else 30)

        # Create rendering object
        rendering = Rendering(
            scene=Shape("z+4", color="#fff") + self,
            **rendering_kwargs,
        )

        # Create animation
        while True:
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

                clear_output()
                print(img2ascii(result))

    def save(self, path, display=True):

        # Render static scene and save as temporary file
        result = self.render()
        result.save(path)

        if display:
            IPython.display.display(
                IPython.display.Image(data=open(path, "rb").read(), format="png")
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
            MultiPolygon([Polygon(c) for c in find_contours(labels, level=level + 0.5)])
            for level in sorted(
                range(n_colors),
                key=lambda i: norm(kmeans.cluster_centers_[i]),
                reverse=True,
            )
        ]

        return color_layers

    """
    def pen_plotter(self, n_colors=2, scale=1, save=None, rendering_kwargs=None):

        self.rendering_kwargs = rendering_kwargs

        vecs = self.vectorize(n_colors)

        class MySketch(vsketch.SketchClass):
            def draw(self, vsk: vsketch.Vsketch) -> None:
                vsk.size("a4", landscape=True)
                vsk.scale(scale, scale)

                for i, vec in enumerate(vecs):
                    vsk.stroke(i + 1)  # vsk.fill(i+1)
                    if i % 2 == 1:
                        vsk.fill(i + 1)
                    vsk.geometry(vec)

                if save is not None:
                    vsk.save(save)

            def finalize(self, vsk: vsketch.Vsketch) -> None:
                vsk.vpype("linemerge linesimplify reloop linesort")

        MySketch().display()
    """

    def to_mesh(
        self,
        resolution=400,
        level=0.0,
        save_path="output.ply",
        preview=True,
        simplify=None,
    ):
        """
        Generates a mesh using the marching cubes algorithm and saves it as an OBJ file.

        :param resolution: The resolution of the grid for the marching cubes algorithm.
        :param level: The level value to use for the isosurface.
        :param save_path: The path to save the OBJ file.
        """
        # Create a grid of points
        x = ti.field(dtype=ti.f32, shape=resolution)
        y = ti.field(dtype=ti.f32, shape=resolution)
        z = ti.field(dtype=ti.f32, shape=resolution)
        s = 10
        x.from_numpy(np.linspace(-s, s, resolution, dtype=np.float32))
        y.from_numpy(np.linspace(-s, s, resolution, dtype=np.float32))
        z.from_numpy(np.linspace(-s, s, resolution, dtype=np.float32))
        grid = np.meshgrid(x.to_numpy(), y.to_numpy(), z.to_numpy(), indexing="ij")
        points = np.stack(grid, axis=-1).reshape(-1, 3)

        # Evaluate the SDF at each point
        sdf_values = ti.field(dtype=ti.f32, shape=(resolution, resolution, resolution))

        @ti.kernel
        def compute_sdf():
            for i, j, k in ti.ndrange(resolution, resolution, resolution):
                p = ti.Vector([x[i], y[j], z[k]])
                sdf_values[i, j, k] = self.sdf(p, 0)

        compute_sdf()

        # Apply Gaussian smoothing to the SDF values
        sdf_np = sdf_values.to_numpy()
        # sdf_smoothed = gaussian_filter(sdf_np, sigma=0.5)

        # Use marching cubes to extract the mesh
        verts, faces, normals, values = marching_cubes(sdf_np, level=level)

        # Convert verts to a Taichi field
        verts_field = ti.Vector.field(3, dtype=ti.f32, shape=verts.shape[0])
        verts_field.from_numpy(verts)

        # Get colors for each vertex
        colors = ti.Vector.field(3, dtype=ti.f32, shape=verts.shape[0])

        @ti.kernel
        def compute_colors():
            for i in range(verts_field.shape[0]):
                colors[i] = self.color(verts_field[i], 0)

        compute_colors()
        colors_np = 255 * colors.to_numpy()

        # Transform vertices back to original coordinates
        verts = (verts / (resolution - 1)) * 2 - 1

        # Create a trimesh object
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_colors=colors_np)

        # Save the mesh as a PLY file
        mesh.export(save_path, file_type="obj")

        # Load the mesh from the .ply file using Open3D
        mesh = o3d.io.read_triangle_mesh(save_path)

        if simplify:
            # Simplify the mesh
            simplified_mesh = mesh.simplify_quadric_decimation(simplify)
            # Save the simplified mesh
            o3d.io.write_triangle_mesh(save_path, simplified_mesh)

        if preview:
            mesh = trimesh.load_mesh(save_path)

            scene = trimesh.Scene(mesh)

            return scene.show()

    # Binary operators

    def __and__(self, o):
        """

        Computes the intersection between two Shapes, f and g, expressed as Signed Distance Functions (SDF).

        The intersection is computed as:

        .. math::
            (f \& g)(p) = \max(f(p),g(p))

        :param other: A Shape expressed as a SDF
        :type other: Shape
        :return: Intersection between 'self' and 'other'
        :rtype: Shape

        .. code-block::
            :caption: Example: Intersection between sphere and a half-plane

                input: Sphere(1) & Shape('x')
                output:

        .. image:: ../source/_static/and.png
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
        """

        Computes:

        - if **type(g) == Shape**: The union of two Shapes f and g.

        The union is computed as:

        .. math::
            (f + g)(p) = \min(f(p),g(p))

        .. code-block::
            :caption: Example: Union between a cyan sphere and a magenta box

                input: Sphere(1,'#2ff') + Box(.7,'#f2f').isometric()
                output:

        .. image:: ../source/_static/add.png
            :scale: 60 %
            :alt: alternate text
            :align: center

        - if g is a string formatted as '(dx,dy,dz)': Translation of the Shape f with displacement in the x, y and z axes given by dx,dy,dz respectively.

        The translation is computed as:

        .. math::
            (f + (dx,dy,dz))(p) = f(p+(dx,dy,dz))

        .. code-block::
            :caption: Example: Sphere translated .9 units in the x direction, 0 units in the y direction, 0 units in the z direction

                input: Sphere(.5) + '(.9,0,0)'
                output:

        .. image:: ../source/_static/translation.png
            :scale: 60 %
            :alt: alternate text
            :align: center


        :param other: A Shape expressed as a SDF
        :type other: Shape
        :return: Union between 'self' and 'other'
        :rtype: Shape


        """

        if isinstance(o, Iterable):
            o = str(o)

        if isinstance(o, str):
            if o.startswith("(") and o.endswith(")"):
                # Translation (syntax: shape + '(1,2,3)')
                sdf = self.augment_locals(f"self(p-ti.Vector(list({o})),t)")
                color = self.augment_locals(f"self.color(p+ti.Vector(list({o})),t)")
            elif any([o.startswith(f"d{axis}") for axis in "xyz"]):
                # Translation (syntax: shape + 'dx 1')
                # Get axis name
                axis = o[1]
                # Get distance
                d = o.split(f"d{axis}")[1]
                # Build distance vector
                if axis == "x":
                    dxyz = f"({d},0,0)"
                elif axis == "y":
                    dxyz = f"(0,{d},0)"
                else:
                    dxyz = f"(0,0,{d})"
                # Update sdf and color functions
                sdf = self.augment_locals(f"self(p-ti.Vector(list({dxyz})),t)")
                color = self.augment_locals(f"self.color(p+ti.Vector(list({dxyz})),t)")

            elif any([o.startswith(f"r{axis}") for axis in "xyz"]):
                # Rotation around axis (syntax: shape + 'rx pi/4')
                # Get axis name
                axis = o[1]
                # Get rotation angle
                angle = o.split(f"r{axis}")[1]
                # Return rotated version of 'self
                return self.rotate(angle, axis)
            else:
                # Displacement (syntax: shape + '.1*sin(x)')
                sdf = self.augment_locals(f"self(p,t) - {o}")
                color = self.augment_locals(f"self.color(p,t) - {o}")

        elif isinstance(o, Number):

            @ti.func
            def sdf(p, t):
                return self(p, t) - o

        elif isinstance(o, Shape):

            @ti.func
            def sdf(p, t):
                return min(self(p, t), o(p, t))

            @ti.func
            def color(p, t):
                return self.color(p, t) if (self(p, t) < o(p, t)) else o.color(p, t)

        else:
            raise Exception(f"You cannot add a Shape with a '{type(o)}'")

        return Shape(
            sdf,
            color,
            **{k: v for k, v in self.__dict__.items() if k not in ["sdf", "color"]},
        )

    def __sub__(self, o):
        """

        Computes the difference between two Shapes, f and g, expressed as Signed Distance Functions (SDF).

        The difference is computed as:

        .. math::
            (f - g)(p) = \max(f(p),-g(p))

        :param other: A Shape expressed as a SDF
        :type other: Shape
        :return: Intersection between 'self' and 'other'
        :rtype: Shape

        .. code-block::
            :caption: Example: Difference between sphere and a half-plane

                input: Box(.75).isometric() - Sphere(.9)
                output:

        .. image:: ../source/_static/sub.png
            :scale: 60 %
            :alt: alternate text
            :align: center

        """

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
        """

        Computes the intersection between two Shapes, f and g, expressed as Signed Distance Functions (SDF).

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

        .. image:: ../source/_static/and.png
            :scale: 60 %
            :alt: alternate text
            :align: center

        """

        if isinstance(o, str):
            sdf = self.augment_locals(f"self(p/{o},t)")
            color = self.color
        elif isinstance(o, Number):

            @ti.func
            def sdf(p, t):
                return o * self(p / o, t)

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

    def __truediv__(self, o: Number):
        """

        Computes the intersection between two Shapes, f and g, expressed as Signed Distance Functions (SDF).

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

        .. image:: ../source/_static/and.png
            :scale: 60 %
            :alt: alternate text
            :align: center

        """

        @ti.func
        def sdf(p, t):
            return self(p * o, t) / o

        return Shape(sdf, color=self.color)

    def __or__(self, o):
        """

        Applies transformation to input coordinates.

        The intersection is computed as:

        .. math::
            (g | f)(p) = f(g(p))

        :param other: A Shape expressed as a SDF
        :type other: Shape
        :return: SDF applied to transformed input coordinates
        :rtype: Shape

        .. code-block::
            :caption: Example: Applying the input coordinates transformation (x,y,z) -> (x+.25*cos(3*y),y+.25*sin(3*x),z) to a magenta boxframe

                input: '(x+.25*cos(3*y),y+.25*sin(3*x),z)' | BoxFrame(.9,.1).isometric()
                output:

        .. image:: ../source/_static/or.png
            :scale: 60 %
            :alt: alternate text
            :align: center

        """

        # sdf,color = self.sdf, self.color

        if isinstance(o, str):
            sdf = self.augment_locals(f"{o.replace('dist','self(p,t)')}")
            color = self.color
        # else:
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

        Computes the intersection between two Shapes, f and g, expressed as Signed Distance Functions (SDF).

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

        .. image:: ../source/_static/and.png
            :scale: 60 %
            :alt: alternate text
            :align: center

        """

        # sdf,color = self.sdf, self.color

        if isinstance(o, str):
            # sdf = self.augment_locals(f"")
            def sdf_(p, t):
                x, y, z = p
                p = ti.Vector(eval(o))
                return self.sdf(p, t)

            sdf = sdf_
            color = self.color
        # else:
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
        """

        Computes the smooth union of two Shapes, f and g.

        The union is computed as:

        .. math::
            d_1 = o(g,t)

            d_2 = f(p,t)

            h = clamp(0.5 + 0.5 (d_2 - d_1) k^{-1}, 0, 1)

            (f \ll su(k) \gg g)(p) = d_2 h + d_1 (1-h) - k h (1 - h)

        .. code-block::
            :caption: Example: Smooth union between a cyan sphere and a magenta box

                x = Sphere(.25,'#5ff') + '(+.5,0,0)'
                y = Box(.25,'#f5f').isometric() + '(-.5,0,0)'

                input: x <<su(.5)>> y
                output:

        .. code-block::
            :caption: Alternative syntax:

                x = Sphere(.25,'#5ff') + '(+.5,0,0)'
                y = Box(.25,'#f5f').isometric() + '(-.5,0,0)'

                input: x.smooth_union(y,.5)
                output:

        .. image:: ../source/_static/su.png
            :scale: 60 %
            :alt: alternate text
            :align: center

        :param o: _description_
        :type o: _type_
        :param k: _description_, defaults to 0.5
        :type k: float, optional
        :return: _description_
        :rtype: _type_
        """

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

    def twist(self, k, axis="y", justcolor=False):
        """

        Twists a shape along the y axis.

        The twist is computed as:

        .. math::
            twist(f,k) = f((\cos(k y) x - \sin(k y) z, \sin(k y) x + \cos(k y) z, y))

        .. code-block::
            :caption: Example: Twisted isometric box

                input: Box(.5).isometric().twist(4)
                output:

        .. image:: ../source/_static/twist.png
            :scale: 60 %
            :alt: alternate text
            :align: center

        :param o: _description_
        :type o: _type_
        :param k: _description_, defaults to 0.5
        :type k: float, optional
        :return: _description_
        :rtype: _type_
        """

        if type(k) != str:
            k = str(k)

        if axis == "x":
            # @ti.func
            def sdf(p, t):
                c_ = ti.cos(eval(k) * p.x)
                s_ = ti.sin(eval(k) * p.x)
                q = ti.Vector([c_ * p.y - s_ * p.z, s_ * p.y + c_ * p.z, p.x])
                return self.sdf(q, t)

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
                return self.sdf(q, t)

            # @ti.func
            def color(p, t):
                c_ = ti.cos(eval(k) * p.y)
                s_ = ti.sin(eval(k) * p.y)
                q = ti.Vector([c_ * p.x - s_ * p.z, s_ * p.x + c_ * p.z, p.y])
                return self.color(q, t)

        elif axis == "z":
            # @ti.func
            def sdf(p, t):
                c_ = ti.cos(eval(k) * p.z)
                s_ = ti.sin(eval(k) * p.z)
                q = ti.Vector([c_ * p.x - s_ * p.y, s_ * p.x + c_ * p.y, p.z])
                return self.sdf(q, t)

            # @ti.func
            def color(p, t):
                c_ = ti.cos(eval(k) * p.z)
                s_ = ti.sin(eval(k) * p.z)
                q = ti.Vector([c_ * p.x - s_ * p.y, s_ * p.x + c_ * p.y, p.z])
                return self.color(q, t)

        return Shape(
            sdf if not justcolor else self.sdf,
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
        """

        Repeats the Shape along the x,y,z axes with displacement given by **c** and bounds given by **l**.

        .. code-block::
            :caption: Example: Using 'repeat()' to create a 3x3x3 3D grid of spheres

                input: x = Sphere(.5,'#f0c04c').repeat((.9,.9,.9),(1,1,1)).isometric()
                output:

        .. image:: ../source/_static/repeat.png
            :scale: 60 %
            :alt: alternate text
            :align: center

        :param o: _description_
        :type o: _type_
        :param k: _description_, defaults to 0.5
        :type k: float, optional
        :return: _description_
        :rtype: _type_
        """

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
            color=self.augment_locals(f'self.color({rot}(p, eval("{angle}")), t)'),
            texture=lambda i, x, y: self.texture(i, *rot(p, eval(angle)).xy),
            **{
                k: v
                for k, v in self.__dict__.items()
                if k not in ["sdf", "color", "texture"]
            },
        )

    def isometric(self):
        return self.rotate(pi / 4, "x").rotate(atan2(1, sqrt(2)), "y")

    def onion(self, n=60, thickness=0.4, n2="10"):
        """

        Slices the Shape concentrically, creating an "onion" Shape.

        .. code-block::
            :caption: Example: Creating a "sliced onion" from a sphere

                input: (Sphere(1).onion(60,.2) & Shape('z-.2')).isometric()
                output:

        .. image:: ../source/_static/onion.png
            :scale: 60 %
            :alt: alternate text
            :align: center

        :param o: _description_
        :type o: _type_
        :param k: _description_, defaults to 0.5
        :type k: float, optional
        :return: _description_
        :rtype: _type_
        """

        if n2 is None:
            n2 = n

        if type(n) != str:
            n = str(n)
        if type(n2) != str:
            n2 = str(n2)
        if type(thickness) != str:
            thickness = str(thickness)

        n, n2, thickness = map(self.augment_locals, [n, n2, thickness])

        @ti.func
        def sdf(p, t):
            f = self(p, t) * n2(p, t)
            i = int(f)
            if f < 0:
                if i % 2 == 1:
                    f -= ti.floor(f)
                else:
                    f = ti.floor(f) + 1 - f
            f = (f - thickness(p, t)) / n(p, t)
            return f

        return Shape(
            sdf,
            self.color,
            **{k: v for k, v in self.__dict__.items() if k not in ["sdf", "color"]},
        )

    def make_nested(self):
        @ti.func
        def sdf(p, t):
            f = self.sdf(p, t)
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

    def with_background(self, background="#fff", background_distance=2):
        return Shape(f"z+{background_distance}", color=background) + self


def su(k=1):
    @infix
    def su_(x, y):
        return x.smooth_union(y, k=k)

    return su_


def si(k=1):
    @infix
    def si_(x, y):
        return x.smooth_intersection(y, k=k)

    return si_


def sd(k=1):
    @infix
    def sd_(x, y):
        return x.smooth_difference(y, k=k)

    return sd_
