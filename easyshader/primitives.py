import numpy as np
import taichi as ti

# import taichi_glsl as tg
from numbers import Number
from typing import Iterable, Callable, Union
from .shape import Shape
import taichi_glsl as tg


class Box(Shape):
    """

    .. code-block::
        :caption: Example: Box with side length = 1

            input: Box(1,'#e63746').isometric()
            output:

    .. image:: ../source/_static/box.png
        :scale: 40 %
        :alt: alternate text
        :align: center
    """

    def __init__(
        self, size: Union[Iterable, str], color: Union[Callable, str] = "#fff", **kwargs
    ):
        """
        :param size: Box size. Can be either a scalar value or a tuple (lx,ly,lz) with the box dimensions in the x,y,z axes
        :type size: Union[Iterable,str]
        :param color: Box color. Can be either 1) a string encoding a hexadecimal color (example: "#fff"), 2) a color-computing function that should return a 3-uple (example: lambda p,t: (p.x,p.y,p.z)) or 3) a string expression (example: "(x,y,z)"). defaults to "#fff"
        :type color: Union[Callable,str], optional
        :return: Box object.
        :rtype: Box
        """

        # Init parent class
        super().__init__(color=color, **kwargs)

        #
        if type(size) != ti.Vector:
            if not isinstance(size, Iterable):
                size = 3 * [size]
            size = ti.Vector(size)

        @ti.func
        def box(p, t):
            q = abs(p) - size
            return ti.Vector([max(0, q[0]), max(0, q[1]), max(0, q[2])]).norm() + min(
                q.max(), 0
            )

        self.sdf = box


class Sphere(Shape):
    """
    .. code-block::
        :caption: Example: Sphere with radius = 1

            input: Sphere(1,'#e63746')
            output:

    .. image:: ../source/_static/sphere.png
        :scale: 40 %
        :alt: alternate text
        :align: center
    """

    def __init__(
        self, radius: Union[Number, str], color: Union[Callable, str] = "#fff", **kwargs
    ):
        """
        :param radius: Sphere radius. Can be either a number or a string
        :type radius: Union[Number, str]
        :param color: Box color. Can be either 1) a string encoding a hexadecimal color (example: "#fff"), 2) a color-computing function that should return a 3-uple (example: lambda p,t: (p.x,p.y,p.z)) or 3) a string expression (example: "(x,y,z)"). defaults to "#fff"
        :type color: Union[Callable,str], optional
        :return: Sphere object
        :rtype: Sphere
        """

        super().__init__(color=color, **kwargs)

        if type(radius) != str:
            radius = str(radius)

        @ti.func
        def sphere(p, t):
            return p.norm() - eval(radius)

        self.sdf = sphere


class Line(Shape):
    """
    .. code-block::
        :caption: Example: Line from (-.6,-.6,0) to (.6,.6,0) with radius = .3

            input: Line((-.6,-.6,0),(.6,.6,0),.3,'#e63746')
            output:

    .. image:: ../source/_static/line.png
            :scale: 40 %
            :alt: alternate text
            :align: center
    """

    def __init__(
        self,
        p0: Iterable,
        p1: Iterable,
        r: Number,
        color: Union[Callable, str] = "#fff",
        **kwargs
    ):
        """
        :param p0: Initial point. Should be a 3-uple.
        :type p0: Iterable
        :param p1: Final point. Should be a 3-uple.
        :type p1: Iterable
        :param r: Line radius.
        :type r: Number
        :param color: Line color. Can be either 1) a string encoding a hexadecimal color (example: "#fff"), 2) a color-computing function that should return a 3-uple (example: lambda p,t: (p.x,p.y,p.z)) or 3) a string expression (example: "(x,y,z)"). defaults to "#fff"
        :type color: Union[Callable, str], optional
        :return: Line object
        :rtype: Line
        """

        super().__init__(color=color, **kwargs)

        if type(p0) != ti.Vector:
            p0 = ti.Vector(np.array(p0))
        if type(p1) != ti.Vector:
            p1 = ti.Vector(np.array(p1))

        @ti.func
        def line_(p, t):
            pa = p - p0
            ba = p1 - p0
            h = tg.scalar.clamp(pa.dot(ba) / ba.dot(ba), 0.0, 1.0)
            return (pa - ba * h).norm() - r

        self.sdf = line_


class Cyllinder(Shape):
    """
    .. code-block::
        :caption: Example: Line from (-.6,-.6,0) to (.6,.6,0) with radius = .3

            input: Line((-.6,-.6,0),(.6,.6,0),.3,'#e63746')
            output:

    .. image:: ../source/_static/line.png
            :scale: 40 %
            :alt: alternate text
            :align: center
    """

    def __init__(
        self,
        radius: Number,
        height: Number,
        color: Union[Callable, str] = "#fff",
        **kwargs
    ):
        """
        :param p0: Initial point. Should be a 3-uple.
        :type p0: Iterable
        :param p1: Final point. Should be a 3-uple.
        :type p1: Iterable
        :param r: Line radius.
        :type r: Number
        :param color: Line color. Can be either 1) a string encoding a hexadecimal color (example: "#fff"), 2) a color-computing function that should return a 3-uple (example: lambda p,t: (p.x,p.y,p.z)) or 3) a string expression (example: "(x,y,z)"). defaults to "#fff"
        :type color: Union[Callable, str], optional
        :return: Line object
        :rtype: Line
        """

        super().__init__(color=color, **kwargs)

        @ti.func
        def cyllinder(p, t):
            # return (p.xz - c.xy).norm() - c.z
            d = abs(ti.Vector([(p.xz).norm(), p.y])) - ti.Vector([radius, height])
            return min(max(d.x, d.y), 0.0) + max(d, 0.0).norm()

        self.sdf = cyllinder


class Torus(Shape):
    """
    .. code-block::
        :caption: Example: Torus with outer radius = 1 and inner radius = .5

            input: Torus(1,.5,'#e63746').isometric()
            output:

    .. image:: ../source/_static/torus.png
            :scale: 40 %
            :alt: alternate text
            :align: center
    """

    def __init__(
        self,
        outer_radius: Number,
        inner_radius: Number,
        color: Union[Callable, str] = "#fff",
        **kwargs
    ):
        """
        :param outer_radius: Torus outer radius.
        :type outer_radius: Number
        :param inner_radius: Torus inner radius.
        :type inner_radius: Number
        :param color: Torus color. Can be either 1) a string encoding a hexadecimal color (example: "#fff"), 2) a color-computing function that should return a 3-uple (example: lambda p,t: (p.x,p.y,p.z)) or 3) a string expression (example: "(x,y,z)"). defaults to "#fff"
        :type color: Union[Callable, str], optional
        :return: _description_
        :rtype: _type_
        """

        super().__init__(color=color, **kwargs)

        @ti.func
        def torus(p, t):
            q = ti.Vector([(p.xy).norm() - outer_radius, p.z])
            return q.norm() - inner_radius

        self.sdf = torus


class BoxFrame(Shape):
    """
    .. code-block::
        :caption: Example: BoxFrame with side length = 1 and thickness = .2

            input: BoxFrame(1,.2,'#e63746').isometric()
            output:

    .. image:: ../source/_static/boxframe.png
            :scale: 40 %
            :alt: alternate text
            :align: center
    """

    def __init__(
        self,
        size: Number,
        thickness: Number,
        color: Union[Callable, str] = "#fff",
        **kwargs
    ):
        """
        :param size: Side length.
        :type size: Number
        :param thickness: Thickness.
        :type thickness: Number
        :param color: BoxFrame color. Can be either 1) a string encoding a hexadecimal color (example: "#fff"), 2) a color-computing function that should return a 3-uple (example: lambda p,t: (p.x,p.y,p.z)) or 3) a string expression (example: "(x,y,z)"). defaults to "#fff"
        :type color: Union[Callable,str], optional
        :return: BoxFrame Shape.
        :rtype: BoxFrame
        """

        super().__init__(color=color, **kwargs)

        if type(size) != str:
            size = str(size)
        if type(thickness) != str:
            thickness = str(thickness)

        # @ti.func
        def box(p, t):

            size_ = eval(size)
            thickness_ = eval(thickness)
            if not isinstance(size_, Iterable):
                b_ = [size_, size_, size_]
            if not isinstance(thickness_, Iterable):
                thickness_ = [thickness_, thickness_, thickness_]

            p = abs(p) - ti.Vector(b_)
            q = abs(p + ti.Vector(thickness_)) - ti.Vector(thickness_)
            return ti.min(
                ti.min(
                    (ti.max(ti.Vector([p.x, q.y, q.z]), 0.0)).norm()
                    + ti.min(ti.max(p.x, ti.max(q.y, q.z)), 0.0),
                    (ti.max(ti.Vector([q.x, p.y, q.z]), 0.0)).norm()
                    + ti.min(ti.max(q.x, ti.max(p.y, q.z)), 0.0),
                ),
                (ti.max(ti.Vector([q.x, q.y, p.z]), 0.0)).norm()
                + ti.min(ti.max(q.x, ti.max(q.y, p.z)), 0.0),
            )

        self.sdf = box


class Octahedron(Shape):
    """
    .. code-block::
        :caption: Example: Octagon with (circumscribed sphere) radius = 1

            input: Octagon(1,'#e63746').isometric()
            output:

    .. image:: ../source/_static/octagon.png
            :scale: 40 %
            :alt: alternate text
            :align: center
    """

    def __init__(self, radius: Number, color: Union[Callable, str] = "#fff", **kwargs):
        """
        :param radius: (Circunscribed sphere) radius.
        :type radius: Number
        :param color: Octagon color. Can be either 1) a string encoding a hexadecimal color (example: "#fff"), 2) a color-computing function that should return a 3-uple (example: lambda p,t: (p.x,p.y,p.z)) or 3) a string expression (example: "(x,y,z)"). defaults to "#fff"
        :type color: Union[Callable,str], optional
        :return: Octagon Shape
        :rtype: Octagon
        """

        super().__init__(color=color, **kwargs)

        @ti.func
        def octa(p, t):
            p = abs(p)
            return (p.x + p.y + p.z - radius) * 0.57735027

        self.sdf = octa


class Icosahedron(Shape):
    """
    .. code-block::
        :caption: Example: Icosahedron with (circumscribed sphere) radius = 1

            input: Icosahedron(1,'#e63746').isometric()
            output:

    .. image:: ../source/_static/icosahedron.png
            :scale: 40 %
            :alt: alternate text
            :align: center
    """

    def __init__(self, radius: Number, color: Union[Callable, str] = "#fff", **kwargs):
        """
        :param radius: (Circunscribed sphere) radius.
        :type radius: Number
        :param color: Octagon color. Can be either 1) a string encoding a hexadecimal color (example: "#fff"), 2) a color-computing function that should return a 3-uple (example: lambda p,t: (p.x,p.y,p.z)) or 3) a string expression (example: "(x,y,z)"). defaults to "#fff"
        :type color: Union[Callable,str], optional
        :return: Octagon Shape
        :rtype: Octagon
        """

        super().__init__(color=color, **kwargs)

        @ti.func
        def icosa(p, t):
            G = ti.sqrt(5.0) * 0.5 + 0.5
            n = ti.Vector([G, 1.0 / G, 0.0]).normalized()
            d = 0.0
            p = ti.abs(p)
            d = ti.max(d, p.dot(n))
            d = ti.max(d, p.dot(ti.Vector([n[1], n[2], n[0]])))
            d = ti.max(d, p.dot(ti.Vector([n[2], n[0], n[1]])))
            d = ti.max(d, p.dot(ti.Vector([1.0, 1.0, 1.0]).normalized()))
            return d - radius

        self.sdf = icosa


class Cone(Shape):
    """
    .. code-block::
        :caption: Example: Octagon with (circumscribed sphere) radius = 1

            input: Octagon(1,'#e63746').isometric()
            output:

    .. image:: ../source/_static/octagon.png
            :scale: 40 %
            :alt: alternate text
            :align: center
    """

    def __init__(
        self,
        radius: Number,
        height: Number,
        color: Union[Callable, str] = "#fff",
        **kwargs
    ):
        """
        :param radius: Radius of the base of the cone.
        :type radius: Number
        :param height: Height of the cone.
        :type height: Number
        :param color: Cone color. Can be either 1) a string encoding a hexadecimal color (example: "#fff"), 2) a color-computing function that should return a 3-uple (example: lambda p,t: (p.x,p.y,p.z)) or 3) a string expression (example: "(x,y,z)"). defaults to "#fff"
        :type color: Union[Callable,str], optional
        :return: Cone Shape
        :rtype: Cone
        """

        super().__init__(color=color, **kwargs)

        @ti.func
        def cone(p, t):
            q = ti.Vector([radius, height])
            w = ti.Vector([(p.xz).norm(), height / 2 - p.y])
            a = w - q * tg.clamp(w.dot(q) / q.dot(q), 0.0, 1.0)
            b = w - q * ti.Vector([tg.clamp(w.x / q.x, 0.0, 1.0), 1.0])
            k = tg.sign(q.y)
            d = min(a.dot(a), b.dot(b))
            s = max(k * (w.x * q.y - w.y * q.x), k * (w.y - q.y))

            return d ** (1 / 2) * tg.sign(s)

        self.sdf = cone


class Dodecahedron(Shape):
    """
    .. code-block::
        :caption: Example: Icosahedron with (circumscribed sphere) radius = 1

            input: Icosahedron(1,'#e63746').isometric()
            output:

    .. image:: ../source/_static/icosahedron.png
            :scale: 40 %
            :alt: alternate text
            :align: center
    """

    def __init__(self, radius: Number, color: Union[Callable, str] = "#fff", **kwargs):
        """
        :param radius: (Circunscribed sphere) radius.
        :type radius: Number
        :param color: Octagon color. Can be either 1) a string encoding a hexadecimal color (example: "#fff"), 2) a color-computing function that should return a 3-uple (example: lambda p,t: (p.x,p.y,p.z)) or 3) a string expression (example: "(x,y,z)"). defaults to "#fff"
        :type color: Union[Callable,str], optional
        :return: Octagon Shape
        :rtype: Octagon
        """

        super().__init__(color=color, **kwargs)

        @ti.func
        def dodeca(p, t):
            G = ti.sqrt(5.0) * 0.5 + 0.5
            n = ti.Vector([G, 1.0, 0.0])
            n /= n.norm()
            d = 0.0
            p = abs(p)
            d = max(d, p.dot(n))
            d = max(d, p.dot(ti.Vector([n[1], n[2], n[0]])))
            d = max(d, p.dot(ti.Vector([n[2], n[0], n[1]])))
            return d - radius

        self.sdf = dodeca


class Plane(Shape):
    """
    .. code-block::
        :caption: Example: Plane with normal vector (0, 1, 0) and distance from origin 0

            input: Plane((0, 1, 0), 0, '#e63746')
            output:

    .. image:: ../source/_static/plane.png
        :scale: 40 %
        :alt: alternate text
        :align: center
    """

    def __init__(
        self,
        normal: Iterable,
        distance: Number,
        color: Union[Callable, str] = "#fff",
        **kwargs
    ):
        """
        :param normal: Normal vector of the plane.
        :type normal: Iterable
        :param distance: Distance from the origin.
        :type distance: Number
        :param color: Plane color. Can be either 1) a string encoding a hexadecimal color (example: "#fff"), 2) a color-computing function that should return a 3-uple (example: lambda p,t: (p.x,p.y,p.z)) or 3) a string expression (example: "(x,y,z)"). defaults to "#fff"
        :type color: Union[Callable, str], optional
        :return: Plane Shape
        :rtype: Plane
        """

        super().__init__(color=color, **kwargs)

        if type(normal) != ti.Vector:
            normal = ti.Vector(np.array(normal))

        @ti.func
        def plane(p, t):
            return p.dot(normal) + distance

        self.sdf = plane


class Gyroid(Shape):
    """
    .. code-block::
        :caption: Example: Gyroid with scale = 1

            input: Gyroid(1, '#e63746').isometric()
            output:

    .. image:: ../source/_static/gyroid.png
        :scale: 40 %
        :alt: alternate text
        :align: center
    """

    def __init__(self, scale: Number, color: Union[Callable, str] = "#fff", **kwargs):
        """
        :param scale: Scale of the gyroid.
        :type scale: Number
        :param color: Gyroid color. Can be either 1) a string encoding a hexadecimal color (example: "#fff"), 2) a color-computing function that should return a 3-uple (example: lambda p,t: (p.x,p.y,p.z)) or 3) a string expression (example: "(x,y,z)"). defaults to "#fff"
        :type color: Union[Callable, str], optional
        :return: Gyroid Shape
        :rtype: Gyroid
        """

        super().__init__(color=color, **kwargs)

        @ti.func
        def gyroid(p, t):
            return (
                ti.sin(p.x * scale) * ti.cos(p.y * scale)
                + ti.sin(p.y * scale) * ti.cos(p.z * scale)
                + ti.sin(p.z * scale) * ti.cos(p.x * scale)
                + 1
            )

        self.sdf = gyroid
