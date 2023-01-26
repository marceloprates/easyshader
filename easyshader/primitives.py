from .shape import *


class Box(Shape):
    def __init__(self, s, color="#fff", **kwargs):

        super().__init__(color=color, **kwargs)

        if type(s) != ti.Vector:
            if not isinstance(s, Iterable):
                s = 3 * [s]
            s = ti.Vector(s)

        @ti.func
        def box(p, t):
            q = abs(p) - s
            return ti.Vector([max(0, q[0]), max(0, q[1]), max(0, q[2])]).norm() + min(
                q.max(), 0
            )

        self.sdf = box


class Octa(Shape):
    def __init__(self, s, color="#fff", **kwargs):

        super().__init__(color=color, **kwargs)

        @ti.func
        def octa(p, t):
            p = abs(p)
            return (p.x + p.y + p.z - s) * 0.57735027

        self.sdf = octa


class BoxFrame(Shape):
    def __init__(self, b, e, color="#fff", **kwargs):

        super().__init__(color=color, **kwargs)

        if type(b) != str:
            b = str(b)
        if type(e) != str:
            e = str(e)

        # @ti.func
        def box(p, t):

            b_ = eval(b)
            e_ = eval(e)
            if not isinstance(b_, Iterable):
                b_ = [b_, b_, b_]
            if not isinstance(e_, Iterable):
                e_ = [e_, e_, e_]

            p = abs(p) - ti.Vector(b_)
            q = abs(p + ti.Vector(e_)) - ti.Vector(e_)
            return min(
                min(
                    (max(ti.Vector([p.x, q.y, q.z]), 0.0)).norm()
                    + min(max(p.x, max(q.y, q.z)), 0.0),
                    (max(ti.Vector([q.x, p.y, q.z]), 0.0)).norm()
                    + min(max(q.x, max(p.y, q.z)), 0.0),
                ),
                (max(ti.Vector([q.x, q.y, p.z]), 0.0)).norm()
                + min(max(q.x, max(q.y, p.z)), 0.0),
            )

        self.sdf = box


class Sphere(Shape):
    def __init__(self, r, color="#fff", **kwargs):

        super().__init__(color=color, **kwargs)

        if type(r) != str:
            r = str(r)

        @ti.func
        def sphere(p,t):
            return p.norm() - eval(r)

        self.sdf = sphere


class Torus(Shape):
    def __init__(self, t1, t2, color="#fff", **kwargs):

        super().__init__(color=color, **kwargs)

        @ti.func
        def torus(p, t):
            q = ti.Vector([(p.xy).norm() - t1, p.z])
            return q.norm() - t2

        self.sdf = torus


class Line(Shape):
    def __init__(self, a, b, r, color="#fff", **kwargs):

        super().__init__(color=color, **kwargs)

        if type(a) != ti.Vector:
            a = ti.Vector(np.array(a))
        if type(b) != ti.Vector:
            b = ti.Vector(np.array(b))

        @ti.func
        def line_(p, t):
            pa = p - a
            ba = b - a
            h = tg.scalar.clamp(pa.dot(ba) / ba.dot(ba), 0.0, 1.0)
            return (pa - ba * h).norm() - r

        self.sdf = line_


class Hexagon(Shape):
    def __init__(self, hx, hy=None, color="#fff", **kwargs):

        super().__init__(color=color, **kwargs)

        if hy is None:
            hy = hx

        h = ti.Vector([hx, hy])

        def sdf(p, t):
            k = ti.Vector([-0.8660254, 0.5, 0.57735])
            p = abs(p)
            p.xy -= 2.0 * min(k.xy.dot(p.xy), 0.0) * k.xy
            d = ti.Vector(
                [
                    (
                        p.xy
                        - ti.Vector([tg.scalar.clamp(p.x, -k.z * h.x, k.z * h.x), h.x])
                    ).norm()
                    * tg.sign(p.y - h.x),
                    p.z - h.y,
                ]
            )
            return min(max(d.x, d.y), 0.0) + (max(d, 0.0).norm())

        self.sdf = sdf
