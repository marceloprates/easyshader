import taichi as ti
import taichi_glsl as tg
import numpy as np


@ti.func
def rotate_x(p, theta):
    p.x, p.y, p.z = (
        p.x,
        ti.cos(theta) * p.y - ti.sin(theta) * p.z,
        ti.sin(theta) * p.y + ti.cos(theta) * p.z,
    )
    return p


@ti.func
def rotate_y(p, theta):
    p.x, p.y, p.z = (
        ti.cos(theta) * p.x + ti.sin(theta) * p.z,
        p.y,
        -ti.sin(theta) * p.x + ti.cos(theta) * p.z,
    )
    return p


@ti.func
def rotate_z(p, theta):
    p.x, p.y, p.z = (
        ti.cos(theta) * p.x - ti.sin(theta) * p.y,
        ti.sin(theta) * p.x + ti.cos(theta) * p.y,
        p.z,
    )
    return p


def translate(shape, d):
    def shape_(p):
        return shape(p - ti.Vector(d))

    return shape_


@ti.func
def cart2spherical(p):
    r = p.norm()
    θ = ti.math.acos(p.z / r)
    ϕ = 1.0
    if p.x > 0:
        ϕ = ti.atan2(p.x, p.y)
    elif p.x < 0 and p.y >= 0:
        ϕ = ti.atan2(p.x, p.y) + np.pi
    elif p.x < 0 and p.y < 0:
        ϕ = ti.atan2(p.x, p.y) - np.pi
    elif p.x == 0 and p.y > 0:
        ϕ = +np.pi / 2
    elif p.x == 0 and p.y < 0:
        ϕ = -np.pi / 2

    return ti.Vector([r, θ, ϕ])


@ti.func
def θ(p):
    r = p.norm()
    return ti.math.acos(p.z / r)


@ti.func
def ϕ(p):
    r = p.norm()
    θ = ti.math.acos(p.z / r)
    ϕ = 1.0
    if p.x > 0:
        ϕ = ti.atan2(p.x, p.y)
    elif p.x < 0 and p.y >= 0:
        ϕ = ti.atan2(p.x, p.y) + np.pi
    elif p.x < 0 and p.y < 0:
        ϕ = ti.atan2(p.x, p.y) - np.pi
    elif p.x == 0 and p.y > 0:
        ϕ = +np.pi / 2
    elif p.x == 0 and p.y < 0:
        ϕ = -np.pi / 2

    return ϕ


def smooth_union(a, b, k):
    @ti.func
    def smooth_union_(p, t):
        h = 0.5 + 0.5 * (a(p, t) - b(p, t)) / k
        if h < 0:
            h = 0
        if h > 1:
            h = 1
        return (1 - h) * a(p, t) + (h) * b(p, t) - k * h * (1.0 - h)

    return smooth_union_


def difference(a, b):
    @ti.func
    def difference_(p, t):
        return max(a(p, t), -b(p, t))

    return difference_


@ti.func
def twist(p, k):
    c_ = ti.cos(k * p.y)
    s_ = ti.sin(k * p.y)
    q = ti.Vector([c_ * p.x - s_ * p.z, s_ * p.x + c_ * p.z, p.y])
    return q


@ti.func
def twist_x(p, k):
    c_ = ti.cos(k * p.x)
    s_ = ti.sin(k * p.x)
    q = ti.Vector([c_ * p.y - s_ * p.z, s_ * p.y + c_ * p.z, p.x])
    return q


def interpolate(a, b, k):
    @ti.func
    def interpolate_(p, t):
        h = 0.5 + 0.5 * (a(p, t) - b(p, t)) / k
        if h < 0:
            h = 0
        if h > 1:
            h = 1
        return (1 - h) * a(p, t) + (h) * b(p, t) - k * h * (1.0 - h)

    return interpolate_


def union(*shapes):
    return sum(shapes, start=Shape())


def su(k=0.5):
    return Infix(lambda x, y: x.smooth_union(y, k))


def sd(k=0.5):
    return Infix(lambda x, y: x.smooth_difference(y, k))


def si(k=0.5):
    return Infix(lambda x, y: x.smooth_intersection(y, k))
