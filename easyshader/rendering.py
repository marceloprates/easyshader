from typing import Iterable, List, Tuple, Union
from numbers import Number

import numpy as np
import taichi as ti
from matplotlib.colors import hex2color
from numpy import pi as π
from taichi import *
import taichi_glsl as tg
from .utils import hex2numpy, get_palette

from .transformations import rotate_x, rotate_y, rotate_z
from .light import Light

# Define "3D Vector" type
Vec3D = ti.types.vector(3, ti.f32)


@ti.data_oriented
class Rendering:

    def __init__(
        self,
        scene,
        iterations: int = 100,
        width: Union[None, int] = None,
        aspect_ratio: Union[float, str] = 1,
        resolution: Tuple[int, int] = (400, 400),
        frames: int = 20,
        framerate: int = 30,
        max_ray_depth: int = 10,
        max_raymarch_steps: int = 50,
        eps: float = 1e-6,
        inf: float = 1e10,
        fov: float = 0.2,
        dist_limit: float = 100.0,
        camera_pos: List[float] = [0, 0, 10],
        lights: List[Light] = [Light(pos=(0, 1, 1), radius=3.0)],
        palette: List[str] = ["#000", "#fff"],
        use_cel_shading: bool = False,
        animate: bool = False,
        color_buffer: Union[None, ti.Vector.field] = None,
        palette_field: Union[None, ti.Vector.field] = None,
    ) -> None:

        if width is not None:
            if aspect_ratio == "A4":
                aspect_ratio = 8.3 / 11.7
            elif aspect_ratio == "A4_r":
                aspect_ratio = 11.7 / 8.3
            resolution = (width, int(aspect_ratio * width))

        # Set attributes
        self.__dict__.update(
            {
                k: (
                    (
                        ti.Vector(np.array(v, dtype=np.float32), dt=ti.f32)
                        if type(v) != ti.Vector
                        else v
                    )
                    if isinstance(v, Iterable)
                    and k not in ["scene", "resolution", "lights"]
                    else v
                )
                for k, v in locals().items()
                if k not in ["palette"]
            }
        )

        # Init palette field
        self.palette_hex = palette
        self.palette_field = (
            get_palette(palette) if palette_field is None else palette_field
        )

        # Init color buffer
        if color_buffer is not None:
            self.color_buffer = color_buffer
        else:
            self.color_buffer = ti.Vector.field(3, dtype=ti.f32, shape=resolution)

        # Init depth, depth map
        self.depth = ti.field(ti.f32, shape=resolution)
        # self.depth.from_numpy(np.inf*np.ones(resolution, dtype = np.float32))
        self.depth_map = ti.field(ti.f32, shape=resolution)

        # Init iterations count
        self.iteration = 0

        # Init time
        self.time = 0.0

    def render(self, t: float):
        """
        Render scene with 'self.iterations' raymarching iterations.

        :param t: Time parameter (between 0 and 2pi)
        :type t: float
        """
        for i in range(self.iterations):
            self._render(float(t))
        self.iteration += 1

    @ti.kernel
    def _render(self, t: ti.f32):  # type: ignore
        """
        Render scene with one raymarching iteration.

        :param t: Time parameter (between 0 and 2pi)
        :type t: ti.f32
        """

        # Iterate over camera coordinates (u,v)
        for u, v in self.color_buffer:
            # Get camera position
            pos = self.camera_pos
            # Get width, height
            h, w = self.resolution
            # Compute aspect ratio
            aspect_ratio = h / w

            # Get ray direction
            d = ti.Vector(
                [
                    (
                        2 * self.fov * (u + ti.random()) / w
                        - self.fov * aspect_ratio
                        - 1e-5
                    ),
                    2 * self.fov * (v + ti.random()) / w - self.fov - 1e-5,
                    -1.0,
                ]
            ).normalized()

            throughput = ti.Vector([1.0, 1.0, 1.0])

            # Init depth, hit light
            depth, hit_light = 0.0, 0.0

            it = 0
            first_closest = 0.0
            closest_ = self.inf
            while depth < self.max_ray_depth:
                it += 1
                # Cast light ray and get:
                # 1. closest (distance to surface)
                # 2. normal (surface normal)
                # 3. color (color at hit position)
                closest, normal, c = self.next_hit(pos, d, t)
                if it == 1:
                    first_closest = closest

                closest_ = min(closest_, closest)

                depth += 1
                # Intersect light
                light_dist = self.intersect_light(pos, d, t)
                if light_dist < closest:
                    hit_light = 1.0
                    break
                else:
                    hit_pos = pos + closest * d
                    if normal.norm_sqr() != 0:
                        d = self.out_dir(
                            normal, hit_pos, d, self.camera_pos, self.lights[0].pos, t
                        )
                        pos = hit_pos + 5e-2 * d
                        throughput *= c
                    else:
                        depth = self.max_ray_depth

            self.color_buffer[u, v] += throughput * hit_light
            self.depth[u, v] = first_closest
            self.depth_map[u, v] = first_closest < self.inf

    @ti.func
    def next_hit(self, pos: Vec3D, d: Vec3D, t: ti.f32):  # type: ignore
        """
        Given a (x,y,z) position 'pos', a ray direction 'd' and time 't', use raymarching to compute:
        - 'closest': Closest point to the surface
        - 'normal': Normal direction to the surface
        - 'color': Color at hit position

        :param pos: Position in 3D (x,y,z) space.
        :type pos: Vec3D
        :param d: Ray direction vector.
        :type d: Vec3D
        :param t: Time parameter (between 0 and 2pi).
        :type t: ti.f32
        :return: 1. Closest point to the surface; 2. Normal direction to the surface; 3. Color at hit position
        :rtype: (ti.f32, Vec3D, Vec3D)
        """

        # Initialize:
        # 1. closest (distance to surface)
        # 2. normal (surface normal)
        # 3. color (color at hit position)
        closest, normal, color = (
            self.inf,
            ti.Vector.zero(ti.f32, 3),
            ti.Vector.zero(ti.f32, 3),
        )

        # Compute distance to surface using raymarching
        ray_march_dist = self.ray_march(pos, d, t)

        # Only compute normal and color if distance doesn't exceed limit
        if (ray_march_dist < self.dist_limit) and (ray_march_dist < closest):

            closest = ray_march_dist
            normal = self.sdf_normal(pos + d * closest, t)
            hit_pos = pos + d * closest

            color = self.scene.color(hit_pos, t)

        return closest, normal, color

    @ti.func
    def ray_march(self, p: Vec3D, d: Vec3D, t: ti.f32) -> ti.f32:  # type: ignore
        """
        Use raymarching to determine the distance between the (x,y,z) position 'p' and the object surface in the ray direction 'd'.

        :param p: Position in 3D (x,y,z) space.
        :type p: Vec3D
        :param d: Ray direction vector.
        :type d: Vec3D
        :param t: Time parameter (between 0 and 2pi).
        :type t: ti.f32
        :return: Raymarch distance.
        :rtype: ti.f32
        """

        i, dist = 0, 0.0
        sdf_ = self.sdf(p + dist * d, t)
        while (dist < self.inf) and (i < self.max_raymarch_steps) and (sdf_ > 1e-6):
            dist += sdf_
            sdf_ = self.sdf(p + dist * d, t)
            i += 1

        return dist

    @ti.func
    def sdf_normal(self, p: Vec3D, t: ti.f32) -> Vec3D:  # type: ignore
        d = 1e-6
        n = ti.Vector([0.0, 0.0, 0.0])
        sdf_center = self.sdf(p, t)
        for i in ti.static(range(3)):
            inc = 1.0 * p
            inc[i] += d
            n[i] = (1 / d) * (self.sdf(inc, t) - sdf_center)
        return n.normalized()

    @ti.func
    def sdf(self, p: Vec3D, t: ti.f32) -> ti.f32:  # type: ignore
        return min(np.inf, self.scene.sdf(p, t))

    @ti.func
    def intersect_light(self, pos: Vec3D, d: Vec3D, t: ti.f32) -> ti.f32:  # type: ignore
        """
        Compute the distance to the light source if the ray intersects it.

        Args:
            pos (Vec3D): The starting position of the ray.
            d (Vec3D): The direction of the ray.
            t (ti.f32): Time parameter (between 0 and 2pi).

        Returns:
            ti.f32: Distance to the light source if intersected, otherwise a large value.
        """
        light = self.lights[0]
        dot = -d.dot(light.normal)
        dist = d.dot(light.pos - pos)
        dist_to_light = self.inf

        if dot > 0 and dist > 0:
            D = dist / dot
            dist_to_center = (light.pos - (pos + D * d)).norm_sqr()
            if dist_to_center < light.radius**2:
                dist_to_light = D

        return dist_to_light

    @ti.func
    def out_dir(
        self,
        n: Vec3D,
        p: Vec3D,
        d: Vec3D,
        camera_pos: Vec3D,
        light_pos: Vec3D,
        t: ti.f32,
    ) -> Vec3D:  # type: ignore

        d = d.normalized()
        n = n.normalized()

        u = ti.Vector([1.0, 0.0, 0.0])
        if abs(n[1]) < 1 - self.eps:
            u = n.cross(ti.Vector([0.0, 1.0, 0.0])).normalized()
        v = n.cross(u)

        phi = (1 / 1) * 2 * π * ti.random()
        ay = ti.sqrt(ti.random())
        ax = ti.sqrt(1 - ay**2)

        N = n  # Normal vector
        L = (p - light_pos).normalized()  # Point light source
        V = (p - camera_pos).normalized()  # Viewing direction
        R = (2 * L.dot(N)) * N - L  # Ray direction
        R = R.normalized()

        out = n
        if self.use_cel_shading:
            out = n
        elif False:
            diffuse = ti.cos(phi) * u + ti.sin(phi) * v + n
            # diffuse = N.dot(L)
            # specular = d - 2 * d.dot(n) * n
            specular = R.cross(V) ** 1
            out = (diffuse + specular) / 2
            out = specular
        else:
            out = ax * ti.cos(phi) * u + ti.sin(phi) * v + ay * n

        return out

    def result(self, depth: bool = False) -> np.ndarray:
        # Compute "img" as the color buffer normalized by the number of iterations
        img = self.color_buffer.to_numpy() * (1 / (self.iterations + 1))
        # ...
        img = img / img.mean() * 0.24
        img = np.clip(np.sqrt(img), 0, 1)
        # img = np.clip(img,0,1)

        if depth:
            img = np.expand_dims(self.depth.to_numpy(), -1)
            img = img.repeat(4, axis=-1)
        else:
            # Compute and add alpha channel
            alpha = np.expand_dims(self.depth_map.to_numpy(), -1)
            img = np.concatenate([img, alpha], -1)

        return img
