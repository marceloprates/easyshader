from turtle import Shape
import taichi as ti
import numpy as np
from typing import Iterable
from numpy import pi as π

# from .shape import Shape, Box
from taichi import *
from taichi_glsl.sampling import sample, bilerp
from PIL import Image
import os
import subprocess
import IPython
import glob
from matplotlib.colors import hex2color, rgb2hex
import cv2

def hex2numpy(c):
    return np.array(hex2color(c))


@ti.pyfunc
def get_palette(palette):
    palette_field = ti.Vector.field(3, dtype=ti.f32, shape=(len(palette), 1))
    palette_field.from_numpy(
        np.array([[hex2numpy(p)] for p in palette], dtype=np.float32)
    )
    return palette_field


@ti.data_oriented
class Rendering:
    
    def __init__(
        self,
        scene=[],
        iterations=200,
        resolution=(400, 400),
        frames=20,
        max_ray_depth=10,
        max_raymarch_steps=100,
        eps=1e-6,
        inf=1e10,
        fov=0.2,
        dist_limit=100.0,
        camera_pos=[0, 0, 10],
        light_pos=(0, 1, 1),
        #light_normal=(0, -1, 0),
        light_normal = None,
        light_radius=3.0,
        palette=["#000", "#fff"],
        use_cel_shading=False,
        animate=False,
        color_buffer=None,
        palette_field=None,
    ):
        """
        Args:
            resolution (tuple, optional): Rendering resolution in pixels. Defaults to (500,500).
            max_ray_depth (int, optional): Maximum ray depth. Defaults to 6.
            eps (_type_, optional): epsilon for SDF distance. Defaults to 1e-6.
            inf (_type_, optional): inf. Defaults to 1e10.
            fov (float, optional): Field-of-View. Defaults to 0.2.
            dist_limit (int, optional): Distance limit. Defaults to 100.
            camera_pos (list, optional): Camera position. Defaults to [0., 0., 5.].
            light_pos (list, optional): Light position. Defaults to [0., 0., 1.].
            light_normal (list, optional): Light direction. Defaults to [0., 0, -1.].
            light_radius (float, optional): Light radius. Defaults to 3.0.
        """

        # Set attributes
        self.__dict__.update(
            {
                k: (
                    (
                        ti.Vector(np.array(v, dtype=np.float32))
                        if type(v) != ti.Vector
                        else v
                    )
                    if isinstance(v, Iterable) and k not in ["scene", "resolution"]
                    else v
                )
                for k, v in locals().items()
                if k not in ["palette"]
            }
        )

        if self.light_normal is None:
            self.light_normal = -self.light_pos
        
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

        # Init depth map
        self.depth_map = ti.field(ti.f32, shape=resolution)

        # Init iterations count
        self.iteration = 0

        # Init time
        self.time = 0.0

        self.scene = self.scene[0]

    def save(self, filename, display=False):
        # Render
        for i in range(self.max_iterations):
            self.render(0)
        # Transform image
        img = Image.fromarray(
            (255 * self.result().transpose((1, 0, 2))[::-1, :, :]).astype(np.uint8)
        )
        # Save render
        img.save(f"{filename}.png")
        # Display render
        if display:
            IPython.display.display(
                IPython.display.Image(
                    data=open(f"{filename}.png", "rb").read(), format="png"
                )
            )

    def render(self, t=0):
        for i in range(self.iterations):
            self._render(t)
        self.iteration += 1

    @ti.kernel
    def _render(self, t: ti.f32):
        """
        Render scene
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
            while depth < self.max_ray_depth:
                it += 1
                # Cast light ray and get:
                # 1. closest (distance to surface)
                # 2. normal (surface normal)
                # 3. color (color at hit position)
                closest, normal, c = self.next_hit(pos, d, t)
                if it == 1:
                    first_closest = closest

                depth += 1
                # Intersect light
                if self.intersect_light(pos, d, t) < closest:
                    hit_light = 1.0
                    depth = self.max_ray_depth
                else:
                    hit_pos = pos + closest * d
                    if normal.norm_sqr() != 0:
                        d = self.out_dir(normal, hit_pos, t)
                        pos = hit_pos + 5e-2 * d
                        throughput *= c
                    else:
                        depth = self.max_ray_depth

            self.color_buffer[u, v] += throughput * hit_light
            self.depth_map[u, v] = first_closest < self.inf

    @ti.func
    def next_hit(self, pos, d, t):
        """_summary_

        Args:
            pos (_type_): _description_
            d (_type_): _description_

        Returns:
            _type_: _description_
        """

        # Init:
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
    def ray_march(self, p, d, t):
        i, dist = 0, 0.0
        while (
            (dist < self.inf)
            and (i < self.max_raymarch_steps)
            and (self.sdf(p + dist * d, t) > 1e-6)
        ):
            dist += self.sdf(p + dist * d, t)
            i += 1

        return dist

    @ti.func
    def sdf_normal(self, p, t):
        d = 1e-3
        n = ti.Vector([0.0, 0.0, 0.0])
        sdf_center = self.sdf(p, t)
        for i in ti.static(range(3)):
            inc = p
            inc[i] += d
            n[i] = (1 / d) * (self.sdf(inc, t) - sdf_center)
        return n.normalized()

    @ti.func
    def sdf(self, p, t):
        geometry = np.inf
        geometry = min(geometry, self.scene.sdf(p, t))
        return geometry

    @ti.func
    def intersect_light(self, pos, d, t):
        """_summary_

        Args:
            pos (_type_): _description_
            d (_type_): _description_

        Returns:
            _type_: _description_
        """
        # Compute dot product between ray direction and light normal
        dot = -d.dot(self.light_normal)
        # Compute distance between light source and camera
        dist = d.dot(self.light_pos - pos)
        dist_to_light = self.inf
        if dot > 0 and dist > 0:
            D = dist / dot
            dist_to_center = (self.light_pos - (pos + D * d)).norm_sqr()
            if dist_to_center < self.light_radius**2:
                dist_to_light = D
        return dist_to_light

    @ti.func
    def out_dir(self, n, p, t):
        u = ti.Vector([1.0, 0.0, 0.0])
        if abs(n[1]) < 1 - self.eps:
            u = n.cross(ti.Vector([0.0, 1.0, 0.0])).normalized()
        v = n.cross(u)
        phi = 2 * π * ti.random()
        ay = ti.sqrt(ti.random())
        ax = ti.sqrt(1 - ay**2)
        out = (
            n
            if self.use_cel_shading
            else ax * (ti.cos(phi) * u + ti.sin(phi) * v) + ay * n
        )
        return out

    def result(self):

        img = self.color_buffer.to_numpy() * (1 / (self.iteration + 1))
        img = img / img.mean() * 0.24
        img = np.clip(np.sqrt(img), 0, 1)

        alpha = self.depth_map.to_numpy()
        # alpha = cv2.blur(alpha, (3, 3))
        # alpha = cv2.erode(alpha, np.ones((3, 3), np.uint8), iterations=1)
        alpha = np.expand_dims(alpha, -1)

        img = np.concatenate([img, alpha], -1)

        return img
