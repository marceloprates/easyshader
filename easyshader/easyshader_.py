
import math
import time
#from tkinter import W
from glob import glob
from typing import Callable
from matplotlib import pyplot as plt
from matplotlib.colors import hex2color
import numpy as np
from shape import hex2numpy
import taichi as ti
import taichi_glsl as tg
from celluloid import Camera
from PIL import Image
from IPython import display
import os
import cv2
import types
import numpy as np

#from .primitives import *
from .transformations import *

@ti.data_oriented
class Rendering():

    @ti.pyfunc
    def __init__(
        self,
        res = (500,500),
        max_ray_depth = 6,
        eps = 1e-6,
        inf = 1e10,
        fov = 0.2,
        dist_limit = 100,
        camera_pos = [0., 0., 5.],
        light_pos = [0., 0., 1.],
        light_normal = [0., 0, -1.],
        light_radius = 3.0,
        palette = ['#E689B3', '#CCDBDC', '#E689B3'],
        textures = []
    ):
    

        self.res = res
        self.max_ray_depth = max_ray_depth
        self.eps = eps
        self.inf = inf
        self.fov = fov
        self.dist_limit = dist_limit
        self.camera_pos = camera_pos
        self.light_pos = light_pos
        self.light_normal = light_normal
        self.light_radius = light_radius
        #self._palette = [ti.Vector(np.array(hex2color(c))/255) for c in palette]
        self.iterations = 0

        self.geom = []

        # Init taichi instance
        ti.init(arch=ti.gpu)

        # Init color buffer and camera position as taichi vectors
        self.color_buffer = ti.Vector.field(3, dtype=ti.f32, shape=res)
        self.camera_pos = ti.Vector(np.array(camera_pos).astype(np.float32))

        '''
        # Init textures
        #self.textures = [self.set_texture(path) for path in textures]
        img = cv2.imread(textures[0]).astype(np.float32) / 255
        img = img[:,::-1,::-1]
        img = np.transpose(img,(1,0,2))
        width, height, _ = img.shape
        self.texture_ = ti.Vector.field(3, dtype=ti.f32, shape=(width,height))
        '''

        self.draw()

    @ti.func
    def palette(self, field: ti.template(), index: ti.f32):
        return tg.sampling.bilerp(field, ti.Vector([index,0.]))

    # Raymarching functions
    @ti.func
    def intersect_light(self, pos, d):
        light_loc = ti.Vector(self.light_pos)
        dot = -d.dot(ti.Vector(self.light_normal))
        dist = d.dot(light_loc - pos)
        dist_to_light = self.inf
        if dot > 0 and dist > 0:
            D = dist / dot
            dist_to_center = (light_loc - (pos + D * d)).norm_sqr()
            if dist_to_center < self.light_radius**2:
                dist_to_light = D
        return dist_to_light

    @ti.func
    def out_dir(self, n, p, t):
        u = ti.Vector([1.0, 0.0, 0.0])
        if abs(n[1]) < 1 - self.eps:
            u = n.cross(ti.Vector([0.0, 1.0, 0.0])).normalized()
        v = n.cross(u)
        phi = 2 * math.pi * ti.random()
        ay = ti.sqrt(ti.random())
        ax = ti.sqrt(1 - ay**2)
        return ax * (ti.cos(phi) * u + ti.sin(phi) * v) + ay * n

    @ti.func
    def ray_march(self, p, d, global_t):
        j = 0
        dist = 0.0
        while j < 100 and self.sdf(p + dist * d,global_t) > 1e-6 and dist < self.inf:
            dist += self.sdf(p + dist * d, global_t)
            j += 1
        return min(self.inf, dist)

    @ti.func
    def sdf_normal(self, p, global_t):
        d = 1e-3
        n = ti.Vector([0.0, 0.0, 0.0])
        sdf_center = self.sdf(p,global_t)
        for i in ti.static(range(3)):
            inc = p
            inc[i] += d
            n[i] = (1 / d) * (self.sdf(inc,global_t) - sdf_center)
        return n.normalized()

    @ti.func
    def next_hit(self, pos, d, t):

        closest, normal, c = self.inf, ti.Vector.zero(ti.f32,3), ti.Vector.zero(ti.f32, 3)
        ray_march_dist = self.ray_march(pos, d, t)
        if ray_march_dist < self.dist_limit and ray_march_dist < closest:
            closest = ray_march_dist
            normal = self.sdf_normal(pos + d * closest, t)
            hit_pos = pos + d * closest

            for geom in ti.static(list(self.geom)):
                if geom.render and (geom.sdf(hit_pos,t) < 1e-6):
                    c = geom.color(hit_pos, t)

        return closest, normal, c

    @ti.func
    def sdf(self, p, t):
        geometry = np.inf
        for geom in ti.static(list(self.geom)):
            if geom.render:
                geometry = min(geometry, geom.sdf(p,t))
        return geometry

    @ti.kernel
    def _render(self, global_t: float):
        for u, v in self.color_buffer:
            aspect_ratio = self.res[0] / self.res[1]
            pos = self.camera_pos
            d = ti.Vector([
                (2 * self.fov * (u + ti.random()) / self.res[1] - self.fov * aspect_ratio - 1e-5),
                2 * self.fov * (v + ti.random()) / self.res[1] - self.fov - 1e-5, -1.0
            ])
            d = d.normalized()

            throughput = ti.Vector([1.0, 1.0, 1.0])

            depth = 0
            hit_light = 0.00

            while depth < self.max_ray_depth:
                closest, normal, c = self.next_hit(pos, d, global_t)
                depth += 1
                dist_to_light = self.intersect_light(pos, d)
                if dist_to_light < closest:
                    hit_light = 1
                    depth = self.max_ray_depth
                else:
                    hit_pos = pos + closest * d
                    if normal.norm_sqr() != 0:
                        d = self.out_dir(normal,hit_pos,global_t)
                        pos = hit_pos + 5e-2 * d
                        throughput *= c
                    else:
                        depth = self.max_ray_depth
            self.color_buffer[u, v] += throughput * hit_light

    def set_texture(self, path):
        img = cv2.imread(path).astype(float)
        width, height, _ = img.shape
        field = ti.field(dtype=ti.f32, shape=(width, height))
        field.from_numpy(img[:,:,0].astype(float32))
        return field

    @ti.func
    def texture(self, p, t):
        w,h = self.texture_.shape
        x,y = +w/2-(w/.5)*(p.x),+h/2-(h/.5)*(p.y)
        return tg.sampling.bilerp(self.texture_, ti.Vector([x,y]))

    def render(self, global_t = 0, clear = False):
        if clear:
            self.color_buffer.fill(0.)
            self.iterations = 0
        self.iterations += 1
        self._render(global_t)

    def scene(self, *shapes):
        self.geom = shapes

    def animate(
        self, iterations = 20, nframes = 100, interval = 50, loop = False,
        filename = 'tmp', folder = 'tmp'
        ):
        gui = ti.GUI('Blob', self.res)

        # Create folder
        if not os.path.exists(folder):
            os.makedirs(folder)

        i = 0
        global_t = 0.
        frames = []
        while gui.running:
            
            for j in range(iterations):
                self.render(global_t, clear = j == 0)

            img = self.color_buffer.to_numpy() * (1 / (self.iterations + 1))
            img = img / img.mean() * 0.24
            img = np.sqrt(img)
            img = np.clip(img, 0, 1)
            gui.set_image(img)
            gui.show()
            
            frames.append(img)
            img = np.transpose(img,[1,0,2])
            img = img[::-1,:,:]
            img = 255*img
            img[img > 255] = 255
            im = Image.fromarray(img.astype(np.uint8))
            im.save(f"{folder}/{i}.png")


            global_t += 2*np.pi/nframes
            if abs(global_t - 2*np.pi) < 1e-6:
                if not loop:
                    break

            i+=1

        gui.close()

        fig,ax = plt.subplots(figsize=(6,6))
        cam = Camera(fig)
        for frame in frames:
            x = np.transpose(frame,[1,0,2])
            x = x[::-1,:,:]
            x = np.clip(x,0,255)
            ax.imshow(x)
            ax.axis('off')
            plt.gca().set_axis_off()
            plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,  hspace = 0, wspace = 0)
            plt.margins(0,0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            cam.snap()

        anim = cam.animate(interval=interval)
        anim.save(f'{filename}.gif')

        plt.close()

        return display.Image(data=open(f'{filename}.gif','rb').read(), format='png')

    def show(self, dpi = 300, iterations = 20):

        gui = ti.GUI('Blob', self.res)

        i = 0
        global_t = 0.
        while gui.running:
            
            self.render(global_t)

            img = self.color_buffer.to_numpy() * (1 / (self.iterations + 1))
            img = img / img.mean() * 0.24
            img = np.sqrt(img)
            gui.set_image(img)
            gui.show()

        gui.close()

        fig,ax = plt.subplots(figsize=(img.shape[0]/dpi,img.shape[1]/dpi),dpi=dpi)
        x = np.transpose(img,[1,0,2])
        x = x[::-1,:,:]
        ax.imshow(x)
        ax.axis('off')
        plt.gca().set_axis_off()
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,  hspace = 0, wspace = 0)
        plt.margins(0,0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
