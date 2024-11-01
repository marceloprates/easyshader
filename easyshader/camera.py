import taichi as ti
import numpy as np
from PIL import Image
import os
import re
import subprocess

import IPython.display

from .rendering import Rendering
import imageio

from tqdm import tqdm


@ti.data_oriented
class Camera:
    def __init__(
        self,
        iterations=100,
        resolution=(400, 400),
        frames=20,
        framerate=30,
        max_ray_depth=10,
        max_raymarch_steps=50,
        eps=1e-6,
        inf=1e10,
        fov=0.2,
        dist_limit=100.0,
        camera_pos=[0, 0, 10],
        palette=["#000", "#fff"],
        use_cel_shading=False,
        animate=False,
        color_buffer=None,
        palette_field=None,
        translations=[],
    ):

        # Set attributes
        self.rendering_kwargs = {
            k: v for k, v in locals().items() if k not in ["self", "translations"]
        }

        self.translations = translations

    def snap(self, shape, lights, depth=False, t=0):

        # Apply camera translations
        camera_pos = self.rendering_kwargs["camera_pos"]
        for translation in self.translations:
            camera_pos += ti.Vector(np.array(eval(translation), dtype=np.float32))

        # Apply light translations
        for i, light in enumerate(lights):
            lights[i] = light.apply_transformations(t=0)

        # Create Rendering object
        rendering = Rendering(
            scene=(
                shape.with_background(shape.background_color, 4)
                if shape.background_color is not None
                else shape
            ),
            lights=lights,
            camera_pos=camera_pos,
            **{
                k: v
                for k, v in self.rendering_kwargs.items()
                if k not in ["camera_pos"]
            },
        )
        # print(rendering.use_cel_shading)

        # Create & return static image
        rendering.render(t)
        result = rendering.result(depth=depth)
        result = (255 * result).transpose(1, 0, 2)[::-1, :, :].astype(np.uint8)
        result = Image.fromarray(result, mode="RGBA")

        return result

    def record(
        self,
        shape,
        lights,
        frames=None,
        framerate=None,
        resume=False,
        depth=False,
        path=".tmp/output.gif",
    ):

        # Create folder
        directory = "/".join(path.split("/")[:-1]) if "/" in path else "."
        if not os.path.exists(directory):
            os.makedirs(directory)
        if resume:
            last_frame = np.max(
                [
                    int(f.split(".png")[0])
                    for f in os.listdir(".tmp")
                    if re.match(r"""\d+.png""", f)
                ]
            )
        else:
            # Clean .tmp folder
            for f in os.listdir(".tmp"):
                if f.endswith(".png"):
                    os.remove(f".tmp/{f}")

        (
            self.rendering_kwargs.pop("background_color")
            if "background_color" in self.rendering_kwargs
            else shape.background_color
        )
        frames = frames if frames is not None else self.rendering_kwargs.pop("frames")
        framerate = (
            framerate
            if framerate is not None
            else self.rendering_kwargs.pop("framerate")
        )

        scene = (
            shape.with_background(shape.background_color, 4)
            if shape.background_color is not None
            else shape
        )
        rendering = Rendering(scene, lights=lights, **self.rendering_kwargs)

        # Create animation
        images = []
        for i, t in enumerate(
            tqdm(np.linspace(0, 2 * np.pi, frames)[:-1], desc="Animating..")
        ):

            if resume and (i <= last_frame):
                continue

            camera_pos = rendering.camera_pos

            for translation in self.translations:
                camera_pos += ti.Vector(eval(translation))

            rendering_ = Rendering(
                scene=scene,
                lights=lights,
                camera_pos=camera_pos,
                **{
                    k: v
                    for k, v in self.rendering_kwargs.items()
                    if k not in ["camera_pos"]
                },
            )

            self.step()

            rendering_.color_buffer.fill(0.0)
            rendering_.iteration = 0
            rendering_.render(t)
            result = (
                (255 * rendering_.result(depth=depth))
                .transpose(1, 0, 2)[::-1, :, :]
                .astype(np.uint8)
            )
            images.append(result)

        vid_format = path.split(".")[-1]

        # Save GIF
        imageio.mimsave(".tmp/output.gif", images, fps=framerate, loop=0)

        if vid_format == "mp4":
            # Save MP4
            imageio.mimsave(path, images, fps=framerate)

        return IPython.display.Image(
            data=open(f".tmp/output.gif", "rb").read(), format="png"
        )

    def step(self):
        pass

    def __add__(self, other):
        return Camera(**self.rendering_kwargs, translations=self.translations + [other])
