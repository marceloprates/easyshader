import taichi as ti
from typing import Union, Tuple, List


@ti.data_oriented
class Light:
    """
    A class to represent a light source in a 3D space.

    Attributes:
    -----------
    pos : Union[Tuple[float, float, float], ti.Vector]
        The position of the light source. Default is (1, 1, 1).
    normal : Union[None, Tuple[float, float, float], ti.Vector]
        The normal vector of the light source. Default is None.
    radius : float
        The radius of the light source. Default is 5.0.
    translations : List[str]
        A list of translation transformations to be applied to the light source. Default is None.

    Methods:
    --------
    _convert_to_vector(value: Union[Tuple[float, float, float], ti.Vector]) -> ti.Vector
        Converts a tuple or ti.Vector to a ti.Vector.
    _determine_normal(normal: Union[None, Tuple[float, float, float], ti.Vector]) -> ti.Vector
        Determines the normal vector. If None, it returns the normalized negative position vector.
    __add__(other: str) -> "Light"
        Adds a translation transformation to the light source.
    apply_transformations(t: float) -> "Light"
        Applies the translation transformations to the light source and returns a new Light object.

    .. code-block::
        :caption: Example: Light with default parameters

            input: Light()
            output: Light object with pos=(1, 1, 1), normal=(-1, -1, -1), radius=5.0, translations=[]

    .. image:: ../source/_static/light_example.png
        :scale: 40 %
        :alt: Light example image
        :align: center
    """

    def __init__(
        self,
        pos: Union[Tuple[float, float, float], ti.Vector] = (0, 4, 4),
        normal: Union[None, Tuple[float, float, float], ti.Vector] = None,
        radius: float = 5.0,
        translations: List[str] = None,
    ) -> None:
        self.pos = self._convert_to_vector(pos)
        self.normal = self._determine_normal(normal)
        self.radius = radius
        self.translations = translations if translations is not None else []

    def _convert_to_vector(
        self, value: Union[Tuple[float, float, float], ti.Vector]
    ) -> ti.Vector:
        if isinstance(value, (ti.Vector, ti.Matrix)):
            value = list(value.to_numpy())
        return ti.Vector(value)

    def _determine_normal(
        self, normal: Union[None, Tuple[float, float, float], ti.Vector]
    ) -> ti.Vector:
        if normal is None:
            return -self.pos.normalized()
        return self._convert_to_vector(normal)

    def __add__(self, other: str) -> "Light":
        return Light(
            self.pos, self.normal, self.radius, translations=self.translations + [other]
        )

    def apply_transformations(self, t: float) -> "Light":
        pos = self.pos
        for translation in self.translations:
            pos += ti.Vector(eval(translation))
        return Light(
            pos,
            self.normal,
            self.radius,
        )
