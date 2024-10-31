from setuptools import setup, find_packages
from pathlib import Path
import os

parent_dir = Path(__file__).resolve().parent

# Install vsketch
os.system("pip install git+https://github.com/abey79/vsketch/")

setup(
    name="easyshader",
    version="v1.0",
    description="",
    # long_description=parent_dir.joinpath("README.md").read_text(),
    long_description_content_type="text/markdown",
    # url="https://github.com/marceloprates/prettymaps",
    author="Marcelo Prates",
    author_email="marceloorp@gmail.com",
    # license="MIT License",
    packages=find_packages(exclude=("assets", "notebooks", "prints", "script")),
    install_requires=parent_dir.joinpath("requirements.txt").read_text().splitlines(),
    classifiers=[
        "Intended Audience :: Science/Research",
    ],
)
