from pathlib import Path

from setuptools import find_namespace_packages, setup

BASE_DIR = Path(__file__).parent
with open(Path(BASE_DIR, "requirements.txt"), "r") as file:
    required_packages = [ln.strip() for ln in file.readlines()]

setup(
    name="PVS",
    version=0.1,
    description="Polyp Video Segmentation",
    author="Konrad Reuter",
    author_email="konrad.reuter@tuhh.de",
    python_requires=">=3.9",
    install_requires=[required_packages],
)
