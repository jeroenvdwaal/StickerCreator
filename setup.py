#!/usr/bin/env python3
"""Setup script for Sticker Creator - AI-powered sticker extraction for KDE."""

from setuptools import setup, find_packages

setup(
    name="sticker-creator",
    version="1.0.0",
    description="AI-powered sticker creator for KDE - extract stickers from images using SAM 2 segmentation",
    long_description=open("README.md").read() if __import__("os").path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    author="Jeroen van der Waal",
    author_email="jeroenvdwaal@gmail.com",
    url="https://github.com/jeroenvdwaal/StickerCreator",
    license="GPLv3+",
    packages=find_packages(),
    py_modules=["main"],
    include_package_data=True,
    package_data={
        "sticker_creator": [
            "resources/*.qss",
            "resources/fonts/*.ttf",
            "resources/icons/*.svg",
            "resources/welcome/*.png",
            "resources/welcome/*.jpg",
            "resources/help/*.qch",
            "resources/help/*.qhc",
        ],
    },
    python_requires=">=3.9",
    install_requires=[
        "PySide6>=6.5",
        "torch>=2.0",
        "sam2>=1.0",
        "opencv-python>=4.8",
        "numpy>=1.24",
        "Pillow>=10.0",
        "requests>=2.31",
    ],
    entry_points={
        "console_scripts": [
            "sticker-creator=main:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: X11 Applications :: Qt",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Topic :: Multimedia :: Graphics :: Editors",
    ],
)
