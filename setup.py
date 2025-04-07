"""
Setup script for natural-pointer package.
"""

from setuptools import setup, find_packages
import os

# Read requirements from requirements.txt
with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()

# Read long description from README
if os.path.exists("README.md"):
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()
else:
    long_description = "AI-based natural mouse movement simulation"

setup(
    name="natural-pointer",
    version="0.1.0",
    author="Claude AI",
    author_email="your.email@example.com",
    description="AI-based natural mouse movement simulation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/natural-pointer",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "natural-pointer=natural_pointer.main:main",
        ],
    },
)