# setup.py
from pathlib import Path

from setuptools import setup, find_packages

setup(
    name="slx2imzml",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "numpy",
        "pandas",
        "jinja2",
        "matplotlib",
        "SimpleITK",
        "Pillow",
        "shiny",
        "faicons",
        # "scilslab" 
    ],
    entry_points={
        "console_scripts": [
            "slx2imzml-gui=slx2imzml_gui.launcher:main",
        ],
    },
    package_data={
        "slx2imzml": ["imzMLTemplate.j2"],
    },
    author="Jonas Cordes",
    author_email="j.cordes@th-mannheim.de",
    description="Convert SCiLS Lab (.slx) files to open-standard imzML files (compatible with M²aia).",
    long_description=Path("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    url="https://github.com/CeMOS-Mannheim/slx2imzml",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v2 (GPLv2)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)