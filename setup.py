# setup.py
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
        # "scilslab" 
    ],
    entry_points={
        "console_scripts": [
            "slx2imzml=slx2imzml:main",
        ],
    },
    package_data={
        "slx2imzml": ["imzMLTemplate.j2"],
    },
    author="Jonas Cordes",
    author_email="j.cordes@th-mannheim.de",
    description="A package to read a proprietary scils file and write regions as open accessible imzML files (compatible with M²aia).",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/CeMOS-Mannheim/slx2imzml",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)