"""
slx2imzml - Convert SCiLS Lab files to imzML format

This package provides tools for converting proprietary SCiLS Lab (.slx) files 
to the open imzML standard for mass spectrometry imaging data.

Key Features:
- Converts SCiLS Lab regions to imzML format (continuous centroid and profile modes)
- Exports additional data as NRRD files (spot images, optical images, region masks)
- Compatible with M²aia and other open-source MSI analysis tools
- JSON-based configuration for flexible batch processing

Main Classes:
- ImzMLWriter: Handles imzML/ibd file generation
- ScilsLabFileHelper: Interface for SCiLS Lab data access and processing

Main Function:
- main: Command-line entry point for conversion

Author: Jonas Cordes
Email: j.cordes@th-mannheim.de  
Institution: Hochschule Mannheim

Example:
    $ slx2imzml export_config.json

Version: 0.1.0
"""

from .ImzMLWriter import ImzMLWriter
from .slxFileHelper import slxFileHelper  
from .ImzMLIO import main

__version__ = "0.1.0"
__author__ = "Jonas Cordes"
__email__ = "j.cordes@th-mannheim.de"

__all__ = ["ImzMLWriter", "slxFileHelper", "main"]
