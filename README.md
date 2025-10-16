# slx2imzml

A Python package for converting SCiLS Lab files (.slx) to open-standard imzML files for mass spectrometry imaging (MSI) data analysis.

## Overview

This package reads proprietary SCiLS Lab files (using API clients) and exports regions as accessible imzML files that are compatible with [M²aia](http://m2aia.de) and other open-source MSI analysis tools. The converter supports both continuous centroid and profile spectrum modes.

## Features

- **Multi-format Export**: Converts SCiLS Lab regions to imzML format with continuous profile/centroid spectrum support
- **Additional Outputs**: Exports spot images, optical images, and region masks as NRRD files
- **Flexible Configuration**: JSON-based export configuration for customizing output
- **M²aia Compatibility**: Generated imzML files are fully compatible with M²aia software
- **Batch Processing**: Process multiple regions and datasets in a single run

## Installation

### Prerequisites

- Python ≥ 3.6
- SCiLS Lab Python API (Bruker Daltonics) 

### Install from Source

```bash
git clone <repository-url>
cd slx2imzml
pip install -e .
```

### Dependencies

The package automatically installs the following dependencies:

- `numpy` - Numerical computations
- `pandas` - Data manipulation
- `jinja2` - XML template rendering
- `matplotlib` - Plotting utilities
- `SimpleITK` - Medical image processing
- `scilslab` - SCiLS Lab API integration

## Usage

### Command Line Interface

```bash
imzmlio export_instructions.json
```

### Export Instructions Format

Create a JSON configuration file to specify export parameters:

```json
{
  "description": "Export configuration for SCiLS to imzML conversion",
  "slice_thickness": 10,
  "data_exports": [
    {
      "filename": "path/to/your/file.slx",
      "final_features": [],
      "final_regions": [["Region1", "region_id_1"]],
      "final_spot_images": ["normalization1", "normalization2"],
      "final_optical_images": ["optical1"],
      "final_labels": [],
      "final_regions_as_labels": []
    }
  ]
}
```

### Output Files

For each processed region, the tool generates:

- **imzML Files**: `region_name.imzML` and `region_name.ibd` - imzML format files
- **NRRD Files**: 
  - `region_name.mask.nrrd` - Region mask as multilabel image
  - `region_name.spot_image_name.nrrd` - Spot images (e.g., normalizations)
  - `region_name.optical_image_name.nrrd` - Optical reference images

The naming is based on the regions_name defined in SCiLS Lab.

## Technical Details

### Spectrum Modes

- **Continuous Profile/Centroid**: Full spectrum data export (when `final_features` is empty)
- **Continuous Centroid**: Feature-based export with specified m/z values

### Coordinate System

The converter preserves spatial information including:
- Pixel spacing in micrometers
- Origin coordinates
- Direction matrix for proper spatial positioning

### Data Processing

1. **Region Processing**: Extracts spectral data for each specified region
2. **Feature Selection**: Applies feature lists or exports full spectra
3. **Spatial Mapping**: Maintains pixel-to-coordinate transformations
4. **Format Conversion**: Generates imzML/ibd file pairs with proper metadata

## File Structure

```
slx2imzml/
├── __init__.py              # Package initialization
├── ImzMLIO.py              # Main conversion logic
├── ImzMLWriter.py          # imzML file writing utilities
├── ScilsLabFileHelper.py   # SCiLS Lab data access helpers
└── imzMLTemplate.j2        # Jinja2 template for imzML XML
```

## Contributing

This project is developed at Hochschule Mannheim in connection with the M²aia project for open-source mass spectrometry imaging analysis.

## License

MIT License - See LICENSE file for details.

## Related Projects

- [M²aia](http://m2aia.de) - Open-source software for mass spectrometry imaging
- [imzML](https://ms-imaging.org/wp/imzml/) - Open standard for mass spectrometry imaging data

## Support

For questions and issues, please contact the author or visit the M²aia project website.