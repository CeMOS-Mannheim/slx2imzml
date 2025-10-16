"""
slx2imzml - Convert SCiLS Lab files to imzML format

This module provides the main functionality for converting proprietary SCiLS Lab (.slx) files
to the open imzML standard for mass spectrometry imaging data. It supports both continuous
centroid and profile spectrum modes, and exports additional data as NRRD files.

Author: Jonas Cordes
Email: j.cordes@th-mannheim.de
Institution: TH Mannheim
"""

import sys
import argparse
import pathlib
import json
import numpy as np

# Data import from SCiLS Lab
import scilslab as sl

# Data export to NRRD (spot images)
import SimpleITK as sitk

# Local modules for imzML export
from . import slxFileHelper
from . import ImzMLWriter


def parse_arguments():
    """
    Parse command line arguments for the slx2imzml converter.
    
    Returns:
        argparse.Namespace: Parsed command line arguments containing the export instructions file path.
    """
    parser = argparse.ArgumentParser(description="Read a scils file and write imzML files (continuous profile/centroid mode).")
    parser.add_argument("export_instructions", type=str, help="Path to the export instructions file.")
    return parser.parse_args()


def main():
    """
    Main function for converting SCiLS Lab files to imzML format.
    
    This function processes export instructions from a JSON file and converts specified
    regions from SCiLS Lab files to imzML format along with additional NRRD exports
    for spot images, optical images, and region masks.
    
    The process includes:
    1. Loading export configuration from JSON
    2. Processing each specified .slx file
    3. Extracting regions and converting to imzML
    4. Exporting additional image data as NRRD files
    
    Raises:
        FileNotFoundError: If the export instructions file is not found.
        ValueError: If the SCiLS Lab file cannot be opened or processed.
    """
    print("Starting slx2imzml...")
    args = parse_arguments()
    helper = slxFileHelper(args.export_instructions)
    export_context = helper.load_export_info()
    
    # Process each .slx file specified in the export configuration
    for slx_context in export_context["data_exports"]:
        # Extract configuration parameters
        filename = slx_context["filename"]
        outputpath = slx_context.get("outputpath", ".")
        final_features = np.array(slx_context["final_features"])
        final_regions = slx_context["final_regions"]
        final_spot_images = slx_context["final_spot_images"]
        final_optical_images = slx_context["final_optical_images"]
        final_labels = slx_context["final_labels"]
        final_regions_as_labels = slx_context["final_regions_as_labels"]
        slice_thickness = slx_context.get("slice_thickness", 10)
        
        if outputpath is None or outputpath == "":
            outputpath = "."

        print(f"Processing file: {filename}")
        print(f"Features to export: {final_features.shape[0] if final_features.size > 0 else 'All (profile mode)'}")
        print(f"Regions to process: {len(final_regions)}")
        print(f"Spot images to export: {len(final_spot_images)}")
        print(f"Optical images to export: {len(final_optical_images)}")
        
        with sl.LocalSession(helper.scils_filepath) as slx_file:
            dataset = helper.get_dataset_proxy(slx_file)
            filename_without_extension = pathlib.Path(filename.replace(".slx", ""))

            # Process each region specified in the configuration
            for r_name, r_id in final_regions:
                print(f"## Processing region: {r_name}")

                # Create output directory structure
                (pathlib.Path(outputpath) / filename_without_extension / r_name).parent.mkdir(parents=True, exist_ok=True)

                # Extract spatial transformation information from the region
                transformation = dataset.get_index_images(r_id)[0].transformation
                spacing = slxFileHelper.get_pixel_spacing_3D_mm(transformation, slice_thickness)
                direction = slxFileHelper.get_direction_matrix(transformation, slice_thickness)
                origin = slxFileHelper.get_origin_mm(transformation)
                
                print(f"Pixel spacing: {spacing}")
                print(f"Origin: {origin}")
                print(f"Direction matrix: {direction}")

                def set_image_properties(image: sitk.Image) -> sitk.Image:
                    """
                    Set spatial properties for SimpleITK images.
                    
                    Args:
                        image: SimpleITK image to configure
                        
                    Returns:
                        sitk.Image: Configured image with spatial properties
                    """
                    image.SetOrigin(origin.tolist())
                    image.SetSpacing(spacing.tolist())
                    return image
                
                # Export region masks as multi-label NRRD image
                if final_regions_as_labels:
                    rlImage = slxFileHelper.load_regions_as_labels(
                        dataset, r_id, final_regions_as_labels, slice_thickness
                    )
                    set_image_properties(rlImage)
                    mask_path = f"{str(filename_without_extension / r_name)}.mask.nrrd"
                    sitk.WriteImage(rlImage, mask_path)
                    print(f"Exported region mask: {mask_path}")

                # Export spot images (normalizations, etc.)
                if final_spot_images:
                    print("Processing spot images...")
                    sImages = slxFileHelper.load_spot_images(
                        dataset, r_name, r_id, final_spot_images, slice_thickness
                    )
                    for name, image in sImages:
                        normalized_name = slxFileHelper.normalize(name)
                        set_image_properties(image)
                        spot_path = f"{str(filename_without_extension / r_name)}.{normalized_name}.nrrd"
                        sitk.WriteImage(image, spot_path)
                        print(f"Exported spot image: {spot_path}")
                
                # Export optical images
                if final_optical_images:
                    print("Processing optical images...")
                    oImages = slxFileHelper.load_optical_image(
                        dataset, r_name, r_id, final_optical_images, slice_thickness
                    )
                    for name, image in oImages:
                        normalized_name = slxFileHelper.normalize(name)
                        set_image_properties(image)
                        optical_path = f"{str(filename_without_extension / r_name)}.{normalized_name}.nrrd"
                        sitk.WriteImage(image, optical_path)
                        print(f"Exported optical image: {optical_path}")

                # Export mass spectrometry data as imzML
                print("Exporting imzML data...")
                imzml_path = f"{filename_without_extension}/{r_name}.imzML"
                writer = ImzMLWriter(imzml_path)
                
                if final_features.size == 0:
                    # Profile mode: export full spectral data
                    print("Using profile mode (full spectra)")
                    xs, data = slxFileHelper.load_region_data_as_continuous_profile(
                        dataset, r_name, r_id
                    )
                    uuid, sha1_hash, spectra_offsets = writer.ibd_write_continuous_spectra(data, xs)
                    writer.set_imzML_export_info(uuid, sha1_hash, "continuous", "profile spectrum")
                else:
                    # Centroid mode: export specified features only
                    print(f"Using centroid mode ({final_features.shape[0]} features)")
                    data = slxFileHelper.load_region_data_as_continuous_centroids(
                        dataset, r_name, r_id, final_features
                    )
                    uuid, sha1_hash, spectra_offsets = writer.ibd_write_continuous_spectra(data, final_features[:, 3])
                    writer.set_imzML_export_info(uuid, sha1_hash, "continuous", "centroid spectrum")

                # Set spatial information and write the imzML file
                writer.set_imzML_export_data(data, spacing, origin, direction, spectra_offsets)
                writer.write_imzML()
                print(f"Successfully exported: {imzml_path}")

    print("Conversion completed successfully!")


if __name__ == "__main__":
    main()

