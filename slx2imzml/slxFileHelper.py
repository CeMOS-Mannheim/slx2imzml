"""
ScilsLabFileHelper - Interface for SCiLS Lab data access and processing

This module provides helper functions and utilities for extracting data from SCiLS Lab files,
processing regions, features, and images, and preparing data for export to imzML format.

The main class handles JSON-based export configuration and provides methods to load
spectral data, spot images, optical images, and region information from SCiLS Lab datasets.

Author: Jonas Cordes
Email: j.cordes@th-mannheim.de
Institution: TH Mannheim
"""

import scilslab as sl
import pandas as pd
import sys
import signal
import json
import numpy as np
import SimpleITK as sitk
import re
from collections import Counter


class slxFileHelper:
    """
    Helper class for accessing and processing SCiLS Lab dataset information.
    
    This class provides an interface to SCiLS Lab datasets through JSON-based
    configuration files, enabling extraction of spectral data, regions, and
    associated images for conversion to imzML format.
    
    Attributes:
        _json_filepath (str): Path to the JSON configuration file
        _context (dict): Loaded JSON configuration context
        slice_thickness (float): Z-axis slice thickness in micrometers
        scils_filepath (str): Path to the SCiLS Lab file being processed
    """

    def __init__(self, json_filepath: str) -> None:
        """
        Initialize the ScilsLabFileHelper with a JSON configuration file.
        
        Args:
            json_filepath: Path to the JSON file containing export configuration
            
        Raises:
            FileNotFoundError: If the JSON configuration file doesn't exist
            json.JSONDecodeError: If the JSON file is malformed
        """
        self._json_filepath = json_filepath
        with open(self._json_filepath, 'r', encoding='utf-8') as f:
            self._context = json.load(f)
        self.slice_thickness = self._context.get("slice_thickness", 10)
        self.scils_filepath = None



    def _check_slx_context_tags(self, slx_context: dict) -> None:
        """
        Validate that required configuration tags are present in the export context.
        
        Args:
            slx_context: Dictionary containing export configuration
            
        Raises:
            ValueError: If any required tag is missing
        """
        required_tags = ["filename", "spot_images", "optical_images", "regions", "featurelists"]
        for tag in required_tags:
            if tag not in slx_context:
                raise ValueError(f"Missing required tag: {tag}")

    def _print_slx_context_tags(self, slx_context: dict) -> None:
        """
        Print summary of export context configuration for debugging.
        
        Args:
            slx_context: Dictionary containing export configuration
        """
        print(f"Filename: {slx_context.get('filename', 'N/A')}")
        print(f"Spot Images: {slx_context.get('spot_images', 'N/A')}")
        print(f"Optical Images: {slx_context.get('optical_images', 'N/A')}")
        print(f"Regions: {slx_context.get('regions', 'N/A')}")
        print(f"Feature Lists: {slx_context.get('featurelists', 'N/A')}")
        print(f"Labels: {slx_context.get('labels', 'N/A')}")

    def get_dataset_proxy(self, slx_file) -> sl.DatasetProxy:
        """
        Get the dataset proxy from a SCiLS Lab file with signal handling.
        
        Sets up signal handlers to properly close the dataset if the process
        is interrupted (Ctrl+C or segmentation fault).
        
        Args:
            slx_file: Open SCiLS Lab file session
            
        Returns:
            sl.DatasetProxy: The dataset proxy for accessing SCiLS Lab data
        """
        dataset = slx_file.dataset_proxy
        
        def signal_handler(sig, frame):
            print('You pressed Ctrl+C! Shutting down the SCiLS dataset...')
            slx_file.close()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGSEGV, signal_handler)
        return dataset


    @staticmethod
    def normalize(filename: str, replace_chars: str = "/\\ ±", replace_with: str = "_") -> str:
        """
        Normalize filename by replacing problematic characters.
        
        Replaces characters that are problematic in filenames (like path separators,
        special characters) with safe alternatives.
        
        Args:
            filename: The filename to normalize
            replace_chars: String containing characters to replace
            replace_with: Character to use as replacement
            
        Returns:
            str: Normalized filename safe for filesystem use
        """
        for c in replace_chars:
            filename = filename.replace(c, replace_with)
        return filename
    

    @staticmethod
    def load_region_data_as_continuous_centroids(dataset: sl.DatasetProxy, r_name: str, r_id: int, features: np.ndarray) -> np.ndarray:
        """
        Load region spectral data as continuous centroid format.
        
        Extracts ion images for specified features from a SCiLS Lab region and
        organizes them into a 4D array suitable for imzML centroid export.
        
        Args:
            dataset: SCiLS Lab dataset proxy
            r_name: Name of the region to process
            r_id: ID of the region to process
            features: Array of features with columns [id, mz_low, mz_high, centroid]
            
        Returns:
            np.ndarray: 4D array with shape (x, y, z, features) containing intensity data
            
        Raises:
            ValueError: If multiple ion images found for the same feature
        """
        print(f"Loading region data as centroids: {r_name} (ID: {r_id})")
        
        I = dataset.get_index_images(r_id)[0]
        # x,y,z + s
        data = np.zeros((I.values.shape[0], I.values.shape[1], 1, features.shape[0]))
        
        for f_index, (_, f_mz_low, f_mz_high, _) in enumerate(features):
            ionimage_list = dataset.get_ion_images(f_mz_low, f_mz_high, r_id)
            if len(ionimage_list) == 1:
                data[:, :, 0, f_index] = ionimage_list[0].values
            else:
                # Optionally, use logging here instead of print for production code
                # import logging
                # logging.warning(f"Multiple ion images found for the same feature: {len(ionimage_list)}")
                raise ValueError("Multiple ion images found for the same feature! "
                                    "This is not supported currently.")
        return data
    
    @staticmethod
    def load_region_data_as_continuous_profile(dataset: sl.DatasetProxy, r_name: str, r_id: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Load region spectral data as continuous profile format.
        
        Extracts full spectral profiles for all pixels in a region, suitable
        for imzML profile mode export. This preserves the complete spectral
        information rather than just peak centroids.
        
        Args:
            dataset: SCiLS Lab dataset proxy
            r_name: Name of the region to process
            r_id: ID of the region to process
            
        Returns:
            tuple: (mz_axis, intensity_data) where:
                - mz_axis: 1D array of m/z values
                - intensity_data: 4D array with shape (x, y, z, mz_points)
        """
        print(f"Loading region data as profile: {r_name} (ID: {r_id})")
        
        I = dataset.get_index_images(r_id)[0]
        un = sorted(np.unique(I.values))
        un.pop(0)
        
        ds = dataset.get_spectrum(0, rebinned=True)
        
        xs = ds['mz']
        
        # print(I.values.shape, xs)
        
        # Create a boolean mask that is True where the array's element is in the values list
        mask = np.isin(I.values, un)

        # Use np.where to get the row and column indices where the mask is True
        rows, cols = np.where(mask)
        
        
        # x,y,z + s
        data = np.zeros((I.values.shape[0], I.values.shape[1], 1, xs.shape[0]))
        progress = 0
        for i, (sid, x, y) in enumerate(zip(un, rows, cols)):
            ds = dataset.get_spectrum(sid, rebinned=True)
            ys = ds['intensities']
            if np.sum(ys) == 0:
                continue
            
            data[x, y, 0, :] = ys
            progress = (i + 1) / len(un) * 100
            print(f'Progress: {progress:.1f}%', end='\r')
            
        print()  # New line after progress is complete
        return xs, data
    
    @staticmethod
    def get_direction_matrix(transformation: np.ndarray, slice_thickness: float) -> np.ndarray:
        """
        Extract and normalize the direction matrix from SCiLS Lab transformation.
        
        Args:
            transformation: 4x4 transformation matrix from SCiLS Lab
            slice_thickness: Z-axis slice thickness in micrometers
            
        Returns:
            np.ndarray: 3x3 normalized direction matrix
        """
        direction_normalized = np.array(transformation)[:3, :3]
        direction_normalized[2, 2] = slice_thickness
        direction_normalized = direction_normalized / np.linalg.norm(direction_normalized, axis=1)
        return direction_normalized
    
    @staticmethod
    def get_origin_mm(transformation: np.ndarray) -> np.ndarray:
        """
        Extract origin coordinates in millimeters from transformation matrix.
        
        Args:
            transformation: 4x4 transformation matrix from SCiLS Lab
            
        Returns:
            np.ndarray: 3D origin coordinates in millimeters
        """
        return np.array(transformation)[:3, 3] / 1000
        
    @staticmethod
    def get_pixel_spacing_3D_mm(transformation: np.ndarray, slice_thickness: float) -> np.ndarray:
        """
        Calculate 3D pixel spacing in millimeters from transformation matrix.
        
        Args:
            transformation: 4x4 transformation matrix from SCiLS Lab
            slice_thickness: Z-axis slice thickness in micrometers
            
        Returns:
            np.ndarray: 3D pixel spacing [x, y, z] in millimeters
        """
        pixel_spacing = np.zeros(3)
        for i in range(2):  # X and Y from transformation matrix
            pixel_spacing[i] = np.sqrt(np.sum(np.square(transformation[i, :2]))) / 1000
        pixel_spacing[2] = slice_thickness / 1000  # Z from slice thickness
        return pixel_spacing

    @staticmethod
    def load_regions_as_labels(dataset: sl.DatasetProxy, r_id: int, final_regions_as_labels: list, slice_thickness: float) -> sitk.Image:
        """
        Load multiple regions as a multi-label image.
        
        Creates a label image where each region is assigned a unique integer value,
        useful for segmentation and region-of-interest analysis.
        
        Args:
            dataset: SCiLS Lab dataset proxy
            r_id: Base region ID for spatial reference
            final_regions_as_labels: List of [name, id] pairs for regions to include
            slice_thickness: Z-axis slice thickness (currently unused)
            
        Returns:
            sitk.Image: SimpleITK image with labeled regions
        """
        index_image = dataset.get_index_images(r_id)[0]
        current_label_value = 1
        labeled_array = np.zeros_like(index_image.values, dtype=np.ushort)

        for rl_name, rl_id in final_regions_as_labels:
            labeled_region = dataset.get_region_spots(rl_id)
            spot_ids = list(labeled_region["spot_id"])
            labeled_array[np.isin(index_image.values, spot_ids)] = current_label_value
            current_label_value += 1

        # Convert to SimpleITK image with proper orientation
        sitk_image = sitk.GetImageFromArray(labeled_array[..., np.newaxis].T)
        return sitk_image
    

    @staticmethod
    def load_spot_images(dataset: sl.DatasetProxy, r_name: str, r_id: int, spot_images: list, slice_thickness: float) -> list:
        """
        Load spot images (normalizations, etc.) for a specific region.
        
        Spot images include normalizations, dimensionality reduction maps, and other
        derived images that correspond to the spatial layout of the mass spectrometry data.
        
        Args:
            dataset: SCiLS Lab dataset proxy
            r_name: Name of the region (for logging)
            r_id: ID of the region to process
            spot_images: List of [name, id] pairs for spot images to load
            slice_thickness: Z-axis slice thickness (currently unused)
            
        Returns:
            list: List of [name, sitk.Image] pairs containing the loaded spot images
        """
        index_image = dataset.get_index_images(r_id)[0]
        mask_foreground = index_image.values >= 0
        mask_indices = index_image.values[mask_foreground].astype(np.int32)
        
        spot_image_list = []
        for s_name, s_id in spot_images:
            spot_image = dataset.get_spot_image(s_id)
            # Create spatial array matching the region dimensions
            spatial_array = np.zeros(index_image.values.shape[:3])
            spatial_array[mask_foreground] = spot_image.values[mask_indices]
            
            # Convert to SimpleITK image
            sitk_image = sitk.GetImageFromArray(spatial_array[..., np.newaxis].T)
            spot_image_list.append([s_name, sitk_image])
        
        return spot_image_list
    
    @staticmethod
    def load_optical_image(dataset: sl.DatasetProxy, r_name: str, r_id: int, optical_images: list, slice_thickness: float) -> list:
        """
        Load optical images associated with a region.
        
        Optical images provide morphological context for mass spectrometry imaging data,
        typically including H&E stains, immunofluorescence, or other reference images.
        
        Args:
            dataset: SCiLS Lab dataset proxy
            r_name: Name of the region (for logging)
            r_id: ID of the region (currently unused)
            optical_images: List of [name, id] pairs for optical images to load
            slice_thickness: Z-axis slice thickness (currently unused)
            
        Returns:
            list: List of [name, sitk.Image] pairs containing the loaded optical images
        """
        optical_image_list = []
        
        index_image = dataset.get_index_images(r_id)[0]
        mask_foreground = index_image.values >= 0
        mask_indices = index_image.values[mask_foreground].astype(np.int32)
        
        for s_name, s_id in optical_images:
            try:
                optical_image = dataset.optical_images.get_image(s_id)
                # Create spatial array matching the region dimensions
                spatial_array = np.zeros(optical_image.values.shape[:3])
                spatial_array[mask_foreground] = optical_image.values[mask_indices]

                # Convert to SimpleITK image
                sitk_image = sitk.GetImageFromArray(spatial_array[..., np.newaxis].T)
                optical_image_list.append([s_name, sitk_image])
                # optical_image_list.append([s_name, optical_image])
            except Exception as e:
                print(f"Warning: Could not load optical image '{s_name}' (ID: {s_id}): {e}")
        
        return optical_image_list

    @staticmethod
    def _match_regions_by_name(region: sl.RegionTree, query_regions: list) -> bool:
        """
        Check if a region matches the query criteria.
        
        A region matches if:
        1. It has no subregions (is a leaf node)
        2. Its name appears in the query list OR matches a regex pattern
        3. If query_regions is None/empty, all leaf regions match
        
        Args:
            region: SCiLS Lab region tree node to check
            query_regions: List of region names or regex patterns to match
            
        Returns:
            bool: True if the region matches the criteria, False otherwise
        """
        # Skip regions that have subregions (only process leaf nodes)
        if len(region.subregions):
            return False
        
        # If no specific regions requested, include all leaf regions
        if query_regions is None or len(query_regions) == 0:
            return True
        
        # Check for regex pattern matches
        for pattern in query_regions:
            if re.search(pattern, region.name):
                return True
        
        # Check for exact name matches
        return region.name in query_regions

    def load_export_info(self) -> dict:
        """
        Load and process export configuration from JSON file.
        
        This method processes the JSON configuration file to:
        1. Validate required configuration tags
        2. Extract region, feature, and image information from SCiLS Lab files
        3. Prepare final export lists with resolved IDs and metadata
        4. Save a detailed log of the export configuration
        
        Returns:
            dict: Complete export context with resolved IDs and metadata
            
        Raises:
            ValueError: If required configuration tags are missing
            FileNotFoundError: If SCiLS Lab files cannot be found
        """
        for slx_context in self._context["data_exports"]:

            self._check_slx_context_tags(slx_context)
            self._print_slx_context_tags(slx_context)
            
            # Open SCiLS Lab file and get dataset proxy
            self.scils_filepath = slx_context["filename"]
            with sl.LocalSession(self.scils_filepath) as slx_file:
                dataset = self.get_dataset_proxy(slx_file)
                
                # Get basic dataset information
                feature_lists = dataset.feature_table.get_feature_lists()
                region_tree = dataset.get_region_tree()
                
                # Process spot images (normalizations, dimensionality reduction maps, etc.)
                all_spot_images = [dataset.get_spot_image(img_id) for img_id in dataset.get_spot_image_ids()]
                if slx_context["spot_images"] is not None and len(slx_context["spot_images"]) > 0:
                    spot_images = [img for img in all_spot_images if img.name in slx_context["spot_images"]]
                else:
                    spot_images = all_spot_images

                slx_context["final_spot_images"] = [[img.name, img.id] for img in spot_images]   
                
                
                df: pd.DataFrame = dataset.optical_images.get_ids()
                optical_images = []
                for i, row in df.iterrows():
                    if not row["has_external_image"]:
                        optical_images.append([row["name"], row["id"]],)
                    
                if slx_context["optical_images"] is not None and len(slx_context["optical_images"]) > 0:
                    optical_images = [S for S in optical_images if S[0] in slx_context["optical_images"]]
                slx_context["final_optical_images"] = [S for S in optical_images]                
                
                
                # Process regions matching the query criteria
                match_regions_by_name = ScilsLabFileHelper._match_regions_by_name
                all_regions = region_tree.get_all_regions()

                slx_context["final_regions"] = [
                    [region.name, region.id] for region in all_regions 
                    if match_regions_by_name(region, slx_context["regions"])
                ]
               
                slx_context["final_regions_as_labels"] = [
                    [region.name, region.id] for region in all_regions 
                    if match_regions_by_name(region, slx_context["regions_as_labels"])
                ]
                
                # Process feature lists
                if slx_context["featurelists"] is not None and len(slx_context["featurelists"]) > 0:
                    self.feature_lists = feature_lists[
                        feature_lists['name'].isin(slx_context["featurelists"])
                    ][['id', 'num_features', 'name']].values.tolist()
                else:
                    self.feature_lists = feature_lists[['id', 'num_features', 'name']].values.tolist()

                # Process features: combine all feature lists and sort by m/z
                features = None
                for f_id, f_count, f_listname in self.feature_lists:
                    feature_data = dataset.feature_table.get_features(f_id)[['id', 'mz_low', 'mz_high']]
                    if features is None:
                        features = feature_data
                    else:
                        features = np.concatenate((features, feature_data))
                
                if features is not None:
                    features = np.array(features)
                    # Calculate centroids as mean of mz_low and mz_high
                    centroids = np.mean(features[..., 1:3], axis=1)
                    # Sort features by centroid m/z value
                    sorted_indices = np.argsort(centroids)
                    # Combine features with centroids for final export
                    slx_context["final_features"] = np.concatenate([
                        features[sorted_indices], 
                        centroids[sorted_indices][..., np.newaxis]
                    ], axis=1).tolist()
                else:
                    slx_context["final_features"] = []

                # Process labels
                all_labels = dataset.get_labels()
                if slx_context["labels"] is not None and len(slx_context["labels"]) > 0:
                    labels = [[label.name, label.id] for label in all_labels if label.name in slx_context["labels"]]
                else:
                    labels = [[label.name, label.id] for label in all_labels]
                slx_context["final_labels"] = labels
        
        # Save detailed export configuration log
        with open(self._json_filepath + ".log", 'w', encoding='utf-8') as json_file:
            json.dump(self._context, json_file, indent=4)

        return self._context
                
    