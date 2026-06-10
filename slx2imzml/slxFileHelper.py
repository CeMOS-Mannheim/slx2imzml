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
from PIL import Image
import io
from typing import Any


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
    def normalize(filename: str, replace_chars: str = "/\\ ±:? \n\r\t", replace_with: str = "_") -> str:
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
            filename = filename.rstrip(". _")  # Remove trailing dots and spaces

        return filename

    @staticmethod
    def _to_float_or_nan(value) -> float:
        """Convert a value to float, returning NaN on failure."""
        try:
            return float(value)
        except (TypeError, ValueError):
            return np.nan

    @staticmethod
    def _is_ccs_feature_row(feature_row: np.ndarray) -> bool:
        """Check if a feature row contains finite CCS bounds.

        Expected row layout:
            [id, name, mz_low, mz_high, ccs_low, ccs_high]
        """
        if len(feature_row) <= 4:
            return False
        
        if len(feature_row) == 6:    
            ccs_low = slxFileHelper._to_float_or_nan(feature_row[4])
            ccs_high = slxFileHelper._to_float_or_nan(feature_row[5])
            print(f"Checking CCS feature row: ccs_low={ccs_low}, ccs_high={ccs_high}")
            return np.isfinite(ccs_low) and np.isfinite(ccs_high)

    # @staticmethod
    # def _build_final_features(features_df: pd.DataFrame, mz_offset_step: float = 1e-4) -> np.ndarray:
    #     """Build final feature matrix and shift duplicate centroids for CCS features.

    #     Returns an array with columns:
    #         [id, mz_low, mz_high, centroid_shifted, ccs_low, ccs_high]

    #     Duplicate centroid m/z values are shifted by `mz_offset_step` to keep separate
    #     feature channels distinguishable in downstream tools at tolerance 0. For groups
    #     with identical centroid m/z values, rows are ordered by CCS center so that higher
    #     CCS receives the larger m/z shift.
    #     """
    #     if features_df is None or features_df.empty:
    #         return np.empty((0, 6), dtype=np.float64)

    #     work = features_df.copy()

    #     required_cols = ["id", "mz_low", "mz_high"]
    #     missing = [c for c in required_cols if c not in work.columns]
    #     if missing:
    #         raise ValueError(f"Missing required feature columns: {missing}")

    #     if "ccs_low" not in work.columns:
    #         work["ccs_low"] = np.nan
    #     if "ccs_high" not in work.columns:
    #         work["ccs_high"] = np.nan

    #     work["id"] = pd.to_numeric(work["id"], errors="coerce")
    #     work["mz_low"] = pd.to_numeric(work["mz_low"], errors="coerce")
    #     work["mz_high"] = pd.to_numeric(work["mz_high"], errors="coerce")
    #     work["ccs_low"] = pd.to_numeric(work["ccs_low"], errors="coerce")
    #     work["ccs_high"] = pd.to_numeric(work["ccs_high"], errors="coerce")

    #     work = work.dropna(subset=["id", "mz_low", "mz_high"]).copy()
    #     if work.empty:
    #         return np.empty((0, 6), dtype=np.float64)

    #     work["centroid"] = (work["mz_low"] + work["mz_high"]) / 2.0
    #     work["ccs_center"] = (work["ccs_low"] + work["ccs_high"]) / 2.0

    #     # Stable base order by centroid and id for deterministic exports.
    #     work = work.sort_values(["centroid", "id"], kind="mergesort")

    #     shifted_centroids = np.empty(len(work), dtype=np.float64)
    #     grouped = work.groupby("centroid", sort=False, dropna=False)

    #     for _, grp in grouped:
    #         if len(grp) == 1:
    #             shifted_centroids[grp.index.to_numpy()] = grp["centroid"].to_numpy()
    #             continue

    #         # Within duplicate-centroid groups, sort by CCS center ascending so the
    #         # highest CCS gets the largest positive m/z offset.
    #         g = grp.copy()
    #         g["_ccs_sort"] = g["ccs_center"].fillna(-np.inf)
    #         g = g.sort_values(["_ccs_sort", "id"], kind="mergesort")

    #         base = float(g["centroid"].iloc[0])
    #         ranks = np.arange(len(g), dtype=np.float64)
    #         shifted = base + ranks * float(mz_offset_step)
    #         shifted_centroids[g.index.to_numpy()] = shifted

    #     work["centroid_shifted"] = shifted_centroids[work.index.to_numpy()]

    #     # Final deterministic order by shifted centroid.
    #     work = work.sort_values(["centroid_shifted", "id"], kind="mergesort")

    #     return work[["id", "mz_low", "mz_high", "centroid_shifted", "ccs_low", "ccs_high"]].to_numpy(dtype=np.float64)
    

    @staticmethod
    def _get_geometry_from_px2world(px2world: np.ndarray, slice_thickness: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get spacing, origin and direction from a documented SCiLS px2world matrix.

        According to the SCiLS API docs, px2world is a 4x4 transform mapping
        optical image pixel coordinates to SCiLS world coordinates.
        """
        tmat = np.asarray(px2world, dtype=np.float64).reshape(4, 4)
        basis = tmat[:3, :3]

        # Spacing is derived from basis vector lengths and converted from um to mm.
        spacing_x = np.linalg.norm(basis[:, 0]) / 1000.0
        spacing_y = np.linalg.norm(basis[:, 1]) / 1000.0
        spacing_z_raw = np.linalg.norm(basis[:, 2]) / 1000.0
        # Use the passed slice_thickness (assumed to be in micrometers) and convert to mm
        spacing_z = slice_thickness / 1000.0

        def _safe_unit(v: np.ndarray, fallback: np.ndarray) -> np.ndarray:
            n = np.linalg.norm(v)
            return (v / n) if n > 0 else fallback

        d0 = _safe_unit(basis[:, 0], np.array([1.0, 0.0, 0.0]))
        d1 = _safe_unit(basis[:, 1], np.array([0.0, 1.0, 0.0]))
        d2_from_t = _safe_unit(basis[:, 2], np.zeros(3))
        if np.linalg.norm(d2_from_t) > 0:
            d2 = d2_from_t
        else:
            d2 = np.cross(d0, d1)
            d2 = _safe_unit(d2, np.array([0.0, 0.0, 1.0]))

        direction = np.column_stack((d0, d1, d2))
        spacing = np.array([spacing_x, spacing_y, spacing_z], dtype=np.float64)
        origin = (tmat[:3, 3] / 1000.0).astype(np.float64)
        return spacing, origin, direction

    @staticmethod
    def get_geometry_from_transformation(transformation: np.ndarray, slice_thickness: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get spacing/origin/direction from a region transformation matrix."""
        return slxFileHelper._get_geometry_from_px2world(transformation, slice_thickness)

    @staticmethod
    def load_region_data_as_continuous_centroids(
        dataset: sl.DatasetProxy, r_name: str, r_id: int, features: np.ndarray,
        W_global: int, H_global: int, mappings: list, valid_spot_ids: set[int] | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Load region spectral data as continuous centroid format.
        
        Extracts ion images for specified features from a SCiLS Lab region and
        organizes them into a 4D array suitable for imzML centroid export.
        
        Args:
            dataset: SCiLS Lab dataset proxy
            r_name: Name of the region to process
            r_id: ID of the region to process
            features: Array of features with columns [id, name, mz_low, mz_high, centroid, ccs_low, ccs_high]
            W_global: Global grid width
            H_global: Global grid height
            mappings: List of subregion mappings (offsets and shapes)
            
        Returns:
            tuple[np.ndarray, np.ndarray]: 4D array with shape (y, x, z, features) containing intensity data and ccs spot image data if included in features.
            
        Raises:
            ValueError: If unexpected number of ion images found
        """
        print(f"Loading region data as centroids: {r_name} (ID: {r_id})")
        
        # x,y,z + s
        data = np.full((H_global, W_global, 1, features.shape[0]), np.nan)
        features_ccs = 0
        for f_index, feature_row in enumerate(features):
            print(feature_row)
            if slxFileHelper._is_ccs_feature_row(feature_row):
                features_ccs += 1
        data_ccs = np.full((H_global, W_global, 1, features_ccs), np.nan)

        index_images = dataset.get_index_images(r_id)
        if len(index_images) != len(mappings):
            raise ValueError(
                f"Unexpected number of index images found: expected {len(mappings)}, got {len(index_images)}"
            )

        valid_spot_ids_array = None
        if valid_spot_ids is not None:
            valid_spot_ids_array = np.array(sorted(int(v) for v in valid_spot_ids), dtype=np.int64)
        
        f_id_ccs = 0
        for f_index, feature_row in enumerate(features):
            f_id = int(feature_row[0])
            f_mz_low = float(feature_row[2])
            f_mz_high = float(feature_row[3])

            # CCS feature path: use per-feature intensities to keep CCS channels separate
            # even if m/z interval is identical.
            if slxFileHelper._is_ccs_feature_row(feature_row):
                ion = dataset.feature_table.get_feature_intensities(f_id, region_id=r_id, mode="area")
                values_by_spot = {int(sid): float(val) for sid, val in zip(ion.spot_ids, ion.values)}

                for mapping in mappings:
                    idx = mapping["index"]
                    offset_x = mapping["offset_x"]
                    offset_y = mapping["offset_y"]
                    idx_image = index_images[idx]
                    H_sub, W_sub = mapping["shape"]

                    local = np.full((H_sub, W_sub), np.nan, dtype=np.float64)
                    mask = idx_image.values >= 0
                    if valid_spot_ids_array is not None:
                        mask = mask & np.isin(idx_image.values, valid_spot_ids_array)
                    if np.any(mask):
                        spot_ids = idx_image.values[mask].astype(np.int64)
                        local[mask] = [values_by_spot.get(int(sid), np.nan) for sid in spot_ids]

                    data_ccs[offset_y:offset_y+H_sub, offset_x:offset_x+W_sub, 0, f_id_ccs] = local
                f_id_ccs += 1

            # Plain m/z feature path.
            ionimage_list = dataset.get_ion_images(f_mz_low, f_mz_high, r_id)
            if len(ionimage_list) == len(mappings):
                for mapping in mappings:
                    idx = mapping["index"]
                    offset_x = mapping["offset_x"]
                    offset_y = mapping["offset_y"]
                    img = ionimage_list[idx]
                    H_sub, W_sub = mapping["shape"]

                    local = np.full((H_sub, W_sub), np.nan, dtype=np.float64)
                    mask = index_images[idx].values >= 0
                    if valid_spot_ids_array is not None:
                        mask = mask & np.isin(index_images[idx].values, valid_spot_ids_array)
                    local[mask] = img.values[mask]

                    data[offset_y:offset_y+H_sub, offset_x:offset_x+W_sub, 0, f_index] = local
            else:
                raise ValueError(
                    f"Unexpected number of ion images found: expected {len(mappings)}, got {len(ionimage_list)}"
                )
        return data, data_ccs
    
    @staticmethod
    def load_region_data_as_continuous_profile(
        dataset: sl.DatasetProxy, r_name: str, r_id: int,
        W_global: int, H_global: int, mappings: list, valid_spot_ids: set[int] | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Load region spectral data as continuous profile format.
        
        Extracts full spectral profiles for all pixels in a region, suitable
        for imzML profile mode export. This preserves the complete spectral
        information rather than just peak centroids.
        
        Args:
            dataset: SCiLS Lab dataset proxy
            r_name: Name of the region to process
            r_id: ID of the region to process
            W_global: Global grid width
            H_global: Global grid height
            mappings: List of subregion mappings (offsets and shapes)
            
        Returns:
            tuple: (mz_axis, intensity_data) where:
                - mz_axis: 1D array of m/z values
                - intensity_data: 4D array with shape (y, x, z, mz_points)
        """
        print(f"Loading region data as profile: {r_name} (ID: {r_id})")
        
        index_images = dataset.get_index_images(r_id)
        all_unique_spot_ids = []
        valid_spot_ids_set = None if valid_spot_ids is None else {int(v) for v in valid_spot_ids}
        for img in index_images:
            un = sorted(np.unique(img.values))
            if un and un[0] == -1:
                un.pop(0)
            if valid_spot_ids_set is not None:
                un = [sid for sid in un if int(sid) in valid_spot_ids_set]
            all_unique_spot_ids.extend(un)
            
        all_unique_spot_ids = sorted(list(set(all_unique_spot_ids)))
        
        if not all_unique_spot_ids:
            raise ValueError(f"No valid spot IDs found in region {r_name}")
            
        ds = dataset.get_spectrum(all_unique_spot_ids[0], rebinned=True)
        xs = ds['mz']
        
        # x,y,z + s
        data = np.full((H_global, W_global, 1, xs.shape[0]), np.nan)
        total_spots = len(all_unique_spot_ids)
        spots_processed = 0
        
        for mapping in mappings:
            idx = mapping["index"]
            offset_x = mapping["offset_x"]
            offset_y = mapping["offset_y"]
            img = index_images[idx]
            
            mask = img.values >= 0
            if valid_spot_ids_set is not None:
                mask = mask & np.isin(img.values, np.array(sorted(valid_spot_ids_set), dtype=np.int64))
            rows, cols = np.where(mask)
            spot_ids = img.values[mask]
            
            for y, x, sid in zip(rows, cols, spot_ids):
                ds = dataset.get_spectrum(sid, rebinned=True)
                ys = ds['intensities']
                if np.sum(ys) == 0:
                    continue
                
                gx = offset_x + x
                gy = offset_y + y
                data[gy, gx, 0, :] = ys
                
                spots_processed += 1
                progress = (spots_processed) / total_spots * 100
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
        _, _, direction = slxFileHelper.get_geometry_from_transformation(transformation, slice_thickness)
        return direction
    
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
        spacing, _, _ = slxFileHelper.get_geometry_from_transformation(transformation, slice_thickness)
        return spacing

    @staticmethod
    def load_regions_as_labels(
        dataset: sl.DatasetProxy, r_id: int, final_regions_as_labels: list, slice_thickness: float,
        W_global: int = None, H_global: int = None, mappings: list = None
    ) -> sitk.Image:
        """
        Load multiple regions as a multi-label image.
        
        Creates a label image where each region is assigned a unique integer value,
        useful for segmentation and region-of-interest analysis.
        
        Args:
            dataset: SCiLS Lab dataset proxy
            r_id: Base region ID for spatial reference
            final_regions_as_labels: List of [name, id] pairs for regions to include
            slice_thickness: Z-axis slice thickness
            W_global: Global grid width (optional)
            H_global: Global grid height (optional)
            mappings: Subregion mappings (optional)
            
        Returns:
            sitk.Image: SimpleITK image with labeled regions
        """
        if W_global is None or H_global is None or mappings is None:
            _, _, _, W_global, H_global, mappings = slxFileHelper.get_combined_geometry(dataset, r_id, slice_thickness)
            
        index_images = dataset.get_index_images(r_id)
        current_label_value = 1
        labeled_array = np.zeros((H_global, W_global), dtype=np.ushort)

        for rl_name, rl_id in final_regions_as_labels:
            labeled_region = dataset.get_region_spots(rl_id)
            spot_ids = set(labeled_region["spot_id"])
            
            for mapping in mappings:
                idx = mapping["index"]
                offset_x = mapping["offset_x"]
                offset_y = mapping["offset_y"]
                img = index_images[idx]
                
                mask = np.isin(img.values, list(spot_ids))
                H_sub, W_sub = mapping["shape"]
                labeled_array[offset_y:offset_y+H_sub, offset_x:offset_x+W_sub][mask] = current_label_value
                
            current_label_value += 1

        # Convert to SimpleITK image with proper orientation
        sitk_image = sitk.GetImageFromArray(labeled_array[..., np.newaxis].T)
        return sitk_image
    

    @staticmethod
    def load_spot_images(
        dataset: sl.DatasetProxy, r_name: str, r_id: int, spot_images: list, slice_thickness: float,
        W_global: int = None, H_global: int = None, mappings: list = None,
        valid_spot_ids: set[int] | None = None
    ) -> list:
        """
        Load spot images (normalizations, etc.) for a specific region.
        
        Spot images include normalizations, dimensionality reduction maps, and other
        derived images that correspond to the spatial layout of the mass spectrometry data.
        
        Args:
            dataset: SCiLS Lab dataset proxy
            r_name: Name of the region (for logging)
            r_id: ID of the region to process
            spot_images: List of [name, id] pairs for spot images to load
            slice_thickness: Z-axis slice thickness
            W_global: Global grid width (optional)
            H_global: Global grid height (optional)
            mappings: Subregion mappings (optional)
            
        Returns:
            list: List of [name, sitk.Image] pairs containing the loaded spot images
        """
        if W_global is None or H_global is None or mappings is None:
            _, _, _, W_global, H_global, mappings = slxFileHelper.get_combined_geometry(dataset, r_id, slice_thickness)
            
        index_images = dataset.get_index_images(r_id)
        valid_spot_ids_array = None
        if valid_spot_ids is not None:
            valid_spot_ids_array = np.array(sorted(int(v) for v in valid_spot_ids), dtype=np.int64)
        
        spot_image_list = []
        for s_name, s_id in spot_images:
            print(f"Loading spot image: {s_name} (ID: {s_id}) for region: {r_name} (ID: {r_id})")
            spot_image = dataset.get_spot_image(s_id)
            
            # Create spatial array matching the global dimensions
            spatial_array = np.zeros((H_global, W_global))
            
            for mapping in mappings:
                idx = mapping["index"]
                offset_x = mapping["offset_x"]
                offset_y = mapping["offset_y"]
                img = index_images[idx]
                
                mask_foreground = img.values >= 0
                if valid_spot_ids_array is not None:
                    mask_foreground = mask_foreground & np.isin(img.values, valid_spot_ids_array)
                mask_indices = img.values[mask_foreground].astype(np.int32)
                
                rows, cols = np.where(mask_foreground)
                global_rows = offset_y + rows
                global_cols = offset_x + cols
                spatial_array[global_rows, global_cols] = spot_image.values[mask_indices]
            
            if "Total Ion Count" in s_name:
                s_name = "TIC"
                # from factor to absolute values 
                spatial_array = spatial_array * dataset.get_mean_spectrum()['intensities'].sum()
                
            if "Root Mean Square" in s_name:
                s_name = "RMS"
            
            # Convert to SimpleITK image
            sitk_image = sitk.GetImageFromArray(spatial_array[..., np.newaxis].T)
            spot_image_list.append([s_name, sitk_image])
        
        return spot_image_list

    @staticmethod
    def _extract_region_spot_ids(dataset: sl.DatasetProxy, region_id: Any) -> set[int]:
        """Return spot IDs for a region, best-effort with empty fallback."""
        try:
            spots = dataset.get_region_spots(region_id)
            if isinstance(spots, dict) and "spot_id" in spots:
                return {int(sid) for sid in spots["spot_id"]}
        except Exception:
            pass
        return set()

    @staticmethod
    def _collect_leaf_regions(region: sl.RegionTree) -> list[sl.RegionTree]:
        """Collect all leaf regions under a region (including itself if leaf)."""
        if not getattr(region, "subregions", None):
            return [region]

        leaves = []
        for sub in region.subregions:
            leaves.extend(slxFileHelper._collect_leaf_regions(sub))
        return leaves

    @staticmethod
    def _build_region_export_tasks(dataset: sl.DatasetProxy, region_tree: sl.RegionTree, query_regions: list | None) -> list[dict[str, Any]]:
        """Resolve selected regions into export tasks.

        Task types:
        - `leaf`: selected region is a leaf — exported directly.
        - `folder`: selected region is a folder — one task, folder spotlist = valid=1,
                    all descendant leaves become segmentation labels (2, 3, …).
        """
        all_regions = region_tree.get_all_regions()

        if query_regions is None or len(query_regions) == 0:
            selected_regions = [r for r in all_regions if len(getattr(r, "subregions", [])) == 0]
        else:
            selected_regions = []
            seen_ids: set[str] = set()

            for query in query_regions:
                exact = [r for r in all_regions if r.name == query]
                if exact:
                    for r in exact:
                        rid = str(r.id)
                        if rid not in seen_ids:
                            selected_regions.append(r)
                            seen_ids.add(rid)
                    continue

                # Backward-compatible regex fallback.
                try:
                    regex = re.compile(query)
                except re.error:
                    continue

                for r in all_regions:
                    if regex.search(r.name):
                        rid = str(r.id)
                        if rid not in seen_ids:
                            selected_regions.append(r)
                            seen_ids.add(rid)

        tasks: list[dict[str, Any]] = []
        used_names: Counter = Counter()

        for selected in selected_regions:
            selected_id = selected.id
            selected_name = selected.name
            subregions = getattr(selected, "subregions", [])

            if not subregions:
                task_name = selected_name
                used_names[task_name] += 1
                if used_names[task_name] > 1:
                    task_name = f"{task_name}__{used_names[task_name]:02d}"

                tasks.append({
                    "name": task_name,
                    "id": selected_id,
                    "selection_type": "leaf",
                    "selected_region_name": selected_name,
                    "selected_region_id": selected_id,
                    "valid_region_ids": [selected_id],
                    "segmentation_regions": [],
                })
                continue

            descendant_leaves = slxFileHelper._collect_leaf_regions(selected)
            descendant_leaves = [r for r in descendant_leaves if str(r.id) != str(selected_id)]
            seg_regions = sorted([[r.name, r.id] for r in descendant_leaves], key=lambda x: x[0])

            task_name = selected_name
            used_names[task_name] += 1
            if used_names[task_name] > 1:
                task_name = f"{task_name}__{used_names[task_name]:02d}"

            tasks.append({
                "name": task_name,
                "id": selected_id,
                "selection_type": "folder",
                "selected_region_name": selected_name,
                "selected_region_id": selected_id,
                # valid=1 = folder's own spotlist; leaves become labels 2, 3, …
                "valid_region_ids": [selected_id],
                "segmentation_regions": seg_regions,
            })

        return tasks

    @staticmethod
    def _union_spot_ids_for_regions(dataset: sl.DatasetProxy, region_ids: list[Any] | None) -> set[int]:
        """Return union of spot IDs for all region IDs."""
        if region_ids is None:
            return set()
        union_ids: set[int] = set()
        for rid in region_ids:
            union_ids.update(slxFileHelper._extract_region_spot_ids(dataset, rid))
        return union_ids

    @staticmethod
    def load_region_mask_with_segmentations(
        dataset: sl.DatasetProxy,
        r_id: Any,
        valid_region_ids: list[Any],
        segmentation_regions: list[list],
        slice_thickness: float,
        W_global: int = None,
        H_global: int = None,
        mappings: list = None,
    ) -> sitk.Image:
        """Create mask for a selected export region with segmentation overlays.

        Label semantics:
        - 0: background
        - 1: valid spectra (union of `valid_region_ids`)
        - 2..N: child/segmentation regions in `segmentation_regions`
                (in provided order, overriding label 1)
        """
        if W_global is None or H_global is None or mappings is None:
            _, _, _, W_global, H_global, mappings = slxFileHelper.get_combined_geometry(dataset, r_id, slice_thickness)

        index_images = dataset.get_index_images(r_id)
        labeled_array = np.zeros((H_global, W_global), dtype=np.ushort)

        valid_spot_ids = slxFileHelper._union_spot_ids_for_regions(dataset, valid_region_ids)
        valid_array = np.array(sorted(valid_spot_ids), dtype=np.int64) if valid_spot_ids else np.array([], dtype=np.int64)

        if valid_array.size > 0:
            for mapping in mappings:
                idx = mapping["index"]
                offset_x = mapping["offset_x"]
                offset_y = mapping["offset_y"]
                img = index_images[idx]
                H_sub, W_sub = mapping["shape"]
                mask_valid = np.isin(img.values, valid_array)
                labeled_array[offset_y:offset_y+H_sub, offset_x:offset_x+W_sub][mask_valid] = 1

        label_value = 2
        for _, seg_id in segmentation_regions:
            seg_spots = slxFileHelper._extract_region_spot_ids(dataset, seg_id)
            if valid_spot_ids:
                seg_spots = seg_spots.intersection(valid_spot_ids)
            seg_array = np.array(sorted(seg_spots), dtype=np.int64) if seg_spots else np.array([], dtype=np.int64)

            if seg_array.size > 0:
                for mapping in mappings:
                    idx = mapping["index"]
                    offset_x = mapping["offset_x"]
                    offset_y = mapping["offset_y"]
                    img = index_images[idx]
                    H_sub, W_sub = mapping["shape"]
                    mask_seg = np.isin(img.values, seg_array)
                    labeled_array[offset_y:offset_y+H_sub, offset_x:offset_x+W_sub][mask_seg] = label_value
            label_value += 1

        sitk_image = sitk.GetImageFromArray(labeled_array[..., np.newaxis].T)
        return sitk_image
    
    @staticmethod
    def load_optical_image(dataset: sl.DatasetProxy, optical_images: list, slice_thickness: float) -> list:
        """
        Load optical images associated with a region.
        
        Optical images provide morphological context for mass spectrometry imaging data,
        typically including H&E stains, immunofluorescence, or other reference images.
        
        Args:
            dataset: SCiLS Lab dataset proxy
            optical_images: List of [name, id] pairs for optical images to load
            slice_thickness: Z-axis slice thickness (currently unused)
            
        Returns:
            list: List of [name, sitk.Image] pairs containing the loaded optical images
        """
        optical_image_list = []
        
        for s_name, s_id in optical_images:
            try:
                optical_image = dataset.optical_images.get_image(s_id)
                print(optical_image.name, optical_image.type)

                # Load PNG binary data and convert to numpy array
                if hasattr(optical_image, 'data') and optical_image.data is not None:
                    # Decode PNG from binary blob
                    png_bytes = io.BytesIO(optical_image.data)
                    pil_image = Image.open(png_bytes)
                    # Convert to luminance (single channel grayscale) and then to numpy
                    image_array = np.array(pil_image.convert("L"))

                    # Convert to SimpleITK image (z, y, x)
                    sitk_image = sitk.GetImageFromArray(image_array[np.newaxis, ...])

                    if hasattr(optical_image, 'px2world') and optical_image.px2world is not None:
                        try:
                            spacing, origin, direction = slxFileHelper._get_geometry_from_px2world(
                                optical_image.px2world,
                                slice_thickness,
                            )
                            sitk_image.SetSpacing(spacing.tolist())
                            sitk_image.SetOrigin(origin.tolist())
                            sitk_image.SetDirection(direction.reshape(-1).tolist())
                        except Exception as geo_error:
                            print(
                                f"Warning: Could not set spacing/origin from px2world for "
                                f"'{s_name}' (ID: {s_id}): {geo_error}"
                            )
                    
                    optical_image_list.append([s_name, sitk_image])
                else:
                    print(f"Warning: No data or values attribute found for optical image '{s_name}' (ID: {s_id})")
            except Exception as e:
                print(f"Warning: Could not load optical image '{s_name}' (ID: {s_id}): {e}")
        
        return optical_image_list

    @staticmethod
    def get_combined_geometry(dataset: sl.DatasetProxy, r_id: int, slice_thickness: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int, list]:
        """
        Calculates the global bounding box and combined geometry for a region (leaf or folder).
        
        Args:
            dataset: SCiLS Lab dataset proxy
            r_id: Region ID (could be leaf or folder)
            slice_thickness: Z-axis slice thickness in micrometers
            
        Returns:
            tuple: (spacing, origin_global, direction, W_global, H_global, subregion_mappings)
                where:
                - spacing: 3D pixel spacing [sx, sy, sz] in millimeters
                - origin_global: 3D origin [ox, oy, oz] in millimeters representing global top-left
                - direction: 3x3 direction matrix
                - W_global: global grid width
                - H_global: global grid height
                - subregion_mappings: list of dicts with offset and shape keys
        """
        index_images = dataset.get_index_images(r_id)
        if not index_images:
            raise ValueError(f"No index images found for region ID: {r_id}")
            
        # All subregions share spacing and direction, so extract from the first
        spacing, origin_0, direction = slxFileHelper.get_geometry_from_transformation(
            index_images[0].transformation, slice_thickness
        )
        sx, sy, _ = spacing
        
        # Calculate world bounds over all subregions
        global_min_x = float('inf')
        global_max_x = float('-inf')
        global_min_y = float('inf')
        global_max_y = float('-inf')
        
        for img in index_images:
            H, W = img.values.shape
            origin = img.transformation[:3, 3] / 1000.0  # in mm
            ox, oy = origin[0], origin[1]
            
            x0 = ox
            x1 = ox + (W - 1) * sx
            global_min_x = min(global_min_x, x0, x1)
            global_max_x = max(global_max_x, x0, x1)
            
            y0 = oy
            y1 = oy - (H - 1) * sy
            global_min_y = min(global_min_y, y0, y1)
            global_max_y = max(global_max_y, y0, y1)
            
        origin_global = np.array([global_min_x, global_max_y, origin_0[2]], dtype=np.float64)
        
        W_global = int(round((global_max_x - global_min_x) / sx)) + 1
        H_global = int(round((global_max_y - global_min_y) / sy)) + 1
        
        subregion_mappings = []
        for idx, img in enumerate(index_images):
            ox, oy = img.transformation[0, 3] / 1000.0, img.transformation[1, 3] / 1000.0
            offset_x = int(round((ox - global_min_x) / sx))
            offset_y = int(round((global_max_y - oy) / sy))
            
            subregion_mappings.append({
                "index": idx,
                "offset_x": offset_x,
                "offset_y": offset_y,
                "shape": img.values.shape,
                "transformation": img.transformation
            })
            
        return spacing, origin_global, direction, W_global, H_global, subregion_mappings

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
        # If no specific regions requested, include all leaf regions (folders are skipped
        # unless explicitly named in query_regions)
        if query_regions is None or len(query_regions) == 0:
            # Default behaviour: only include leaf nodes
            return len(region.subregions) == 0
        
        # Check for exact name matches
        if region.name in query_regions:
            return True
        
        # Check for regex pattern matches
        for pattern in query_regions:
            # If the region is a subregion of the pattern (meaning region.name starts with pattern + "/"),
            # we do not want the pattern to match it (unless the subregion itself is in query_regions,
            # which is handled by the exact match above).
            if region.name.startswith(pattern + "/"):
                continue
            try:
                if re.search(pattern, region.name):
                    return True
            except re.error:
                pass
        return False

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
                    # if row["has_external_image"] == False:
                    optical_images.append([row["name"], row["id"]],)
                    
                if slx_context["optical_images"] is not None and len(slx_context["optical_images"]) > 0:
                    optical_images = [S for S in optical_images if S[0] in slx_context["optical_images"]]
                slx_context["final_optical_images"] = [S for S in optical_images]                
                
                
                # Process regions into export tasks (leafs and folder-clusters)
                all_regions = region_tree.get_all_regions()
                match_regions_by_name = slxFileHelper._match_regions_by_name
                slx_context["final_regions"] = slxFileHelper._build_region_export_tasks(
                    dataset, region_tree, slx_context["regions"]
                )
               
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

                features = None
                for f_id, f_count, f_listname in self.feature_lists:
                    feature_data = dataset.feature_table.get_features(f_id,  mode="area")
                    if "ccs_low" in feature_data.columns:
                        feature_data = feature_data[['id', 'name', 'mz_low', 'mz_high', 'ccs_low', 'ccs_high', ]]
                    else:
                        feature_data = feature_data[['id', 'name', 'mz_low', 'mz_high']]
                    if features is None:
                        features = feature_data
                    else:
                        features = np.concatenate((features, feature_data))
                
                if features is not None:
                    features = np.array(features)
                    # Calculate centroids as mean of mz_low and mz_high
                    centroids = np.mean(features[..., 2:4], axis=1)
                    # Sort features by centroid m/z value
                    sorted_indices = np.argsort(centroids)
                    # Combine features with centroids for final export
                    slx_context["final_features"] = features[sorted_indices].tolist()
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
                
    