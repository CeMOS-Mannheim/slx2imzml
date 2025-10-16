"""
ImzMLWriter - Write mass spectrometry imaging data to imzML format

This module provides functionality to write mass spectrometry imaging data
in the standard imzML format, which consists of an XML metadata file (.imzML)
and a binary data file (.ibd).

The imzML format is an open standard for sharing mass spectrometry imaging data
and is compatible with various analysis tools including M²aia.

Author: Jonas Cordes
Email: j.cordes@th-mannheim.de
Institution: Hochschule Mannheim
"""

import numpy as np
import hashlib
import uuid
import jinja2
import pathlib


class ImzMLWriter:
    """
    A class for writing mass spectrometry imaging data to imzML format.
    
    The ImzMLWriter handles the creation of both the XML metadata file (.imzML)
    and the binary data file (.ibd) that together comprise the imzML format.
    
    Attributes:
        imzML_path (str): Path to the output .imzML file
        ibd_path (str): Path to the output .ibd file
        context (dict): Template context for imzML generation
    """
    def __init__(self, imzML_path: str) -> None:
        """
        Initialize the ImzMLWriter.
        
        Args:
            imzML_path: Path where the .imzML file should be written.
                       The corresponding .ibd file will be created automatically.
        """
        self.imzML_path = imzML_path
        self.ibd_path = imzML_path.replace(".imzML", ".ibd")
        self.context = {}
        self.imzML_spectra_info = None
        self.uuid = None
        self.sha1 = None

    @staticmethod
    def _encode_and_write(ibd_file, data, dtype=np.float32):
        """
        Encode numerical data and write to binary file.
        
        Args:
            ibd_file: Open file handle for the .ibd file
            data: Numerical data to encode and write
            dtype: numpy data type for encoding (default: float32)
            
        Returns:
            tuple: (byte_offset, array_length, encoded_bytes_count)
        """
        data = np.asarray(data, dtype=dtype)
        offset = ibd_file.tell()
        encoded_bytes = data.tobytes()
        ibd_file.write(encoded_bytes)
        return offset, data.shape[0], data.nbytes 

    @staticmethod
    def _ibd_write_uuid(ibd_file):
        """
        Generate and write a UUID to the .ibd file header.
        
        Args:
            ibd_file: Open file handle for the .ibd file
            
        Returns:
            str: String representation of the generated UUID
        """
        u = uuid.uuid4()
        uuid_string = str(u)
        ibd_file.write(u.bytes)
        return uuid_string
    
    @staticmethod
    def _ibd_generate_sha1(ibd_file):
        """
        Generate SHA-1 hash of the entire .ibd file for data integrity.
        
        Args:
            ibd_file: Open file handle for the .ibd file
            
        Returns:
            str: Hexadecimal SHA-1 hash of the file contents
        """
        generator = hashlib.sha1()
        buffer_size = 2048
        ibd_file.seek(0)
        while chunk := ibd_file.read(buffer_size):
            generator.update(chunk)
        sha1_hash = generator.hexdigest()
        return sha1_hash

    
    def ibd_write_continuous_spectra(self, data, mz_values):
        """
        Write spectral data to .ibd file in continuous format.
        
        In continuous format, all spectra share the same m/z axis, which is stored
        once and referenced by all spectra. This is efficient for processed data.
        
        Args:
            data: 4D numpy array with shape (x, y, z, mz_points) containing intensity values
            mz_values: 1D numpy array containing m/z values for the shared axis
            
        Returns:
            tuple: (uuid_string, sha1_hash, spectra_metadata_list)
            
        Raises:
            NotImplementedError: If mz_values is not 1D (non-continuous format)
        """
        with open(self.ibd_path, "wb+") as ibd_file:
            # Write UUID header
            _uuid = ImzMLWriter._ibd_write_uuid(ibd_file)
            list_of_spectra_info = []
            indices = list(np.ndindex(data.shape[:3]))  # Only 3D coordinates (x,y,z)
            
            if len(mz_values.shape) == 1:
                # Write shared m/z axis once for all spectra (continuous format)
                mz_offset, mz_len, mz_len_enc = ImzMLWriter._encode_and_write(ibd_file, mz_values, dtype=np.float32)
            else:
                raise NotImplementedError("Only continuous memory layout is supported")
            
            # Write intensity data for each pixel
            for i, (x, y, z) in enumerate(indices):
                # Skip empty spectra (NaN or zero intensity)
                if np.isnan(data[x, y, z, 0]) or np.sum(data[x, y, z, :]) == 0:
                    continue

                # Write intensity values for this pixel
                int_offset, int_len, int_len_enc = ImzMLWriter._encode_and_write(ibd_file, data[x, y, z], dtype=np.float32)
                
                # Store metadata for this spectrum
                list_of_spectra_info.append({
                    "index": i + 1,
                    "x": x + 1,
                    "y": y + 1,
                    "z": z + 1,
                    "mz_offset": mz_offset,
                    "mz_len": mz_len,
                    "mz_enc_len": mz_len_enc,
                    "int_offset": int_offset,
                    "int_len": int_len,
                    "int_enc_len": int_len_enc,
                })
                            
            # Generate file integrity hash
            _sha1 = ImzMLWriter._ibd_generate_sha1(ibd_file)
        
        return _uuid, _sha1, list_of_spectra_info
    

    def set_imzML_export_info(self, uuid, sha1_hash,
                            spectrum_mode="continuous",
                            spectrum_type="centroid spectrum",
                            mz_data_type="32-bit float", 
                            int_data_type="32-bit float"):
        """
        Set imzML export metadata and controlled vocabulary codes.
        
        Args:
            uuid: UUID string from the .ibd file
            sha1_hash: SHA-1 hash of the .ibd file for integrity checking
            spectrum_mode: "continuous" or "processed" format
            spectrum_type: "centroid spectrum" or "profile spectrum"
            mz_data_type: Data type for m/z values (default: "32-bit float")
            int_data_type: Data type for intensity values (default: "32-bit float")
            
        Raises:
            ValueError: If uuid or sha1_hash is None
            KeyError: If an unsupported data type or mode is specified
        """
        self.export_type = " ".join([spectrum_mode, spectrum_type])

        # Controlled vocabulary mapping for imzML standard
        text_to_code_map = {
            "16-bit float": "1000520",
            "32-bit integer": "1000519",
            "32-bit float": "1000521",
            "64-bit integer": "1000522",
            "64-bit float": "1000523",
            "continuous": "1000030",
            "processed": "1000031",
            "zlib compression": "1000574",
            "no compression": "1000576",
            "positive scan": "1000130",
            "negative scan": "1000129",
            "centroid spectrum": "1000127",
            "profile spectrum": "1000128"
        }

        # Set spectrum format information
        self.context["spectrum_mode"] = spectrum_mode
        self.context["spectrum_type"] = spectrum_type
        self.context["spectrum_mode_code"] = text_to_code_map[spectrum_mode]
        self.context["spectrum_type_code"] = text_to_code_map[spectrum_type]
        
        # Set file integrity information
        if uuid is None:
            raise ValueError("UUID not set")
        self.context["uuid"] = uuid

        if sha1_hash is None:
            raise ValueError("SHA1 hash not set")
        self.context["sha1sum"] = sha1_hash

        # Set data type information
        self.context["mz_data_type"] = mz_data_type
        self.context["mz_data_type_code"] = text_to_code_map[mz_data_type]

        self.context["int_data_type"] = int_data_type
        self.context["int_data_type_code"] = text_to_code_map[int_data_type]

        # Set compression (currently no compression supported)
        self.context["mz_compression"] = "no compression"
        self.context["mz_compression_code"] = text_to_code_map["no compression"]

        self.context["int_compression"] = "no compression"
        self.context["int_compression_code"] = text_to_code_map["no compression"]



    def set_imzML_export_data(self, spectral_data, spacing, origin, direction, list_of_spectra_info):
        """
        Set spatial and spectral metadata for the imzML export.
        
        Args:
            spectral_data: 4D numpy array with shape (x, y, z, mz_points)
            spacing: 3D pixel spacing in millimeters
            origin: 3D origin coordinates in millimeters  
            direction: 3x3 direction matrix (currently unused but kept for compatibility)
            list_of_spectra_info: List of spectrum metadata dictionaries
        """
        # Convert from millimeters to micrometers (imzML standard unit)
        spacing_mu = spacing * 1000
        origin_mu = origin * 1000
        
        # Set image dimensions
        self.context["size_x"] = int(spectral_data.shape[0])
        self.context["size_y"] = int(spectral_data.shape[1])
        self.context["size_z"] = int(spectral_data.shape[2])

        # Set pixel spacing in micrometers
        self.context["pixel_size_x"] = spacing_mu[0]
        self.context["pixel_size_y"] = spacing_mu[1]
        self.context["pixel_size_z"] = spacing_mu[2]

        # Set maximum dimensions in micrometers
        self.context["max_dimension_x"] = spectral_data.shape[0] * spacing_mu[0]
        self.context["max_dimension_y"] = spectral_data.shape[1] * spacing_mu[1]
        self.context["max_dimension_z"] = spectral_data.shape[2] * spacing_mu[2]

        # Set origin coordinates in micrometers
        self.context["origin_x"] = origin_mu[0]
        self.context["origin_y"] = origin_mu[1]
        self.context["origin_z"] = origin_mu[2]

        # Set run information
        self.context["run_id"] = "0"
        self.context["num_spectra"] = len(list_of_spectra_info)
        self.context["spectra"] = list_of_spectra_info
    
    
    def write_imzML(self):
        """
        Generate and write the .imzML XML file using the Jinja2 template.
        
        This method renders the imzML XML using all the context data that has been
        set through the various set_* methods and writes it to the specified file path.
        
        Raises:
            FileNotFoundError: If the imzML template file cannot be found
            PermissionError: If the output file cannot be written
        """
        # Set up Jinja2 template environment
        template_loader = jinja2.FileSystemLoader(searchpath=pathlib.Path(__file__).parent)
        template_env = jinja2.Environment(loader=template_loader, trim_blocks=True, lstrip_blocks=True)
        
        # Load and render the imzML template
        template_file = "imzMLTemplate.j2"
        template = template_env.get_template(template_file)
        output_text = template.render(self.context)
        
        # Write the rendered XML to file
        with open(self.imzML_path, "w", encoding="utf-8") as f:
            f.write(output_text)