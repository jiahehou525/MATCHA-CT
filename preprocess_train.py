import os
import numpy as np
import torch
import nibabel as nib
import pandas as pd
import torch.nn.functional as F
import scipy.ndimage
from pathlib import Path
from typing import Tuple, Optional, List
import logging
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
import math

# Set environment variables
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

class ParallelBatchPreprocessor:
    def __init__(self,
                 metadata_path: str,
                 output_dir: str,
                 batch_size: int = 8,
                 num_workers: int = 16,
                 target_shape: Tuple[int, int, int] = (512, 512, 256),
                 target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
                 hu_min: int = -1024,
                 hu_max: int = 600,
                 window_center: int = -600,
                 window_width: int = 1500):

        self.output_dir = Path(output_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.target_shape = target_shape
        self.target_spacing = target_spacing
        self.hu_min = hu_min
        self.hu_max = hu_max
        self.window_center = window_center
        self.window_width = window_width

        # Set up logging
        self.setup_logging()

        # Load metadata
        self.metadata_df = pd.read_csv(metadata_path)

        # Initialize CUDA if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Using device: {self.device}")

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('preprocessing.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def get_rescale_params(self, file_name: str) -> Tuple[float, float]:
        """Get rescale parameters from metadata"""
        try:
            row = self.metadata_df[self.metadata_df['VolumeName'] == file_name]
            slope = float(row['RescaleSlope'].iloc[0]) if not row.empty else 1.0
            intercept = float(row['RescaleIntercept'].iloc[0]) if not row.empty else 0.0
            return slope, intercept
        except Exception as e:
            self.logger.error(f"Error getting rescale params for {file_name}: {str(e)}")
            return 1.0, 0.0

    def convert_to_hu(self, image: np.ndarray, slope: float, intercept: float) -> np.ndarray:
        """Convert raw values to HU units"""
        hu_image = image * slope + intercept
        return np.clip(hu_image, self.hu_min, self.hu_max)

    def apply_window_level(self, hu_image: np.ndarray) -> np.ndarray:
        """Apply window-level transform"""
        window_min = self.window_center - self.window_width // 2
        window_max = self.window_center + self.window_width // 2
        return np.clip(hu_image, window_min, window_max)

    def resample_to_spacing(self, image: np.ndarray, current_spacing: Tuple[float, float, float]) -> np.ndarray:
        """Resample image to target spacing"""
        resize_factors = [c / t for c, t in zip(current_spacing, self.target_spacing)]
        resampled = scipy.ndimage.zoom(image, resize_factors, order=3, mode='nearest')
        return resampled

    def process_batch(self, file_batch: List[Path]) -> None:
        """Process a batch of images in parallel"""
        try:
            # Load images in parallel
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                image_data = list(executor.map(self.load_and_convert, file_batch))

            # Filter out None values from failed loads
            image_data = [img for img in image_data if img is not None]

            if not image_data:
                return

            # All images should now be the same size
            # Convert to batch tensor
            batch_tensor = torch.stack([torch.from_numpy(img).float() for img in image_data])
            batch_tensor = batch_tensor.to(self.device)

            # Save results in parallel
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                executor.map(self.save_processed_image,
                             batch_tensor.cpu().numpy(),
                             [f.stem for f in file_batch])

        except Exception as e:
            self.logger.error(f"Error processing batch: {str(e)}")

    def load_and_convert(self, file_path: Path) -> Optional[np.ndarray]:
        """Load and convert a single image"""
        try:
            # Load image
            nii_img = nib.load(str(file_path))
            data = nii_img.get_fdata()

            # Get rescale parameters
            slope, intercept = self.get_rescale_params(file_path.name)

            # Convert to HU
            hu_data = self.convert_to_hu(data, slope, intercept)

            # Apply window level
            windowed_data = self.apply_window_level(hu_data)

            # Convert to tensor for resizing
            tensor_data = torch.from_numpy(windowed_data).float()

            # Add batch and channel dimensions
            tensor_data = tensor_data.unsqueeze(0).unsqueeze(0)

            # Resize to target shape
            resized_data = F.interpolate(
                tensor_data,
                size=self.target_shape,
                mode='trilinear',
                align_corners=True
            )

            # Remove batch and channel dimensions
            resized_data = resized_data.squeeze(0).squeeze(0)

            return resized_data.numpy()

        except Exception as e:
            self.logger.error(f"Error loading {file_path}: {str(e)}")
            self.logger.error(f"Original shape: {data.shape}")
            return None

    @torch.no_grad()
    def process_tensor_batch(self, batch_tensor: torch.Tensor) -> torch.Tensor:
        """Process a batch of tensors"""
        try:
            # Add channel dimension
            batch_tensor = batch_tensor.unsqueeze(1)

            # Resize to target shape
            resized_batch = F.interpolate(
                batch_tensor,
                size=self.target_shape,
                mode='trilinear',
                align_corners=True
            )

            return resized_batch.squeeze(1)
        except Exception as e:
            self.logger.error(f"Error processing tensor batch: {str(e)}")
            raise

    def save_processed_image(self, data: np.ndarray, original_name: str) -> None:
        """Save processed image"""
        try:
            output_path = self.output_dir / f"{original_name}_processed.nii.gz"

            # Create header
            new_header = nib.Nifti1Header()
            new_header.set_data_shape(self.target_shape)
            new_header.set_data_dtype(np.float32)
            new_header.set_zooms(self.target_spacing)

            # Create affine
            new_affine = np.diag(list(self.target_spacing) + [1.0])

            # Save image
            nib.save(
                nib.Nifti1Image(data.astype(np.float32), new_affine, header=new_header),
                str(output_path)
            )

        except Exception as e:
            self.logger.error(f"Error saving {original_name}: {str(e)}")

    def process_dataset(self, data_folder: str):
        """Process entire dataset with batching and parallel processing"""
        try:
            # Find all NIFTI files
            nifti_files = list(Path(data_folder).rglob("*.nii.gz"))
            total_files = len(nifti_files)

            self.logger.info(f"Found {total_files} files to process")

            # Log sample of input shapes
            sample_shapes = []
            for file in nifti_files[:5]:
                img = nib.load(str(file))
                sample_shapes.append(img.shape)
            self.logger.info(f"Sample input shapes: {sample_shapes}")
            self.logger.info(f"Target shape: {self.target_shape}")

            # Create output directory
            self.output_dir.mkdir(parents=True, exist_ok=True)

            # Process in batches
            for i in tqdm(range(0, total_files, self.batch_size)):
                batch_files = nifti_files[i:i + self.batch_size]
                self.process_batch(batch_files)

            self.logger.info("Dataset processing completed successfully")

        except Exception as e:
            self.logger.error(f"Error processing dataset: {str(e)}")


def main():
    # Configuration
    data_folder = "D:/CT-RATE/train"
    metadata_path = "D:/CT-RATE/dataset/metadata/train_metadata.csv"
    output_folder = "D:/CT-RATE/dataset/preprocessed_data"

    # Initialize preprocessor with optimized settings for RTX 4090 and 32-core CPU
    preprocessor = ParallelBatchPreprocessor(
        metadata_path=metadata_path,
        output_dir=output_folder,
        batch_size=8,         # Optimized for RTX 4090 24GB VRAM
        num_workers=16,       # Optimized for 32 logical cores
        target_shape=(512, 512, 256),
        target_spacing=(1.0, 1.0, 1.0),
        hu_min=-1024,
        hu_max=600,
        window_center=-600,
        window_width=1500
    )

    # Process dataset
    preprocessor.process_dataset(data_folder)


if __name__ == '__main__':
    main()
