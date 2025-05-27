import argparse
import os
import shutil
import logging
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import List
from tqdm import tqdm
import torch

class AnatomicalSegmentator:
    def __init__(self, gpu_id: int, preprocessed_folder: str, output_folder: str, max_workers: int = 4):
        self.gpu_id = gpu_id
        self.preprocessed_folder = Path(preprocessed_folder)
        self.output_folder = Path(output_folder)
        self.max_workers = max_workers
        self.setup_logging()
        self.check_gpu()

    def setup_logging(self):
        """Set up logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('anatomical_segmentation.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def check_gpu(self):
        """Check GPU availability and memory"""
        if not torch.cuda.is_available():
            self.logger.error("CUDA is not available. Please check GPU configuration.")
            raise RuntimeError("No GPU available")

        gpu_memory = torch.cuda.get_device_properties(self.gpu_id).total_memory / (1024 ** 3)
        self.logger.info(f"Using GPU {self.gpu_id} with {gpu_memory:.2f} GB memory")

    def segment_case(self, img_path: Path) -> bool:
        """Run segmentation for a single case"""
        case_id = img_path.stem.replace('_processed.nii', '')
        output_temp_dir = self.output_folder / case_id / "temp"  # Temporary directory
        output_temp_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Set GPU environment
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = str(self.gpu_id)

            # Command 1: Lung Vessels
            cmd_vessels = [
                'TotalSegmentator',
                '-i', str(img_path),
                '-o', str(output_temp_dir),
                '-ta', 'lung_vessels'
            ]
            self.logger.info(f"Running lung_vessels command: {' '.join(cmd_vessels)}")
            subprocess.run(cmd_vessels, env=env, check=True)

            # Command 2: Total Segmentation
            cmd_total = [
                'TotalSegmentator',
                '-i', str(img_path),
                '-o', str(output_temp_dir),
                '-ta', 'total'
            ]
            self.logger.info(f"Running total command: {' '.join(cmd_total)}")
            subprocess.run(cmd_total, env=env, check=True)

            # Organize the results
            self.organize_segments(output_temp_dir, case_id)
            self.logger.info(f"Successfully segmented {case_id}")
            return True

        except subprocess.CalledProcessError as e:
            self.logger.error(f"Segmentation failed for {case_id}: {str(e)}")
            return False
        finally:
            # Cleanup temp directory
            if output_temp_dir.exists():
                shutil.rmtree(output_temp_dir)
                self.logger.info(f"Cleaned up temporary directory for {case_id}")

    def organize_segments(self, output_temp_dir: Path, case_id: str):
        """Organize segmentation results into appropriate directories"""
        final_output_dir = self.output_folder / case_id
        final_output_dir.mkdir(parents=True, exist_ok=True)

        # Move specific outputs into their final directories
        categories = {
            'lobes': ['lung_upper_lobe_left.nii.gz',
                      'lung_upper_lobe_right.nii.gz',
                      'lung_middle_lobe_right.nii.gz',
                      'lung_lower_lobe_left.nii.gz',
                      'lung_lower_lobe_right.nii.gz'],
            'vessels': ['lung_vessels.nii.gz'],
            'airways': ['lung_trachea_bronchia.nii.gz']
        }

        for category, files in categories.items():
            category_dir = final_output_dir / category
            category_dir.mkdir(exist_ok=True)

            for file in files:
                src = output_temp_dir / file
                if src.exists():
                    shutil.move(str(src), str(category_dir / file))
                else:
                    self.logger.warning(f"File {file} not found for case {case_id}")

    def run_segmentation(self):
        """Run segmentation on all preprocessed images"""
        # Find all preprocessed files
        preprocessed_files = list(self.preprocessed_folder.glob("*_processed.nii.gz"))
        total_cases = len(preprocessed_files)

        if total_cases == 0:
            self.logger.error("No preprocessed files found")
            return

        self.logger.info(f"Starting segmentation of {total_cases} cases")

        # Parallel processing
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(tqdm(
                executor.map(self.segment_case, preprocessed_files),
                total=total_cases,
                desc="Segmenting cases"
            ))

        # Generate summary
        successful = sum(results)
        failed = total_cases - successful

        self.logger.info("\n=== Segmentation Summary ===")
        self.logger.info(f"Total cases processed: {total_cases}")
        self.logger.info(f"Successful: {successful}")
        self.logger.info(f"Failed: {failed}")


def main():
    parser = argparse.ArgumentParser(description='High-Quality Anatomical Structure Segmentation')
    parser.add_argument('--gpu_id', default=0, type=int, help='GPU ID to use')
    parser.add_argument('--preprocessed_folder', type=str, required=True,
                        help='Path to preprocessed images folder')
    parser.add_argument('--output_folder', type=str, required=True,
                        help='Path to segmentation results folder')
    parser.add_argument('--max_workers', type=int, default=4,
                        help='Number of parallel workers')

    args = parser.parse_args()

    segmentor = AnatomicalSegmentator(
        args.gpu_id,
        args.preprocessed_folder,
        args.output_folder,
        args.max_workers
    )
    segmentor.run_segmentation()


if __name__ == '__main__':
    main()