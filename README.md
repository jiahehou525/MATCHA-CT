# MATCHA-CT: Multi-Aspect Text-Conditioned Hierarchical Architecture for Anatomically-Precise 3D CT Synthesis

[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](link-to-your-paper)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-green.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org)

## 📢 Code Release Status

**📋 Current Status**: This repository currently contains the project documentation and basic framework structure.

**🔄 Full Code Release**: The complete implementation, including training scripts, model weights, and evaluation tools, will be released upon paper acceptance.

**📝 Paper Status**: Our paper is currently under review. We will update this repository with the full codebase once the review process is complete.

**⭐ Stay Updated**: Please star this repository to receive notifications when the full code is released!

## 🔬 Overview

MATCHA-CT is a novel diffusion model framework for generating high-resolution, anatomically accurate 3D CT volumes from textual descriptions. Our approach combines text and anatomical guidance to produce clinically relevant synthetic medical images with superior quality and consistency.

### Key Features

- **High-Resolution Generation**: Produces 512×512×512 CT volumes with fine anatomical details
- **Text-Conditional Synthesis**: Generates pathology-specific features based on radiology reports
- **Anatomical Consistency**: Incorporates atlas-guided priors for structural coherence
- **Efficient Architecture**: Novel 3D Medical Integrated Convolution-Transformer (3D-MedICT)
- **Bidirectional Fusion**: Advanced text-image alignment through MBTIF module

## 🏗️ Architecture


Our framework consists of four main components:

1. **3D Medical Integrated Convolution-Transformer (3D-MedICT)**: Combines CNNs and transformers for efficient multi-scale feature extraction
2. **Medical Bidirectional Text-Image Fusion (MBTIF)**: Enhances semantic alignment through cross-modal information exchange
3. **Atlas-Guided Anatomical Prior**: Leverages population-level statistics for structural coherence
4. **Text-Conditional Super-Resolution**: Efficiently generates high-resolution CT images

## 🚀 Installation

### Prerequisites
- Python 3.8+
- CUDA 11.0+
- 12GB+ GPU memory (for inference)

### Setup Environment

```bash
# Clone the repository
git clone https://github.com/yourusername/MATCHA-CT.git
cd MATCHA-CT

# Create conda environment
conda create -n matcha-ct python=3.8
conda activate matcha-ct

# Install PyTorch (adjust for your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt
```

### Dependencies

```bash
pip install -r requirements.txt
```

## 📁 Data Preparation

### Dataset Structure
```
data/
├── CT-RATE/
│   ├── images/
│   │   ├── patient_001.nii.gz
│   │   └── ...
│   ├── reports/
│   │   ├── patient_001.txt
│   │   └── ...
│   └── splits/
│       ├── train.txt
│       ├── val.txt
│       └── test.txt
```

### Preprocessing

```bash
# Preprocess CT volumes and reports
python scripts/preprocess_data.py --data_dir data/CT-RATE --output_dir data/processed

# Generate anatomical atlas
python scripts/generate_atlas.py --segmentation_dir data/segmentations --output atlas/anatomical_prior.pkl
```

## 🎯 Usage

### Training

```bash
# Train the full MATCHA-CT model
python train.py --config configs/matcha_ct.yaml --gpus 4

# Train individual components
python train_encoder.py --config configs/3d_medicit.yaml
python train_fusion.py --config configs/mbtif.yaml
python train_diffusion.py --config configs/super_resolution.yaml
```

### Inference

```bash
# Generate CT volume from text prompt
python generate.py \
    --model_path checkpoints/matcha_ct_best.pth \
    --prompt "Chest CT showing bilateral pleural effusion and cardiomegaly" \
    --output generated_ct.nii.gz \
    --resolution 512

# Batch generation
python batch_generate.py \
    --model_path checkpoints/matcha_ct_best.pth \
    --prompts_file prompts.txt \
    --output_dir results/
```

## 📈 Evaluation

### Reproduce Paper Results

```bash
# Evaluate on CT-RATE validation set
python evaluate.py \
    --model_path checkpoints/matcha_ct_best.pth \
    --data_dir data/CT-RATE \
    --split val \
    --metrics fid ssim dsc mmd

# Run ablation studies
python ablation_study.py --config configs/ablation.yaml

# Downstream task evaluation
python downstream_eval.py --task classification --model_path checkpoints/matcha_ct_best.pth
```

## 🔧 Configuration

Key configuration parameters in `configs/matcha_ct.yaml`:

```yaml
model:
  3d_medicit:
    stages: 4
    expansion_factor: 4
    temperature: 2.0
  
  mbtif:
    text_dim: 768
    visual_dim: 512
    fusion_weight: 0.5
  
  diffusion:
    timesteps: 1000
    inference_steps: 50
    scheduler: "ddim"

training:
  batch_size: 4
  learning_rate: 2e-4
  num_epochs: 100
  mixed_precision: true
```

## 📊 Model Checkpoints

| Model | Size | Performance | Download |
|-------|------|-------------|----------|
| MATCHA-CT Full | 865M | FID: 6.3 | [Download](https://drive.google.com/...) |
| 3D-MedICT Only | 285M | - | [Download](https://drive.google.com/...) |
| MBTIF Module | 120M | - | [Download](https://drive.google.com/...) |

## 🧪 Experiments

### Reproduce Paper Experiments

```bash
# Main comparison experiments
bash scripts/run_comparison.sh

# Ablation studies
bash scripts/run_ablation.sh

# Text-conditional generation evaluation
bash scripts/evaluate_text_conditioning.sh

# Downstream task evaluation
bash scripts/run_downstream_tasks.sh
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## 🙏 Acknowledgments

- CT-RATE dataset for providing the training data
- TotalSegmentator for anatomical segmentation tools
- CXR-BERT for medical text encoding


## 🔗 Related Work

- [GenerateCT](https://github.com/example/generatect)
- [MAISI](https://github.com/example/maisi)
- [Medical Diffusion](https://github.com/example/medical-diffusion)

---
