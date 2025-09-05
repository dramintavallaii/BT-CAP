# BT-CAP: Brain Tumor Compositional Augmentation Pipeline

![BT-CAP Logo](https://via.placeholder.com/150) <!-- Replace with your project logo if available -->

Welcome to **BT-CAP**, a powerful Python-based tool designed to enhance brain tumor imaging datasets through advanced compositional augmentation techniques. This pipeline leverages state-of-the-art image processing and machine learning to generate diverse, high-quality augmented data, supporting research and development in medical imaging and oncology.

## ✨ Features

- **Compositional Augmentation**: Seamlessly integrates tumor core rearrangement, rescaling, deformation, and preitumoral edema refinement.
- **Flexible Input/Output**: Supports customizable input directories and output paths for augmented data.
- **Normalization Options**: Apply Z-score normalization to ensure consistent data scaling.
- **Command-Line Interface**: Run the pipeline effortlessly via CLI or Python module.
- **Cross-Platform**: Designed for Windows, Linux, and macOS with Python 3.8+.

## 🚀 Getting Started

### Prerequisites

Ensure you have Python 3.8 or higher installed, along with the following dependencies:

- `numpy`
- `SimpleITK`
- `scipy`
- `scikit-learn`
- `scikit-image`

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/dramintavallaii/BT-CAP.git
   cd BT-CAP
   ```
2. Install in editable mode:
   ```bash
   pip install -e .
   ```

### Usage

#### Command-Line Interface

Run the pipeline with your data directories:

```bash
BT-CAP --input "C:\path\to\input" --output "C:\path\to\output" --samples 2 --seed 42 --workers 16 --normalize 
```

- `--input`: Path to the directory containing input data (e.g., T1N, T1C, T2F, T2W, SEG images).
- `--output`: Path to save augmented output data.
- `--samples`: Number of augmented samples per case (default: 1).
- `--seed`: Random seed setting for reproducibility (optional).
- `--workers`: Number of parallel workers (default: CPU count) (optional).
- `--normalize`: Apply Z-score normalization (optional).


Check available options:

```bash
BT-CAP --help
```

#### Python Module

Execute as a module for programmatic use:

```bash
python -m BT_CAP --help
```

#### Example Input Structure

```
input/
├── Case1/
│   ├── t1n.nii.gz
│   ├── t1c.nii.gz
│   ├── t2f.nii.gz
│   ├── t2w.nii.gz
│   ├── seg.nii.gz
│   └── (optional) mask.nii.gz
├── Case2/
│   ...
```

**Note**: Applying brain extraction prior to using this pipeline and including a brain mask (e.g., `mask.nii.gz`) in each case's folder is highly recommended. This improves the quality and precision of the augmentation by focusing on relevant brain regions. The pipeline will work without a brain mask and it is optional, but its inclusion is strongly encouraged for optimal results.

#### Example Output

```
output/
├── Aug0-Case1/
│   ├── Aug0-t1n.nii.gz
│   ├── Aug0-t1c.nii.gz
│   ├── Aug0-t2f.nii.gz
│   ├── Aug0-t2w.nii.gz
│   └── Aug0-seg.nii.gz
├── Aug1-Case1/
│   ...
```

## 📖 Project Structure

- `BT-CAP/`
  - `setup.py`: Package configuration and installation script.
  - `src/BT_CAP/`: Core package directory.
    - `main.py`: Main script with augmentation logic.
    - `utils.py`: Utility functions (e.g., `np_to_sitk`, `ZScoreNormalize`).
    - `load.py`, `extract.py`, etc.: Modular processing steps.

## 🛠️ Development

- **Author**: Amin Tavallaii, Neurosurgeon, MD, MSc, MBA
- **Email**: dr.amin.tavallaii@gmail.com
- **License**: MIT
- **Repository**: [https://github.com/dramintavallaii/BT-CAP](https://github.com/dramintavallaii/BT-CAP)

### Contributing

Contributions are welcome! Please fork the repository, create a feature branch, and submit a pull request.

### Issues

Report bugs or suggest features via the [GitHub Issues](https://github.com/dramintavallaii/BT-CAP/issues) page.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🎉 Acknowledgments

- Thanks to the open-source community for tools like SimpleITK and scikit-image.
- Inspired by advancements in medical imaging and AI research.

---

*Last Updated: August 17, 2025*