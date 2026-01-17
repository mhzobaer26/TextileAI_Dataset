# TextileAI Dataset

A dataset of textile images for machine learning tasks, containing defective and non-defective textile samples.

## Dataset Description

This repository contains a collection of textile images organized into two main categories:

- **Defect images**: Textile samples containing various types of defects (106 images)
- **No defect images**: Non-defective textile samples (141 images across multiple subdirectories)

The dataset is designed for binary classification tasks to detect defects in textile manufacturing.

## Current Structure

```
TextileAI_Dataset/
├── defect/                    # Defect images (PNG files)
│   ├── 0001_002_00.png
│   ├── 0002_002_00.png
│   └── ... (106 images total)
└── no_defect/                 # No defect images (organized in subdirectories)
    ├── 2306881-210020u/
    ├── 2306894-210033u/
    ├── 2311517-195063u/
    ├── 2311694-1930c7u/
    ├── 2311694-2040n7u/
    ├── 2311980-185026u/
    └── 2608691-202020u/
```

## Reorganization for Machine Learning

To prepare the dataset for machine learning tasks, we provide a reorganization script that splits the data into train/validation/test sets.

### Prerequisites

- Python 3.6 or higher
- Standard library only (no external dependencies required)

The script uses only Python standard library modules:
- `os` - for file system operations
- `shutil` - for file copying
- `random` - for shuffling with reproducibility
- `argparse` - for command-line interface
- `glob` - for file pattern matching
- `pathlib` - for path handling

### Usage

#### Basic Usage

Run the script with default parameters (70% train, 15% validation, 15% test):

```bash
python reorganize_dataset.py
```

This will create a new `dataset/` directory with the following structure:

```
dataset/
├── train/
│   ├── defect/          # ~74 images (70%)
│   └── no_defect/       # ~99 images (70%)
├── validation/
│   ├── defect/          # ~16 images (15%)
│   └── no_defect/       # ~21 images (15%)
└── test/
    ├── defect/          # ~16 images (15%)
    └── no_defect/       # ~21 images (15%)
```

#### Advanced Usage

Customize the split ratios and other parameters:

```bash
python reorganize_dataset.py \
  --train-ratio 0.80 \
  --val-ratio 0.10 \
  --test-ratio 0.10 \
  --seed 42 \
  --output-dir my_dataset
```

#### Command-Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--defect-dir` | `defect` | Path to defect images directory |
| `--no-defect-dir` | `no_defect` | Path to no_defect images directory |
| `--output-dir` | `dataset` | Output directory for reorganized dataset |
| `--train-ratio` | `0.70` | Ratio of training data (0.0-1.0) |
| `--val-ratio` | `0.15` | Ratio of validation data (0.0-1.0) |
| `--test-ratio` | `0.15` | Ratio of test data (0.0-1.0) |
| `--seed` | `42` | Random seed for reproducibility |
| `--clean` | `False` | Remove existing output directory before reorganizing |

**Note**: The sum of `--train-ratio`, `--val-ratio`, and `--test-ratio` must equal 1.0.

#### Examples

1. **Clean reorganization** (remove existing dataset folder first):
   ```bash
   python reorganize_dataset.py --clean
   ```

2. **Custom split ratios** (80/10/10):
   ```bash
   python reorganize_dataset.py --train-ratio 0.80 --val-ratio 0.10 --test-ratio 0.10
   ```

3. **Different output location**:
   ```bash
   python reorganize_dataset.py --output-dir ml_dataset
   ```

4. **Different random seed** for a different split:
   ```bash
   python reorganize_dataset.py --seed 123
   ```

### Split Strategy

The reorganization script:

1. **Collects images** from both `defect/` and `no_defect/` directories
2. **Shuffles** images randomly with a fixed seed for reproducibility
3. **Splits** images according to specified ratios (default: 70/15/15)
4. **Maintains class balance** by splitting each class independently
5. **Copies** images to preserve originals (does not move or delete)
6. **Handles duplicates** by adding numeric suffixes if needed

### Output Summary

After running the script, you'll see a summary like:

```
============================================================
DATASET REORGANIZATION SUMMARY
============================================================

Original Dataset:
  Defect images:     106
  No defect images:  141
  Total images:      247

Split Distribution:
Split           Defect     No Defect    Total     
--------------------------------------------------
train           74         98           172       
validation      16         22           38        
test            16         21           37        
--------------------------------------------------
Total           106        141          247       

Percentage Distribution:
Split           Defect %    No Defect %
----------------------------------------
train              69.8%       69.5%
validation         15.1%       15.6%
test               15.1%       14.9%
============================================================
```

### Idempotency

The script is designed to be idempotent and safe:

- If the output directory already exists, you'll be prompted to continue or abort
- Use `--clean` flag to automatically remove the existing output directory
- Original images are never modified or deleted (copies only)
- Duplicate filenames are handled automatically with numeric suffixes

## Using the Reorganized Dataset

After reorganization, you can use the dataset with popular ML frameworks:

### PyTorch Example

```python
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])

# Load datasets
train_dataset = datasets.ImageFolder('dataset/train', transform=transform)
val_dataset = datasets.ImageFolder('dataset/validation', transform=transform)
test_dataset = datasets.ImageFolder('dataset/test', transform=transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
```

### TensorFlow/Keras Example

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Create data generators
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

train_generator = datagen.flow_from_directory(
    'dataset/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

val_generator = datagen.flow_from_directory(
    'dataset/validation',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

test_generator = datagen.flow_from_directory(
    'dataset/test',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)
```

## Dataset Statistics

- **Total images**: 247
- **Defect images**: 106 (42.9%)
- **No defect images**: 141 (57.1%)
- **Image format**: PNG
- **Class distribution**: Slightly imbalanced (consider using class weights in training)

## Notes

- The dataset is relatively small; consider using data augmentation techniques
- Images may have different resolutions; preprocessing is recommended
- The `--seed` parameter ensures reproducible splits across runs
- Original images are preserved; the reorganized dataset is a copy

## License

Please check with the dataset owner for licensing information.

## Citation

If you use this dataset in your research, please cite appropriately.

## Contact

For questions or issues, please open an issue on the repository.
