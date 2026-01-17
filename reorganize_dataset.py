#!/usr/bin/env python3
"""
Dataset Reorganization Script for TextileAI Dataset

This script reorganizes textile defect images into a standard ML dataset structure
with train/validation/test splits.
"""

import os
import shutil
import random
import argparse
import glob
from pathlib import Path


def find_images(directory, recursive=False):
    """
    Find all image files in a directory.
    
    Args:
        directory: Path to search for images
        recursive: If True, search recursively in subdirectories
        
    Returns:
        List of image file paths
    """
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']
    images = []
    
    if not os.path.exists(directory):
        print(f"Warning: Directory '{directory}' does not exist")
        return images
    
    if recursive:
        for ext in image_extensions:
            # Search recursively for image files
            pattern = os.path.join(directory, '**', f'*{ext}')
            images.extend(glob.glob(pattern, recursive=True))
            # Also check uppercase extensions
            pattern_upper = os.path.join(directory, '**', f'*{ext.upper()}')
            images.extend(glob.glob(pattern_upper, recursive=True))
    else:
        for ext in image_extensions:
            # Search only in the immediate directory
            pattern = os.path.join(directory, f'*{ext}')
            images.extend(glob.glob(pattern))
            # Also check uppercase extensions
            pattern_upper = os.path.join(directory, f'*{ext.upper()}')
            images.extend(glob.glob(pattern_upper))
    
    return sorted(list(set(images)))  # Remove duplicates and sort


def create_directory_structure(output_dir):
    """
    Create the train/validation/test directory structure.
    
    Args:
        output_dir: Base output directory path
    """
    splits = ['train', 'validation', 'test']
    classes = ['defect', 'no_defect']
    
    for split in splits:
        for class_name in classes:
            dir_path = os.path.join(output_dir, split, class_name)
            os.makedirs(dir_path, exist_ok=True)
            print(f"Created directory: {dir_path}")


def split_dataset(images, train_ratio, val_ratio, test_ratio, seed):
    """
    Split images into train/validation/test sets.
    
    Args:
        images: List of image paths
        train_ratio: Ratio for training set (e.g., 0.70)
        val_ratio: Ratio for validation set (e.g., 0.15)
        test_ratio: Ratio for test set (e.g., 0.15)
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary with 'train', 'validation', 'test' keys containing image lists
    """
    # Validate ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 0.001:
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")
    
    # Shuffle images with fixed seed for reproducibility
    random.seed(seed)
    shuffled_images = images.copy()
    random.shuffle(shuffled_images)
    
    # Calculate split indices
    n_total = len(shuffled_images)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    # Split the data
    train_images = shuffled_images[:n_train]
    val_images = shuffled_images[n_train:n_train + n_val]
    test_images = shuffled_images[n_train + n_val:]
    
    return {
        'train': train_images,
        'validation': val_images,
        'test': test_images
    }


def copy_images(split_dict, output_dir, class_name):
    """
    Copy images to the new directory structure.
    
    Args:
        split_dict: Dictionary with split names and image lists
        output_dir: Base output directory
        class_name: Name of the class (defect or no_defect)
    """
    for split_name, images in split_dict.items():
        dest_dir = os.path.join(output_dir, split_name, class_name)
        
        for img_path in images:
            filename = os.path.basename(img_path)
            dest_path = os.path.join(dest_dir, filename)
            
            # Handle duplicate filenames by adding a counter
            counter = 1
            base_name, ext = os.path.splitext(filename)
            while os.path.exists(dest_path):
                filename = f"{base_name}_{counter}{ext}"
                dest_path = os.path.join(dest_dir, filename)
                counter += 1
            
            shutil.copy2(img_path, dest_path)


def print_summary(defect_images, no_defect_images, defect_splits, no_defect_splits):
    """
    Print a summary of the dataset organization.
    
    Args:
        defect_images: List of defect images
        no_defect_images: List of no_defect images
        defect_splits: Split dictionary for defect images
        no_defect_splits: Split dictionary for no_defect images
    """
    print("\n" + "="*60)
    print("DATASET REORGANIZATION SUMMARY")
    print("="*60)
    
    print("\nOriginal Dataset:")
    print(f"  Defect images:     {len(defect_images)}")
    print(f"  No defect images:  {len(no_defect_images)}")
    print(f"  Total images:      {len(defect_images) + len(no_defect_images)}")
    
    print("\nSplit Distribution:")
    print(f"{'Split':<15} {'Defect':<10} {'No Defect':<12} {'Total':<10}")
    print("-" * 50)
    
    for split_name in ['train', 'validation', 'test']:
        defect_count = len(defect_splits[split_name])
        no_defect_count = len(no_defect_splits[split_name])
        total = defect_count + no_defect_count
        print(f"{split_name:<15} {defect_count:<10} {no_defect_count:<12} {total:<10}")
    
    total_defect = sum(len(defect_splits[s]) for s in defect_splits)
    total_no_defect = sum(len(no_defect_splits[s]) for s in no_defect_splits)
    grand_total = total_defect + total_no_defect
    
    print("-" * 50)
    print(f"{'Total':<15} {total_defect:<10} {total_no_defect:<12} {grand_total:<10}")
    
    print("\nPercentage Distribution:")
    print(f"{'Split':<15} {'Defect %':<12} {'No Defect %':<12}")
    print("-" * 40)
    
    for split_name in ['train', 'validation', 'test']:
        defect_pct = (len(defect_splits[split_name]) / len(defect_images) * 100) if defect_images else 0
        no_defect_pct = (len(no_defect_splits[split_name]) / len(no_defect_images) * 100) if no_defect_images else 0
        print(f"{split_name:<15} {defect_pct:>10.1f}%  {no_defect_pct:>10.1f}%")
    
    print("="*60 + "\n")


def main():
    """Main function to orchestrate dataset reorganization."""
    parser = argparse.ArgumentParser(
        description='Reorganize textile images into train/validation/test splits',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--defect-dir',
        default='defect',
        help='Path to defect images directory'
    )
    
    parser.add_argument(
        '--no-defect-dir',
        default='no_defect',
        help='Path to no_defect images directory'
    )
    
    parser.add_argument(
        '--output-dir',
        default='dataset',
        help='Output directory for reorganized dataset'
    )
    
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.70,
        help='Ratio of training data (0.0-1.0)'
    )
    
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.15,
        help='Ratio of validation data (0.0-1.0)'
    )
    
    parser.add_argument(
        '--test-ratio',
        type=float,
        default=0.15,
        help='Ratio of test data (0.0-1.0)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    parser.add_argument(
        '--clean',
        action='store_true',
        help='Remove existing output directory before reorganizing'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.train_ratio + args.val_ratio + args.test_ratio != 1.0:
        parser.error(f"Ratios must sum to 1.0, got {args.train_ratio + args.val_ratio + args.test_ratio}")
    
    print("="*60)
    print("TEXTILE DATASET REORGANIZATION")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Defect directory:     {args.defect_dir}")
    print(f"  No defect directory:  {args.no_defect_dir}")
    print(f"  Output directory:     {args.output_dir}")
    print(f"  Train ratio:          {args.train_ratio}")
    print(f"  Validation ratio:     {args.val_ratio}")
    print(f"  Test ratio:           {args.test_ratio}")
    print(f"  Random seed:          {args.seed}")
    
    # Clean output directory if requested
    if args.clean and os.path.exists(args.output_dir):
        print(f"\nRemoving existing output directory: {args.output_dir}")
        shutil.rmtree(args.output_dir)
    
    # Check if output directory exists and is not empty
    if os.path.exists(args.output_dir):
        if os.listdir(args.output_dir):
            print(f"\nWarning: Output directory '{args.output_dir}' already exists and is not empty.")
            print("Use --clean flag to remove it first, or choose a different output directory.")
            response = input("Continue anyway? (y/n): ")
            if response.lower() != 'y':
                print("Aborted.")
                return
    
    # Find images
    print("\nFinding images...")
    defect_images = find_images(args.defect_dir, recursive=False)
    no_defect_images = find_images(args.no_defect_dir, recursive=True)
    
    print(f"Found {len(defect_images)} defect images")
    print(f"Found {len(no_defect_images)} no_defect images")
    
    if not defect_images and not no_defect_images:
        print("\nError: No images found in the specified directories!")
        return
    
    # Create directory structure
    print("\nCreating directory structure...")
    create_directory_structure(args.output_dir)
    
    # Split datasets
    print("\nSplitting datasets...")
    defect_splits = split_dataset(
        defect_images, 
        args.train_ratio, 
        args.val_ratio, 
        args.test_ratio, 
        args.seed
    ) if defect_images else {'train': [], 'validation': [], 'test': []}
    
    no_defect_splits = split_dataset(
        no_defect_images, 
        args.train_ratio, 
        args.val_ratio, 
        args.test_ratio, 
        args.seed
    ) if no_defect_images else {'train': [], 'validation': [], 'test': []}
    
    # Copy images
    print("\nCopying images to new structure...")
    if defect_images:
        print("  Copying defect images...")
        copy_images(defect_splits, args.output_dir, 'defect')
    
    if no_defect_images:
        print("  Copying no_defect images...")
        copy_images(no_defect_splits, args.output_dir, 'no_defect')
    
    # Print summary
    print_summary(defect_images, no_defect_images, defect_splits, no_defect_splits)
    
    print(f"Dataset reorganization complete! Output saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
