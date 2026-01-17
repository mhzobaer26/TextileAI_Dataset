# Copilot Instructions for TextileAI_Dataset

## Repository Overview

This repository contains a curated dataset of textile images for AI-based defect detection and quality control. The dataset is organized to facilitate machine learning model training and evaluation for textile manufacturing quality assurance.

## Dataset Structure

### Directory Organization

- **`defect/`**: Contains images of textile samples with defects (106 images)
  - Images are named with the pattern: `{id}_{type}_{category}.png`
  - Examples: `0001_002_00.png`, `0010_006_02.png`
  
- **`no_defect/`**: Contains images of defect-free textile samples organized in subdirectories (7 categories)
  - Each subdirectory represents a different textile sample batch
  - Subdirectory naming pattern: `{id}-{code}u` (e.g., `2306881-210020u`)
  - Images within subdirectories follow the pattern: `{id}_{subid}_{category}.png`

### File Naming Conventions

**Defect Images:**
- Format: `{sequential_id}_{defect_type}_{defect_category}.png`
- Sequential ID: 4-digit padded number (e.g., `0001`, `0106`)
- Defect type: 3-digit code identifying the type of defect
- Defect category: 2-digit code for classification

**No Defect Images:**
- Format: `{sequential_id}_{subid}_{quality_code}.png`
- Sequential ID: 4-digit padded number
- Subid: 3-digit identifier
- Quality code: 2-digit code (typically `05` for verified quality samples)

## Guidelines for Contributors

### When Working with This Repository

1. **Preserve Dataset Integrity**
   - Never modify existing images without explicit approval
   - Maintain the existing file naming conventions
   - Keep the directory structure intact

2. **Adding New Images**
   - Follow the established naming conventions precisely
   - Place defect images directly in the `defect/` directory
   - Place no-defect images in appropriate subdirectories within `no_defect/`
   - Ensure images are in PNG format

3. **Documentation Updates**
   - Update dataset statistics if adding or removing images
   - Document any new defect types or categories
   - Maintain changelog for dataset versions

4. **Code and Scripts**
   - If adding data processing scripts, place them in a separate `scripts/` directory
   - Document dependencies and usage instructions
   - Ensure scripts preserve original data and use copies for processing

5. **Data Quality**
   - Verify image quality before adding to the dataset
   - Ensure consistent image dimensions within categories when possible
   - Validate file naming matches the established patterns

## Best Practices

- **Version Control**: Commit images in logical groups (e.g., by defect type or source batch)
- **Commit Messages**: Use descriptive messages like "Add 10 fabric tear defect images from batch X"
- **Large Files**: Be mindful of repository size; consider Git LFS if adding many large images
- **Testing**: If adding data processing code, include unit tests and example outputs
- **Documentation**: Keep README.md updated with dataset usage instructions and statistics

## Common Tasks

### Analyzing Dataset Statistics
```bash
# Count images by category
ls defect/ | wc -l
ls -d no_defect/*/ | wc -l

# List all no-defect batches
ls no_defect/
```

### Validating File Names
Ensure all files follow the naming convention before committing. Image filenames should match the patterns described above.

## Important Notes

- This is a dataset repository, not a code repository. The primary content is image data.
- Backward compatibility is critical - existing model training pipelines depend on consistent file paths and naming.
- Always test any structural changes against existing data loading scripts before committing.
