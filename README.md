# RadPredict

A web application for radiation resistance prediction in microorganisms based on genome analysis.

## Features

- Genome feature extraction and analysis
- Radiation resistance prediction based on machine learning models
- Web interface for easy submission and analysis
- Comprehensive modeling of radiation resistance traits

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/JoyliuBio/RadPredict.git
   cd RadPredict
   ```

2. Create the conda environment:
   ```
   conda env create -f environment.yml
   conda activate RadPredict
   ```

## Large Files (Git LFS)

This repository uses Git Large File Storage (LFS) for managing large files, particularly the DIAMOND database file (`Feature_set/OG_all_seqs.dmnd`, ~187MB).

### Option 1: Using Git LFS (Recommended)

1. Install Git LFS:
   ```
   # For macOS (using Homebrew)
   brew install git-lfs
   
   # For Ubuntu/Debian
   sudo apt-get install git-lfs
   
   # For CentOS/RHEL
   sudo yum install git-lfs
   
   # For Windows (using Chocolatey)
   choco install git-lfs
   ```

2. Initialize Git LFS:
   ```
   git lfs install
   ```

3. If you've already cloned the repository, run:
   ```
   cd RadPredict
   git lfs pull
   ```

4. Or clone with LFS support directly:
   ```
   git lfs clone https://github.com/JoyliuBio/RadPredict.git
   ```

### Option 2: Manual Download

If you're having issues with Git LFS, you can manually download the large file:

1. Download the DIAMOND database file directly from GitHub:
   - Visit: https://github.com/JoyliuBio/RadPredict/blob/main/Feature_set/OG_all_seqs.dmnd
   - Click the "Download" button

2. Replace the existing file in your local repository:
   - Navigate to the `Feature_set` directory in your cloned repository
   - The existing file is likely only a few KB (LFS pointer file)
   - Replace it with the downloaded file (should be ~187MB)
   - Verify replacement was successful by checking the file size

3. If the program reports an error about the DIAMOND database file, check:
   - File size should be approximately 187MB, not just a few KB
   - File name is exactly `OG_all_seqs.dmnd` (case sensitive)
   - File is in the correct location: `RadPredict/Feature_set/OG_all_seqs.dmnd`

## Usage

1. Start the web application:
   ```
   python run.py
   ```

2. Access the web interface at `http://localhost:5000` in your browser.

3. Upload your genome file in FASTA format and submit for analysis.

## Example Data

The repository includes example data in the `Example` directory:

- `Example/00`: Contains Deinococcus species examples with analysis results
- `Example/01`: Contains Bacillus species and PVC examples with analysis results

Each example folder contains:
- `input/`: Input genome files
- `results/`: Analysis results including:
  - Diamond search results
  - Prokka annotation results
  - Prediction results

You can use these examples to test the application or as templates for your own analyses.

## Model Information

The prediction model is based on machine learning algorithms trained on radiation-resistant microorganisms data. It analyzes genomic features including DNA repair genes, stress response systems, and genome structural characteristics.

## Directory Structure

- `app/`: Web application code
- `Feature_set/`: Feature extraction modules
- `static/`: Static web files (CSS, JS)
- `uploads/`: Temporary upload directory
- `temp/`: Temporary processing files
- `Example/`: Example data and results

## Troubleshooting

### CUDA Toolkit Installation Issue

If conda tries to install CUDA Toolkit (cudatoolkit-11.1.1) during environment creation, it is because XGBoost has CUDA as an optional dependency for GPU acceleration.

**Solution Options:**

1. **CPU-only Installation (Recommended for most users):**
   ```
   conda env create -f environment.yml --no-deps
   conda activate RadPredict
   conda install --file <(grep -v "xgboost" environment.yml | grep "dependencies:" -A 100) -c conda-forge -c bioconda
   conda install xgboost=1.6.2 -c conda-forge
   ```

2. **Skip CUDA Installation:**
   ```
   CONDA_OVERRIDE_CUDA='' conda env create -f environment.yml
   ```

3. **Allow CUDA Installation (for GPU acceleration):**
   - If you have an NVIDIA GPU and want to use GPU acceleration, you can proceed with the normal installation.

The application works perfectly well on CPU-only mode, and GPU acceleration is not required for normal operation.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 