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
   conda activate radpredict
   ```

3. Install additional dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Start the web application:
   ```
   python run.py
   ```

2. Access the web interface at `http://localhost:5000` in your browser.

3. Upload your genome file in FASTA format and submit for analysis.

## Model Information

The prediction model is based on machine learning algorithms trained on radiation-resistant microorganisms data. It analyzes genomic features including DNA repair genes, stress response systems, and genome structural characteristics.

## Directory Structure

- `app/`: Web application code
- `Feature_set/`: Feature extraction modules
- `static/`: Static web files (CSS, JS)
- `uploads/`: Temporary upload directory
- `temp/`: Temporary processing files

## License

This project is licensed under the MIT License - see the LICENSE file for details. 