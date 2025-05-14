#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import subprocess
import shutil
import argparse
import tempfile
import glob
import time

def run_command(cmd, description="Command"):
    """Execute shell command and handle errors"""
    print(f"Running {description}...")
    try:
        process = subprocess.run(cmd, shell=True, check=True, 
                                text=False, capture_output=True)
        print(f"{description} completed successfully")
        return process.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error in {description}: {e}")
        stdout = e.stdout.decode('utf-8', errors='replace') if e.stdout else ""
        stderr = e.stderr.decode('utf-8', errors='replace') if e.stderr else ""
        print(f"Command output: {stdout}")
        print(f"Command error: {stderr}")
        sys.exit(1)

def check_dependencies():
    """Check if required tools are installed"""
    dependencies = ["prokka", "diamond"]
    
    for tool in dependencies:
        try:
            subprocess.run(["which", tool], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print(f"✓ {tool} is installed")
        except subprocess.CalledProcessError:
            print(f"✗ {tool} is not installed. Please install it before proceeding.")
            sys.exit(1)

def check_diamond_db(diamond_db_path):
    """Verify that the Diamond database file exists and has the correct size"""
    if not os.path.exists(diamond_db_path):
        print(f"ERROR: Diamond database file not found: {diamond_db_path}")
        print("Please make sure to:")
        print("1. Install Git LFS and run 'git lfs pull' to download large files, or")
        print("2. Manually download the file from GitHub (see README.md for instructions)")
        print("\nFor more information, see the 'Large Files (Git LFS)' section in README.md")
        sys.exit(1)
    
    # Check if the file size is too small (LFS pointer file is typically a few hundred bytes)
    file_size = os.path.getsize(diamond_db_path)
    if file_size < 10000000:  # Less than 10MB
        print(f"WARNING: Diamond database file appears to be a Git LFS pointer (size: {file_size} bytes)")
        print("This is likely because Git LFS files were not properly downloaded.")
        print("Please make sure to:")
        print("1. Install Git LFS and run 'git lfs pull' to download large files, or")
        print("2. Manually download the file from GitHub (see README.md for instructions)")
        print("\nFor more information, see the 'Large Files (Git LFS)' section in README.md")
        sys.exit(1)
    
    print(f"✓ Diamond database verified: {diamond_db_path} ({file_size / 1024 / 1024:.1f} MB)")

def process_fasta_files(input_dir, output_dir, threads=1):
    """Process multiple FASTA files with Prokka"""
    # Create a dedicated directory for Prokka results
    prokka_output_dir = os.path.join(output_dir, "prokka_results")
    os.makedirs(prokka_output_dir, exist_ok=True)
    
    # Create directory for FAA files within prokka results
    faa_dir = os.path.join(prokka_output_dir, "faa_files")
    os.makedirs(faa_dir, exist_ok=True)
    
    # Find all fasta files in the input directory
    fasta_files = glob.glob(os.path.join(input_dir, "*.fasta"))
    fasta_files.extend(glob.glob(os.path.join(input_dir, "*.fa")))
    fasta_files.extend(glob.glob(os.path.join(input_dir, "*.fna")))
    
    # Also check for existing FAA files directly
    faa_files = glob.glob(os.path.join(input_dir, "*.faa"))
    
    # If no FASTA or FAA files found, exit
    if not fasta_files and not faa_files:
        print("No FASTA or FAA files found in the input directory.")
        sys.exit(1)
    
    # Process existing FAA files if any
    if faa_files:
        print(f"Found {len(faa_files)} existing FAA files. Skipping Prokka annotation.")
        for faa_file in faa_files:
            base_name = os.path.basename(faa_file).split('.')[0]
            target_path = os.path.join(faa_dir, f"{base_name}.faa")
            if not os.path.exists(target_path):
                shutil.copy(faa_file, faa_dir)
                print(f"Copied existing {faa_file} to {faa_dir}")
    
    # Process FASTA files with Prokka if any
    if fasta_files:
        print(f"Found {len(fasta_files)} FASTA files to process.")
        for fasta_file in fasta_files:
            base_name = os.path.basename(fasta_file).split('.')[0]
            prokka_dir = os.path.join(prokka_output_dir, f"prokka_{base_name}")
            faa_output = os.path.join(faa_dir, f"{base_name}.faa")
            
            # Skip if FAA file already exists
            if os.path.exists(faa_output):
                print(f"FAA file for {base_name} already exists at {faa_output}. Skipping annotation.")
                continue
            
            # If prokka directory exists but FAA not copied, check if FAA is there
            if os.path.exists(prokka_dir):
                faa_file = os.path.join(prokka_dir, f"{base_name}.faa")
                if os.path.exists(faa_file):
                    shutil.copy(faa_file, faa_dir)
                    print(f"Prokka output already exists. Copied {faa_file} to {faa_dir}")
                    continue
                else:
                    print(f"Prokka directory exists but no FAA file found. Re-running Prokka.")
            
            # Run Prokka
            cmd = f"prokka --outdir '{prokka_dir}' --prefix {base_name} --cpus {threads} '{fasta_file}'"
            run_command(cmd, f"Prokka annotation for {base_name}")
            
            # Copy the FAA file to the FAA directory
            faa_file = os.path.join(prokka_dir, f"{base_name}.faa")
            if os.path.exists(faa_file):
                shutil.copy(faa_file, faa_dir)
                print(f"Copied {faa_file} to {faa_dir}")
            else:
                print(f"Warning: FAA file not found for {base_name}")
    
    # Check if we have any FAA files in the output directory
    faa_files_output = glob.glob(os.path.join(faa_dir, "*.faa"))
    if not faa_files_output:
        print("No FAA files were generated or found. Cannot proceed.")
        sys.exit(1)
    
    print(f"Total of {len(faa_files_output)} FAA files ready for analysis.")
    print(f"Prokka results saved to: {prokka_output_dir}")
    return faa_dir

def run_diamond_analysis(faa_dir, diamond_script, output_dir):
    """Run diamond.sh script on the FAA files"""
    # Create a dedicated directory for Diamond results
    diamond_output_dir = os.path.join(output_dir, "diamond_results")
    os.makedirs(diamond_output_dir, exist_ok=True)
    
    # Get absolute paths
    faa_dir_abs = os.path.abspath(faa_dir)
    diamond_output_dir_abs = os.path.abspath(diamond_output_dir)
    diamond_script_abs = os.path.abspath(diamond_script)
    
    # Set paths for diamond resources
    script_dir = os.path.dirname(diamond_script_abs)
    diamond_db = os.path.join(script_dir, "Feature_set", "OG_all_seqs.dmnd")
    og_list = os.path.join(script_dir, "Feature_set", "important_ogs.txt")
    
    # Check that the Diamond database exists and has the correct size
    check_diamond_db(diamond_db)
    
    # Create matches directory within diamond results directory
    matches_dir = os.path.join(diamond_output_dir_abs, "matches")
    os.makedirs(matches_dir, exist_ok=True)
    
    # Create a timestamped output CSV filename in the diamond results directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_csv = os.path.join(diamond_output_dir_abs, f"presence_matrix_{timestamp}.csv")
    
    # Make the diamond script executable if it's not already
    os.chmod(diamond_script_abs, 0o755)
    
    # Run the diamond script with all necessary parameters
    cmd = f"bash '{diamond_script_abs}' '{faa_dir_abs}' '{diamond_db}' '{og_list}' '{output_csv}' '{matches_dir}'"
    run_command(cmd, "Diamond analysis")
    
    # Check if output file was created
    if os.path.exists(output_csv):
        print(f"Diamond analysis completed successfully.")
        print(f"Diamond results saved to: {diamond_output_dir_abs}")
        return output_csv
    else:
        print(f"Warning: Diamond output file {output_csv} not found")
        return None

def main():
    parser = argparse.ArgumentParser(description="Process bacterial genomes for radiation resistance prediction")
    parser.add_argument("-i", "--input", required=True, help="Directory containing FASTA files")
    parser.add_argument("-o", "--output", required=True, help="Output directory")
    parser.add_argument("-d", "--diamond", required=True, help="Path to diamond.sh script")
    parser.add_argument("-t", "--threads", type=int, default=1, help="Number of CPU threads to use")
    
    args = parser.parse_args()
    
    # Check if required tools are installed
    check_dependencies()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Process FASTA files with Prokka
    print("\n=== Starting Prokka annotation ===")
    faa_dir = process_fasta_files(args.input, args.output, args.threads)
    
    # Run Diamond analysis
    print("\n=== Starting Diamond analysis ===")
    csv_output = run_diamond_analysis(faa_dir, args.diamond, args.output)
    
    if csv_output:
        print(f"\n=== Pipeline completed successfully ===")
        print(f"The feature presence/absence matrix has been saved to: {csv_output}")
        print(f"You can now use this file as input for the radiation resistance prediction model.")
    else:
        print("\n=== Pipeline completed with errors ===")

if __name__ == "__main__":
    main()