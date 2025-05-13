import os
import sys
import subprocess
import time
from datetime import datetime
from flask import current_app
from app import db
from app.models import Analysis, UploadedFile, ProcessStep
import shutil
import zipfile
import tempfile

# Add script directory to system path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'scripts'))

# In actual application, this should be a Celery task
def run_analysis_task(analysis_id):
    """Run genome analysis task"""
    with current_app.app_context():
        analysis = Analysis.query.get(analysis_id)
        if not analysis:
            return
        
        try:
            # Update status to processing
            analysis.status = 'processing'
            db.session.commit()
            
            # Get file paths
            file_path = analysis.original_file_path
            upload_dir = os.path.dirname(file_path)
            result_dir = analysis.result_directory
            
            # Step 1: Run genome_analysis.py (synchronous execution required)
            genome_script = os.path.join(current_app.root_path, '..', 'scripts', 'genome_analysis.py')
            diamond_script = os.path.join(current_app.root_path, '..', 'scripts', 'diamond.sh')
            
            cmd = [
                'python', genome_script,
                '-i', upload_dir,
                '-o', result_dir,
                '-d', diamond_script,
                '-t', '10' 
            ]
            
            process = subprocess.run(cmd, capture_output=True, text=True)
            
            if process.returncode != 0:
                raise Exception(f"Genome analysis failed: {process.stderr}")
            
            # Find presence_matrix files
            presence_matrix_files = [f for f in os.listdir(result_dir) if f.startswith('presence_matrix_') and f.endswith('.csv')]
            
            if not presence_matrix_files:
                raise Exception("Feature matrix file not found")
            
            presence_matrix_path = os.path.join(result_dir, presence_matrix_files[0])
            
            # Step 2: Run Prediction_model.py
            prediction_script = os.path.join(current_app.root_path, '..', 'scripts', 'Prediction_model.py')
            
            cmd = [
                'python', prediction_script,
                '-i', presence_matrix_path,
                '-o', result_dir
            ]
            
            process = subprocess.run(cmd, capture_output=True, text=True)
            
            if process.returncode != 0:
                raise Exception(f"Prediction model execution failed: {process.stderr}")
            
            # Find prediction result files
            prediction_files = [f for f in os.listdir(result_dir) if f.startswith('predicted_d10_values_') and f.endswith('.csv')]
            
            if not prediction_files:
                raise Exception("Prediction result file not found")
            
            prediction_path = os.path.join(result_dir, prediction_files[0])
            
            # Read prediction results to get D10 values
            import pandas as pd
            predictions_df = pd.read_csv(prediction_path)
            mean_d10 = predictions_df['Predicted_D10'].mean()
            
            # Find visualization charts
            viz_images = [f for f in os.listdir(result_dir) if f.startswith('predicted_d10_visualization_') and f.endswith('.png')]
            
            # Update analysis record
            analysis.status = 'completed'
            analysis.completed_at = datetime.utcnow()
            analysis.predicted_d10 = mean_d10
            analysis.result_file = prediction_files[0] if prediction_files else None
            analysis.result_image = viz_images[0] if viz_images else None
            
            db.session.commit()
            
        except Exception as e:
            # Handle errors
            analysis.status = 'failed'
            analysis.error_message = str(e)
            db.session.commit()
            print(f"Analysis task failed: {e}")

def add_progress_step(analysis_id, step_name, message, status, file_id=None):
    """Add a processing step record"""
    try:
        # Ensure analysis exists before adding progress
        analysis = Analysis.query.get(analysis_id)
        if not analysis:
            print(f"Warning: Attempting to add progress for non-existent analysis {analysis_id}")
            return
        
        # Create progress step
        step = ProcessStep(
            analysis_id=analysis_id,
            file_id=file_id,
            step_name=step_name,
            message=message,
            status=status
        )
        db.session.add(step)
        db.session.commit()
        print(f"[Progress] {step_name}: {message}")
    except Exception as e:
        print(f"Error: Unable to add progress step: {e}")
        try:
            db.session.rollback()
        except:
            pass

def process_files(analysis_id):
    """Process all uploaded files for analysis"""
    
    print(f"Starting analysis {analysis_id}")
    
    # Get the analysis record
    analysis = Analysis.query.get(analysis_id)
    if not analysis:
        print(f"Analysis {analysis_id} not found")
        return
    
    try:
        # Update status to processing
        analysis.status = 'processing'
        db.session.commit()
        
        add_progress_step(analysis.id, "Initialization", "Starting analysis", "in_progress")
        
        # Create a common directory for all input files
        input_dir = os.path.join(analysis.result_dir, "input")
        os.makedirs(input_dir, exist_ok=True)
        
        # Copy all uploaded files to the common directory
        for uploaded_file in analysis.files:
            # Create symbolic link or copy file
            dest_path = os.path.join(input_dir, uploaded_file.name)
            try:
                # Try creating symbolic link first (more efficient)
                os.symlink(uploaded_file.path, dest_path)
            except (OSError, AttributeError):
                # Fall back to copying if symlinks not supported
                shutil.copy2(uploaded_file.path, dest_path)
            
            add_progress_step(
                analysis.id,
                "File Preparation",
                f"Preparing file: {uploaded_file.name}",
                "in_progress",
                uploaded_file.id
            )
        
        # Create results directory
        result_dir = os.path.join(analysis.result_dir, "results")
        os.makedirs(result_dir, exist_ok=True)
        
        # Step 1: Run genome_analysis.py for genomic annotation
        add_progress_step(
            analysis.id,
            "Genome Analysis",
            "Running genome analysis",
            "in_progress"
        )
        
        # Get paths for scripts
        genome_script = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'genome_analysis.py')
        diamond_script = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'diamond.sh')
        
        # Run genome analysis
        cmd = [
            sys.executable,
            genome_script,
            "-i", input_dir,
            "-o", result_dir,
            "-d", diamond_script,
            "-t", "10"
        ]
        
        try:
            print(f"Running genome analysis command: {' '.join(cmd)}")
            print(f"Files in input directory: {os.listdir(input_dir)}")
            
            genome_result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            print(f"Genome analysis completed with return code: {genome_result.returncode}")
            
            add_progress_step(
                analysis.id,
                "Genome Analysis",
                "Genome analysis completed successfully",
                "success"
            )
            
        except subprocess.CalledProcessError as e:
            error_msg = f"Genome analysis failed: {e.stderr}"
            add_progress_step(analysis.id, "Genome Analysis Error", error_msg, "failed")
            raise Exception(error_msg)
        
        # Step 2: Find the generated feature matrix
        add_progress_step(
            analysis.id,
            "Feature Extraction",
            "Processing feature matrix",
            "in_progress"
        )
        
        # Find feature matrix file using recursive search
        presence_matrix_files = []
        for root, dirs, files in os.walk(result_dir):
            for file in files:
                if file.startswith('presence_matrix_') and file.endswith('.csv'):
                    presence_matrix_files.append(os.path.join(root, file))
        
        if not presence_matrix_files:
            # Print directory structure for debugging
            print("Directory structure of result_dir:")
            print_directory_structure(result_dir)
            
            error_msg = "Feature matrix file not found after genome analysis"
            add_progress_step(analysis.id, "Feature Error", error_msg, "failed")
            raise Exception(error_msg)
        
        presence_matrix_path = presence_matrix_files[0]
        add_progress_step(
            analysis.id,
            "Feature Extraction",
            "Feature matrix processed successfully",
            "success"
        )
        
        # Step 3: Run Prediction_model.py on the feature matrix
        add_progress_step(
            analysis.id,
            "Prediction Model",
            "Running radiation resistance prediction model",
            "in_progress"
        )
        
        # Run prediction model
        prediction_dir = os.path.join(result_dir, "prediction_results")
        os.makedirs(prediction_dir, exist_ok=True)
        
        result = run_prediction_model(presence_matrix_path, prediction_dir)
        
        # Check for errors
        if 'error' in result:
            error_msg = result['error']
            add_progress_step(analysis.id, "Model Error", f"Prediction model returned error: {error_msg}", "failed")
            raise Exception(error_msg)
        
        # Update all files as processed
        for uploaded_file in analysis.files:
            uploaded_file.processed = True
            uploaded_file.set_results(result)
        
        # Update analysis status
        analysis.status = 'completed'
        db.session.commit()
        
        add_progress_step(analysis.id, "Completed", "Analysis completed successfully", "success")
        
        print(f"Analysis {analysis_id} completed successfully")
        
    except Exception as e:
        error_msg = str(e)
        print(f"Error in analysis {analysis_id}: {error_msg}")
        analysis.status = 'failed'
        db.session.commit()
        add_progress_step(analysis.id, "Error", f"Error during analysis: {error_msg}", "failed")

def process_single_file(uploaded_file, analysis):
    """Process a single uploaded file"""
    
    try:
        # Create a directory for this file's results
        file_result_dir = os.path.join(analysis.result_dir, uploaded_file.id)
        os.makedirs(file_result_dir, exist_ok=True)
        
        add_progress_step(
            analysis.id,
            "Preparation",
            f"Starting to process file: {uploaded_file.name}",
            "in_progress",
            uploaded_file.id
        )
        
        # Step 1: Run genome_analysis.py for annotation and feature extraction
        add_progress_step(
            analysis.id,
            "Genome Annotation",
            f"Running genome annotation for: {uploaded_file.name}",
            "in_progress", 
            uploaded_file.id
        )
        
        # Get paths for scripts
        genome_script = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'genome_analysis.py')
        diamond_script = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'diamond.sh')
        
        # Create input directory for the single file
        input_dir = os.path.dirname(uploaded_file.path)
        
        # Run genome analysis
        cmd = [
            sys.executable,
            genome_script,
            "-i", input_dir,
            "-o", file_result_dir,
            "-d", diamond_script,
            "-t", "10"  # Use 4 threads
        ]
        
        try:
            print(f"Running genome analysis command: {' '.join(cmd)}")
            genome_result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            print(f"Genome analysis completed with return code: {genome_result.returncode}")
        except subprocess.CalledProcessError as e:
            error_msg = f"Genome analysis failed: {e.stderr}"
            add_progress_step(
                analysis.id,
                "Genome Analysis Error",
                error_msg,
                "failed",
                uploaded_file.id
            )
            raise Exception(error_msg)
        
        # Step 2: Find the generated feature matrix
        add_progress_step(
            analysis.id,
            "Feature Extraction",
            "Looking for generated feature matrix",
            "in_progress",
            uploaded_file.id
        )
        
        # Find all possible presence matrix files, regardless of subdirectory
        presence_matrix_files = []
        for root, dirs, files in os.walk(file_result_dir):
            for file in files:
                if file.startswith('presence_matrix_') and file.endswith('.csv'):
                    presence_matrix_files.append(os.path.join(root, file))
        
        if not presence_matrix_files:
            error_msg = "Feature matrix file not found after genome analysis"
            add_progress_step(analysis.id, "Feature Error", error_msg, "failed", uploaded_file.id)
            raise Exception(error_msg)
        
        # Use the first found file
        presence_matrix_path = presence_matrix_files[0]
        
        # Step 3: Run Prediction_model.py on the feature matrix
        add_progress_step(
            analysis.id,
            "Running Model",
            f"Starting prediction model on the feature matrix",
            "in_progress", 
            uploaded_file.id
        )
        
        # Create a subdirectory for prediction results
        prediction_dir = os.path.join(file_result_dir, "prediction_results")
        os.makedirs(prediction_dir, exist_ok=True)
        
        result = run_prediction_model(presence_matrix_path, prediction_dir)
        
        # Check for errors
        if 'error' in result:
            error_msg = result['error']
            add_progress_step(
                analysis.id,
                "Model Error",
                f"Prediction model returned error: {error_msg}",
                "failed",
                uploaded_file.id
            )
            raise Exception(error_msg)
        
        # Update the file record
        uploaded_file.processed = True
        uploaded_file.set_results(result)
        db.session.commit()
        
        add_progress_step(
            analysis.id,
            "Processing Complete",
            f"File processing completed: {uploaded_file.name}",
            "success",
            uploaded_file.id
        )
        
        return True
    
    except Exception as e:
        error_msg = str(e)
        add_progress_step(
            analysis.id,
            "Processing Error",
            f"Error processing file: {error_msg}",
            "failed",
            uploaded_file.id
        )
        raise

def run_prediction_model(input_file, output_dir):
    """Run the prediction model on the input file"""
    
    # Get the path to the model file and script
    model_path = current_app.config['MODEL_PATH']
    script_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Prediction_model.py')
    
    # Prepare the command
    cmd = [
        sys.executable,
        script_path,  # Use absolute path
        "-i", input_file,
        "-o", output_dir
    ]
    
    # Run the command
    try:
        print(f"Running command: {' '.join(cmd)}")  # Add debugging
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        print(f"Command completed with return code: {result.returncode}")
        
        # Return the results
        return {
            'stdout': result.stdout,
            'stderr': result.stderr,
            'returncode': result.returncode,
            'output_files': os.listdir(output_dir)
        }
    
    except subprocess.CalledProcessError as e:
        print(f"Error running prediction model: {e}")
        print(f"Command stdout: {e.stdout}")
        print(f"Command stderr: {e.stderr}")
        return {
            'error': str(e),
            'stdout': e.stdout,
            'stderr': e.stderr,
            'returncode': e.returncode
        }

def prepare_download_package(analysis_id):
    """Prepare a ZIP file with all analysis results"""
    
    analysis = Analysis.query.get(analysis_id)
    if not analysis or analysis.status != 'completed':
        return None
    
    # Create a temporary file for the ZIP
    fd, zip_path = tempfile.mkstemp(suffix='.zip')
    os.close(fd)
    
    try:
        # Create the ZIP file
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add files from the shared results directory
            result_dir = os.path.join(analysis.result_dir, "results")
            
            if os.path.exists(result_dir):
                print(f"Adding files from result directory: {result_dir}")
                
                # Walk through the result directory and add all files
                for root, dirs, files in os.walk(result_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        # Create a relative path for the ZIP
                        rel_path = os.path.relpath(file_path, analysis.result_dir)
                        print(f"Adding file to ZIP: {rel_path}")
                        # Add to ZIP with the relative path
                        zipf.write(file_path, rel_path)
            else:
                print(f"Result directory not found: {result_dir}")
                
            # Also add original uploaded files for reference
            uploads_dir = os.path.join(analysis.result_dir, "input")
            if os.path.exists(uploads_dir):
                print(f"Adding uploaded files from: {uploads_dir}")
                for file in os.listdir(uploads_dir):
                    file_path = os.path.join(uploads_dir, file)
                    if os.path.isfile(file_path):
                        zipf.write(file_path, os.path.join("original_files", file))
            
            # Add a summary file with analysis information
            summary_fd, summary_path = tempfile.mkstemp(suffix='.txt')
            try:
                with os.fdopen(summary_fd, 'w') as f:
                    f.write(f"Analysis ID: {analysis.id}\n")
                    f.write(f"Analysis Date: {analysis.created_at}\n")
                    # Use updated_at instead of completed_at
                    f.write(f"Last Updated: {analysis.updated_at}\n\n")
                    f.write("Status: Completed\n\n")
                    f.write("Uploaded Files:\n")
                    for file in analysis.files:
                        f.write(f"- {file.name}\n")
                
                zipf.write(summary_path, "analysis_summary.txt")
            finally:
                if os.path.exists(summary_path):
                    os.remove(summary_path)
        
        # Check ZIP file size to verify content was written correctly
        zip_size = os.path.getsize(zip_path)
        if zip_size == 0:
            print("Warning: Generated ZIP file is empty!")
            return None
            
        print(f"Created ZIP package at: {zip_path}, size: {zip_size} bytes")
        return zip_path
        
    except Exception as e:
        print(f"Error creating ZIP package: {str(e)}")
        if os.path.exists(zip_path):
            os.remove(zip_path)
        return None

def print_directory_structure(directory, level=0):
    """Recursively print directory structure, for debugging"""
    if not os.path.exists(directory):
        print(f"Directory does not exist: {directory}")
        return
    
    print(f"{'  ' * level}[DIR] {os.path.basename(directory)}/")
    try:
        for item in os.listdir(directory):
            path = os.path.join(directory, item)
            if os.path.isdir(path):
                print_directory_structure(path, level + 1)
            else:
                print(f"{'  ' * (level + 1)}[FILE] {item}")
    except Exception as e:
        print(f"{'  ' * (level + 1)}Error reading directory: {e}")
