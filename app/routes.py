import os
import uuid
from datetime import datetime, timezone, timedelta
from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify, current_app, send_from_directory, send_file
from werkzeug.utils import secure_filename
from app import db
from app.models import Analysis, UploadedFile, ProcessStep
from app.tasks import process_files, prepare_download_package
import threading
import shutil

main_bp = Blueprint('main', __name__)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in current_app.config['ALLOWED_EXTENSIONS']

@main_bp.route('/')
def index():
    return render_template('index.html')

@main_bp.route('/upload', methods=['POST'])
def upload_files():
    # Add logs
    print("Starting file upload process")
    
    # Check if files were uploaded
    if 'files' not in request.files:
        return jsonify({'error': 'No files provided'}), 400
    
    files = request.files.getlist('files')
    
    # Check if any files were selected
    if not files or files[0].filename == '':
        return jsonify({'error': 'No files selected'}), 400
    
    # Create a new analysis record
    analysis = Analysis()
    
    # Create result directory for this analysis
    result_dir = os.path.join(current_app.config['RESULTS_FOLDER'], analysis.id)
    os.makedirs(result_dir, exist_ok=True)
    analysis.result_dir = result_dir
    
    db.session.add(analysis)
    db.session.commit()
    
    uploaded_files = []
    
    # Save each file
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_uuid = str(uuid.uuid4())
            
            # Create directory for this file
            file_dir = os.path.join(current_app.config['UPLOAD_FOLDER'], file_uuid)
            os.makedirs(file_dir, exist_ok=True)
            
            filepath = os.path.join(file_dir, filename)
            file.save(filepath)
            
            # Create record for uploaded file
            uploaded_file = UploadedFile(
                id=file_uuid,
                name=filename,
                path=filepath,
                analysis_id=analysis.id
            )
            
            db.session.add(uploaded_file)
            uploaded_files.append(uploaded_file)
    
    db.session.commit()
    
    print(f"Preparing to process analysis ID: {analysis.id}, with {len(uploaded_files)} files")
    
    
    print(f"Files uploaded. Use 'Run Analysis Now' button to start processing for analysis ID: {analysis.id}")
    
    return jsonify({
        'analysis_id': analysis.id,
        'status': 'files_uploaded',
        'message': f'Successfully uploaded {len(uploaded_files)} files. Click "Run Analysis Now" button to start processing.',
        'redirect': url_for('main.analysis_status', analysis_id=analysis.id)
    })

@main_bp.route('/analysis/<analysis_id>')
def analysis_status(analysis_id):
    analysis = Analysis.query.get_or_404(analysis_id)
    return render_template('results.html', analysis=analysis)

@main_bp.route('/api/status/<analysis_id>')
def api_analysis_status(analysis_id):
    analysis = Analysis.query.get_or_404(analysis_id)
    response = {
        'id': analysis.id,
        'status': analysis.status,
        'created_at': analysis.created_at.isoformat(),
        'updated_at': analysis.updated_at.isoformat(),
        'files': [{
            'id': file.id,
            'name': file.name,
            'processed': file.processed,
            'uploaded_at': file.uploaded_at.isoformat()
        } for file in analysis.files]
    }
    return jsonify(response)

@main_bp.route('/api/download/<analysis_id>')
def download_results(analysis_id):
    """Download all results as a ZIP package"""
    analysis = Analysis.query.get_or_404(analysis_id)
    
    if analysis.status != 'completed':
        return jsonify({'error': 'Analysis not completed yet'}), 400
    
    result_dir = os.path.join(analysis.result_dir, "results")
    if os.path.exists(result_dir):
        print(f"Result directory exists at: {result_dir}")
        print("Files in result directory:")
        for root, dirs, files in os.walk(result_dir):
            for file in files:
                print(f" - {os.path.join(root, file)}")
    else:
        print(f"Warning: Result directory does not exist: {result_dir}")
    
    print(f"Preparing download package for analysis: {analysis_id}")
    zip_path = prepare_download_package(analysis_id)
    
    if not zip_path or not os.path.exists(zip_path) or os.path.getsize(zip_path) == 0:
        print(f"Failed to create valid ZIP package for analysis: {analysis_id}")
        return jsonify({'error': 'Failed to prepare download package. No result files found.'}), 500
    
    print(f"Sending ZIP file: {zip_path}, size: {os.path.getsize(zip_path)} bytes")
    
    try:
        response = send_file(
            zip_path,
            mimetype='application/zip',
            as_attachment=True,
            download_name=f'radiation_analysis_{analysis_id}.zip'
        )
        
        
        @response.call_on_close
        def cleanup():
            try:
                if os.path.exists(zip_path):
                    os.remove(zip_path)
                    print(f"Cleaned up temporary file: {zip_path}")
            except Exception as e:
                print(f"Error during cleanup: {e}")
        
        return response
    
    except Exception as e:
        print(f"Error sending file: {e}")
        if os.path.exists(zip_path):
            os.remove(zip_path)
        return jsonify({'error': f'Error sending file: {str(e)}'}), 500

def format_datetime_to_china_tz(dt):

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone(timedelta(hours=8)))

@main_bp.route('/dashboard')
def dashboard():
    analyses = Analysis.query.order_by(Analysis.created_at.desc()).limit(20).all()
    
    for analysis in analyses:
        analysis.created_at_local = format_datetime_to_china_tz(analysis.created_at)
        analysis.updated_at_local = format_datetime_to_china_tz(analysis.updated_at)
    
    return render_template('dashboard.html', analyses=analyses)

@main_bp.route('/history')
def history():
    analyses = Analysis.query.order_by(Analysis.created_at.desc()).all()
    return render_template('history.html', analyses=analyses)

@main_bp.route('/download/<int:analysis_id>/<path:filename>')
def download_file(analysis_id, filename):
    analysis = Analysis.query.get_or_404(analysis_id)
    return send_from_directory(analysis.result_directory, filename, as_attachment=True)

@main_bp.route('/debug/<analysis_id>')
def debug_analysis(analysis_id):
    analysis = Analysis.query.get_or_404(analysis_id)
    
    # Get analysis details
    details = {
        'id': analysis.id,
        'status': analysis.status,
        'created_at': analysis.created_at,
        'updated_at': analysis.updated_at,
        'result_dir': analysis.result_dir,
        'files': []
    }
    
    # Check if result directory exists
    result_dir_exists = os.path.exists(analysis.result_dir)
    details['result_dir_exists'] = result_dir_exists
    
    # If directory exists, list its files
    if result_dir_exists:
        details['result_files'] = os.listdir(analysis.result_dir)
    
    # Get information for each file
    for file in analysis.files:
        file_info = {
            'id': file.id,
            'name': file.name,
            'path': file.path,
            'file_exists': os.path.exists(file.path),
            'processed': file.processed
        }
        
        # If file has been processed, add result information
        if file.processed and file.results:
            try:
                results = file.get_results()
                file_info['results'] = results
            except:
                file_info['results'] = 'Error parsing results'
        
        details['files'].append(file_info)
    
    return jsonify(details)

@main_bp.route('/api/progress/<analysis_id>')
def api_analysis_progress(analysis_id):
    analysis = Analysis.query.get_or_404(analysis_id)
    
    # Get all processing steps
    steps = ProcessStep.query.filter_by(analysis_id=analysis_id).order_by(ProcessStep.timestamp.desc()).all()
    
    # Convert to JSON format
    steps_json = []
    for step in steps:
        steps_json.append({
            'step_name': step.step_name,
            'message': step.message,
            'timestamp': step.timestamp.isoformat(),
            'status': step.status,
            'file_id': step.file_id
        })
    
    # Calculate overall progress
    progress = 0
    if steps:
        completed_files = UploadedFile.query.filter_by(
            analysis_id=analysis_id, 
            processed=True
        ).count()
        total_files = len(analysis.files)
        
        if total_files > 0:
            progress = int((completed_files / total_files) * 100)
    
    return jsonify({
        'analysis_id': analysis.id,
        'status': analysis.status,
        'progress': progress,
        'steps': steps_json
    })

@main_bp.route('/process_status/<analysis_id>')
def process_status(analysis_id):
    """View processing status in plain text format"""
    analysis = Analysis.query.get_or_404(analysis_id)
    steps = ProcessStep.query.filter_by(analysis_id=analysis_id).order_by(ProcessStep.timestamp.desc()).all()
    
    response = f"Analysis ID: {analysis.id}\n"
    response += f"Status: {analysis.status}\n"
    response += f"Created: {analysis.created_at}\n"
    response += f"Updated: {analysis.updated_at}\n"
    response += f"Files: {len(analysis.files)}\n\n"
    
    if steps:
        response += "Progress Steps:\n"
        for step in steps:
            response += f"[{step.timestamp}] {step.step_name} - {step.status}\n"
            response += f"  {step.message}\n"
    else:
        response += "No progress steps recorded.\n\n"
        response += "Possible issues:\n"
        response += "1. Backend thread is not running correctly\n"
        response += "2. Database connection issue\n"
        response += "3. Process is still initializing\n"
    
    return response, 200, {'Content-Type': 'text/plain; charset=utf-8'}

@main_bp.route('/run_analysis_now/<analysis_id>', methods=['POST'])
def run_analysis_now(analysis_id):
    """在前台同步运行分析"""
    try:
        # Check analysis status first
        analysis = Analysis.query.get_or_404(analysis_id)
        
        # Prevent duplicate analysis
        if analysis.status == 'processing':
            return jsonify({'success': False, 'error': 'Analysis is already in progress'})
        elif analysis.status == 'completed':
            return jsonify({'success': False, 'error': 'Analysis has already been completed'})
        
        # If status allows, start analysis
        process_files(analysis_id)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@main_bp.route('/api/file_results/<file_id>')
def file_results(file_id):
    """Get detailed results for a specific file"""
    file = UploadedFile.query.get_or_404(file_id)
    analysis = Analysis.query.get_or_404(file.analysis_id)
    
    # Prepare result data
    result_data = {
        'file_id': file.id,
        'file_name': file.name,
        'analysis_id': analysis.id,
        'processed': file.processed,
        'prediction_csv': None,
        'visualization_images': []
    }
    
    # If file has been processed
    if file.processed:
        # Find result directory
        result_dir = os.path.join(analysis.result_dir, "results")
        prediction_dir = os.path.join(result_dir, "prediction_results")
        
        # Find prediction result CSV file
        prediction_files = []
        if os.path.exists(prediction_dir):
            for f in os.listdir(prediction_dir):
                if f.startswith('predicted_d10_values_') and f.endswith('.csv'):
                    prediction_files.append(os.path.join(prediction_dir, f))
        
        # If prediction file is found, read CSV data
        if prediction_files:
            try:
                import pandas as pd
                df = pd.read_csv(prediction_files[0])
                result_data['prediction_csv'] = df.to_dict(orient='records')
            except Exception as e:
                print(f"Error reading prediction CSV: {e}")
        
        # Find visualization images
        for root, dirs, files in os.walk(result_dir):
            for f in files:
                if f.endswith('.png') and ('visualization' in f or 'predicted_d10' in f):
                    # Create URL for web access
                    rel_path = os.path.relpath(os.path.join(root, f), result_dir)
                    url = f"/files/results/{analysis.id}/{rel_path}"
                    
                    result_data['visualization_images'].append({
                        'url': url,
                        'title': f.replace('_', ' ').replace('.png', '')
                    })
    
    return jsonify(result_data)

# Modify static file access endpoint
@main_bp.route('/files/results/<analysis_id>/<path:filename>')
def result_files(analysis_id, filename):
    """Access result files"""
    analysis = Analysis.query.get_or_404(analysis_id)
    result_dir = os.path.join(analysis.result_dir, "results")
    return send_from_directory(result_dir, filename)

@main_bp.route('/api/analysis_results/<analysis_id>')
def analysis_results(analysis_id):
    """Get unified results for the entire analysis"""
    analysis = Analysis.query.get_or_404(analysis_id)
    
    # Prepare result data
    result_data = {
        'analysis_id': analysis.id,
        'status': analysis.status,
        'prediction_csv': None,
        'visualization_images': []
    }
    
    # If analysis is completed
    if analysis.status == 'completed':
        # Find result directory
        result_dir = os.path.join(analysis.result_dir, "results")
        prediction_dir = os.path.join(result_dir, "prediction_results")
        
        # Find prediction result CSV file
        prediction_files = []
        if os.path.exists(prediction_dir):
            for f in os.listdir(prediction_dir):
                if f.startswith('predicted_d10_values_') and f.endswith('.csv'):
                    prediction_files.append(os.path.join(prediction_dir, f))
        
        # If prediction file is found, read CSV data
        if prediction_files:
            try:
                import pandas as pd
                df = pd.read_csv(prediction_files[0])
                result_data['prediction_csv'] = df.to_dict(orient='records')
            except Exception as e:
                print(f"Error reading prediction CSV: {e}")
        
        # Find visualization images
        for root, dirs, files in os.walk(result_dir):
            for f in files:
                if f.endswith('.png') and ('visualization' in f or 'predicted_d10' in f):
                    # Create URL for web access
                    rel_path = os.path.relpath(os.path.join(root, f), result_dir)
                    url = f"/files/results/{analysis.id}/{rel_path}"
                    
                    result_data['visualization_images'].append({
                        'url': url,
                        'title': f.replace('_', ' ').replace('.png', '')
                    })
    
    return jsonify(result_data)

@main_bp.route('/api/server_time')
def get_server_time():
    current_time = datetime.now(timezone.utc)
    return jsonify({
        'server_time': current_time.isoformat(),
        'timezone': 'UTC'
    })

@main_bp.route('/contact')
def contact():
    """Contact page with team information and contact details"""
    return render_template('contact.html')

@main_bp.route('/submit_contact', methods=['POST'])
def submit_contact():
    """Handle contact form submission"""
    name = request.form.get('name')
    email = request.form.get('email')
    subject = request.form.get('subject')
    message = request.form.get('message')
    
    flash('感谢您的留言！我们会尽快回复。', 'success')
    return redirect(url_for('main.contact'))

@main_bp.route('/about')
def about():
    """About page with information about the tool."""
    return render_template('about.html')

@main_bp.route('/delete_analyses', methods=['POST'])
def delete_analyses():
    data = request.get_json()
    
    if not data or 'analysis_ids' not in data:
        return jsonify({'success': False, 'error': 'Invalid request data'}), 400
    
    analysis_ids = data['analysis_ids']
    if not analysis_ids:
        return jsonify({'success': False, 'error': 'No analysis IDs provided'}), 400
    
    try:
        deleted_count = 0
        
        for analysis_id in analysis_ids:
            # 获取分析记录
            analysis = Analysis.query.filter_by(id=analysis_id).first()
            if analysis:
                # 删除相关文件
                try:
                    # 删除上传文件
                    for file in analysis.files:
                        file_dir = os.path.dirname(file.path)
                        if os.path.exists(file_dir):
                            shutil.rmtree(file_dir)
                    
                    # 删除结果目录
                    if analysis.result_dir and os.path.exists(analysis.result_dir):
                        shutil.rmtree(analysis.result_dir)
                except Exception as e:
                    print(f"Error deleting files for analysis {analysis_id}: {str(e)}")
                
                # 从数据库删除记录
                db.session.delete(analysis)
                deleted_count += 1
        
        db.session.commit()
        return jsonify({'success': True, 'message': f'Successfully deleted {deleted_count} analyses'})
    
    except Exception as e:
        db.session.rollback()
        print(f"Error deleting analyses: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500
