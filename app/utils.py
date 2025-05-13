import os
import shutil
from datetime import datetime, timezone
from flask import current_app
from app import db
from app.models import Analysis

def cleanup_old_files():
    """Clean up expired results and unused uploads"""
    
    # Get current time
    now = datetime.now(timezone.utc)
    
    # Find analyses with expired results
    expiry_time = current_app.config['RESULTS_EXPIRY']
    
    expired_analyses = Analysis.query.filter(
        Analysis.created_at < (now - expiry_time)
    ).all()
    
    for analysis in expired_analyses:
        # Delete result directory
        if analysis.result_dir and os.path.exists(analysis.result_dir):
            shutil.rmtree(analysis.result_dir)
        
        # Delete uploaded files
        for uploaded_file in analysis.files:
            if uploaded_file.path and os.path.exists(uploaded_file.path):
                # Get the directory containing the file
                file_dir = os.path.dirname(uploaded_file.path)
                if os.path.exists(file_dir):
                    shutil.rmtree(file_dir)
        
        # Delete database record
        db.session.delete(analysis)
    
    # Commit changes
    db.session.commit()
