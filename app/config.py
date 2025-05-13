import os
from dotenv import load_dotenv
from datetime import timedelta

basedir = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(basedir, '.env'))

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'hard-to-guess-string'
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///app.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # File storage paths
    UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'temp', 'uploads')
    RESULTS_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'temp', 'results')
    
    # Maximum file size (16 MB)
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024
    
    # Allowed file extensions
    ALLOWED_EXTENSIONS = {'fasta', 'fa', 'fna', 'faa'}
    
    # Model file path
    MODEL_PATH = os.environ.get('MODEL_PATH') or os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Feature_set', 'comprehensive_model.pkl')
    
    # Results expiration time (24 hours)
    RESULTS_EXPIRY = timedelta(hours=24)
