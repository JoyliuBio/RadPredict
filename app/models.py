from app import db
from datetime import datetime, timezone, timedelta
import uuid
import json
from sqlalchemy.types import DateTime

# 定义中国时区
CHINA_TZ = timezone(timedelta(hours=8))

class Analysis(db.Model):
    id = db.Column(db.String(36), primary_key=True)
    status = db.Column(db.String(20), default='pending')  # pending, processing, completed, failed
    created_at = db.Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = db.Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    result_dir = db.Column(db.String(255))
    
    # Relationship with uploaded files
    files = db.relationship('UploadedFile', backref='analysis', lazy=True)
    
    def __init__(self, **kwargs):
        super(Analysis, self).__init__(**kwargs)
        if not self.id:
            self.id = str(uuid.uuid4())
    
    def to_dict(self):
        return {
            'id': self.id,
            'status': self.status,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'files': [file.name for file in self.files],
            'result_dir': self.result_dir
        }

class UploadedFile(db.Model):
    id = db.Column(db.String(36), primary_key=True)
    name = db.Column(db.String(255))
    path = db.Column(db.String(255))
    analysis_id = db.Column(db.String(36), db.ForeignKey('analysis.id'))
    uploaded_at = db.Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    processed = db.Column(db.Boolean, default=False)
    results = db.Column(db.Text)  # JSON string of results or metadata
    
    def __init__(self, **kwargs):
        super(UploadedFile, self).__init__(**kwargs)
        if not self.id:
            self.id = str(uuid.uuid4())
    
    def set_results(self, results_dict):
        self.results = json.dumps(results_dict)
    
    def get_results(self):
        if self.results:
            return json.loads(self.results)
        return {}

class ProcessStep(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    analysis_id = db.Column(db.String(36), db.ForeignKey('analysis.id'))
    file_id = db.Column(db.String(36), db.ForeignKey('uploaded_file.id'), nullable=True)
    step_name = db.Column(db.String(100))
    message = db.Column(db.Text)
    timestamp = db.Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    status = db.Column(db.String(20))  # success, failed, in_progress
    
    analysis = db.relationship('Analysis', backref=db.backref('steps', lazy='dynamic'))
