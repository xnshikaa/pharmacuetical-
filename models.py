from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash

db = SQLAlchemy()

class User(UserMixin, db.Model):
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    role = db.Column(db.String(50), default='user')
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime)
    
    # Relationships
    reports = db.relationship('Report', backref='user', lazy=True)
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
    
    def __repr__(self):
        return f'<User {self.username}>'

class Report(db.Model):
    __tablename__ = 'reports'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    batch_id = db.Column(db.Integer, db.ForeignKey('batches.id'), nullable=True)  # Added batch foreign key
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Basic Information
    tablet_name = db.Column(db.String(100), nullable=False)
    batch_number = db.Column(db.String(100), nullable=False)
    agency = db.Column(db.String(100), nullable=False)
    type = db.Column(db.String(100))
    composition = db.Column(db.String(100))
    size = db.Column(db.String(50))
    recall_needed = db.Column(db.String(10))
    
    # Contamination Details
    contaminant_type = db.Column(db.String(100))
    
    # CAPA Information
    previous_cases = db.Column(db.Text)
    immediate_actions = db.Column(db.Text)
    quality_impact = db.Column(db.Text)
    training_needed = db.Column(db.Text)
    additional_notes = db.Column(db.Text)
    
    # Report Content
    content = db.Column(db.Text, nullable=False)
    risk_level = db.Column(db.String(20))
    status = db.Column(db.String(20), default='draft')  # draft, submitted, approved, rejected
    
    # Image Analysis
    image_filename = db.Column(db.String(200))
    defects_detected = db.Column(db.Integer)
    
    # FDA Data
    fda_recalls = db.Column(db.JSON)
    fda_events = db.Column(db.JSON)
    fda_label = db.Column(db.JSON)
    recall_alert = db.Column(db.Boolean, default=False)
    recall_summary = db.Column(db.Text)
    
    # Relationships
    image_analysis = db.relationship('ImageAnalysis', backref='report', uselist=False)
    revisions = db.relationship('ReportRevision', backref='report', lazy=True)
    capa_actions = db.relationship('CAPAAction', backref='report', lazy=True)
    
    def __repr__(self):
        return f'<Report {self.id}: {self.tablet_name} - {self.batch_number}>'

class ImageAnalysis(db.Model):
    __tablename__ = 'image_analyses'
    
    id = db.Column(db.Integer, primary_key=True)
    report_id = db.Column(db.Integer, db.ForeignKey('reports.id'), nullable=False)
    
    # Image Information
    filename = db.Column(db.String(255), nullable=False)
    filepath = db.Column(db.String(500), nullable=False)
    analysis_timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Analysis Results
    total_defects = db.Column(db.Integer, default=0)
    analysis_summary = db.Column(db.Text)
    confidence_score = db.Column(db.Float)
    
    # Defect Details (JSON)
    defects_data = db.Column(db.JSON)
    
    def __repr__(self):
        return f'<ImageAnalysis {self.id}: {self.filename}>'

class ReportRevision(db.Model):
    __tablename__ = 'report_revisions'
    
    id = db.Column(db.Integer, primary_key=True)
    report_id = db.Column(db.Integer, db.ForeignKey('reports.id'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    user_feedback = db.Column(db.Text)
    original_content = db.Column(db.Text)
    revised_content = db.Column(db.Text)
    
    def __repr__(self):
        return f'<ReportRevision {self.id}>'

class CAPAAction(db.Model):
    __tablename__ = 'capa_actions'
    
    id = db.Column(db.Integer, primary_key=True)
    report_id = db.Column(db.Integer, db.ForeignKey('reports.id'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    
    # CAPA Details
    action_type = db.Column(db.String(50))  # corrective, preventive
    description = db.Column(db.Text, nullable=False)
    priority = db.Column(db.String(20))  # low, medium, high, critical
    status = db.Column(db.String(20), default='pending')  # pending, in_progress, completed, verified
    
    # Timeline
    due_date = db.Column(db.DateTime)
    completed_date = db.Column(db.DateTime)
    
    # Responsibility
    assigned_to = db.Column(db.String(100))
    department = db.Column(db.String(100))
    
    # Effectiveness
    effectiveness_score = db.Column(db.Float)
    verification_date = db.Column(db.DateTime)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f'<CAPAAction {self.id}: {self.action_type} - {self.status}>'

class AuditLog(db.Model):
    __tablename__ = 'audit_logs'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Action Details
    action = db.Column(db.String(200), nullable=False)
    resource_type = db.Column(db.String(50))  # report, user, capa, etc.
    resource_id = db.Column(db.Integer)
    details = db.Column(db.Text)
    
    # IP and Session
    ip_address = db.Column(db.String(45))
    user_agent = db.Column(db.String(500))
    
    def __repr__(self):
        return f'<AuditLog {self.id}: {self.action}>'

class Batch(db.Model):
    __tablename__ = 'batches'
    
    id = db.Column(db.Integer, primary_key=True)
    batch_number = db.Column(db.String(50), unique=True, nullable=False)
    tablet_name = db.Column(db.String(100), nullable=False)
    
    # Manufacturing Details
    manufacturing_date = db.Column(db.Date)
    expiry_date = db.Column(db.Date)
    quantity = db.Column(db.Integer)
    
    # Quality Metrics
    defect_rate = db.Column(db.Float)
    risk_score = db.Column(db.Float)
    
    # Status
    status = db.Column(db.String(20), default='active')  # active, quarantined, recalled
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    reports = db.relationship('Report', backref='batch_info', lazy=True)
    
    def __repr__(self):
        return f'<Batch {self.batch_number}: {self.tablet_name}>'

class Recommendation(db.Model):
    __tablename__ = 'recommendations'
    
    id = db.Column(db.Integer, primary_key=True)
    report_id = db.Column(db.Integer, db.ForeignKey('reports.id'), nullable=False)
    content = db.Column(db.Text, nullable=False)
    context = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<Recommendation {self.id}>'

class RiskTrajectory(db.Model):
    __tablename__ = 'risk_trajectories'
    
    id = db.Column(db.Integer, primary_key=True)
    report_id = db.Column(db.Integer, db.ForeignKey('reports.id'), nullable=False)
    risk_score = db.Column(db.Float)
    risk_level = db.Column(db.String(50))
    mapped_path = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<RiskTrajectory {self.id}>'

class RegulatoryQA(db.Model):
    __tablename__ = 'regulatory_qas'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    question = db.Column(db.Text, nullable=False)
    answer = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<RegulatoryQA {self.id}>'

class InvestigationTask(db.Model):
    __tablename__ = 'investigation_tasks'
    
    id = db.Column(db.Integer, primary_key=True)
    report_id = db.Column(db.Integer, db.ForeignKey('reports.id'))
    description = db.Column(db.String(255), nullable=False)
    status = db.Column(db.String(20), default='pending')  # pending, in_progress, complete
    assignee = db.Column(db.String(80))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    started_at = db.Column(db.DateTime)
    completed_at = db.Column(db.DateTime)
    due_date = db.Column(db.DateTime)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class InvestigationTemplate(db.Model):
    __tablename__ = 'investigation_templates'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    complaint_type = db.Column(db.String(100))
    steps = db.Column(db.Text)  # JSON or comma-separated list of steps
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class InvestigationComment(db.Model):
    __tablename__ = 'investigation_comments'
    
    id = db.Column(db.Integer, primary_key=True)
    report_id = db.Column(db.Integer, db.ForeignKey('reports.id'))
    user = db.Column(db.String(80))
    message = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    parent_id = db.Column(db.Integer, db.ForeignKey('investigation_comments.id'), nullable=True)

def init_db(app):
    db.init_app(app)
    with app.app_context():
        db.create_all() 