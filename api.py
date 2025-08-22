from flask import Blueprint, request, jsonify, current_app
from flask_login import login_required, current_user
from models import db, User, Report, ImageAnalysis, CAPAAction, AuditLog, Batch
from datetime import datetime
import json
import os
from werkzeug.utils import secure_filename
from image_processor import defect_detector
import requests

api = Blueprint('api', __name__)

def log_audit(action, resource_type=None, resource_id=None, details=None):
    """Log audit trail"""
    audit = AuditLog(
        user_id=current_user.id if current_user.is_authenticated else None,
        action=action,
        resource_type=resource_type,
        resource_id=resource_id,
        details=details,
        ip_address=request.remote_addr,
        user_agent=request.headers.get('User-Agent', '')
    )
    db.session.add(audit)
    db.session.commit()

# Authentication endpoints
@api.route('/auth/login', methods=['POST'])
def login():
    """User login API"""
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    
    user = User.query.filter_by(username=username).first()
    if user and user.check_password(password):
        user.last_login = datetime.utcnow()
        db.session.commit()
        log_audit('login', 'user', user.id)
        return jsonify({
            'success': True,
            'user': {
                'id': user.id,
                'username': user.username,
                'email': user.email,
                'role': user.role
            }
        })
    
    return jsonify({'success': False, 'message': 'Invalid credentials'}), 401

@api.route('/auth/register', methods=['POST'])
def register():
    """User registration API"""
    data = request.get_json()
    
    if User.query.filter_by(username=data['username']).first():
        return jsonify({'success': False, 'message': 'Username already exists'}), 400
    
    if User.query.filter_by(email=data['email']).first():
        return jsonify({'success': False, 'message': 'Email already exists'}), 400
    
    user = User(
        username=data['username'],
        email=data['email'],
        role=data.get('role', 'analyst')
    )
    user.set_password(data['password'])
    
    db.session.add(user)
    db.session.commit()
    
    log_audit('register', 'user', user.id)
    return jsonify({'success': True, 'message': 'User created successfully'})

# Reports API
@api.route('/reports', methods=['GET'])
@login_required
def get_reports():
    """Get all reports with pagination and filtering"""
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 10, type=int)
    status = request.args.get('status')
    agency = request.args.get('agency')
    risk_level = request.args.get('risk_level')
    
    query = Report.query
    
    if status:
        query = query.filter(Report.status == status)
    if agency:
        query = query.filter(Report.agency == agency)
    if risk_level:
        query = query.filter(Report.risk_level == risk_level)
    
    # Role-based filtering
    if current_user.role == 'viewer':
        query = query.filter(Report.user_id == current_user.id)
    
    reports = query.order_by(Report.timestamp.desc()).paginate(
        page=page, per_page=per_page, error_out=False
    )
    
    return jsonify({
        'reports': [{
            'id': report.id,
            'tablet_name': report.tablet_name,
            'batch_number': report.batch_number,
            'agency': report.agency,
            'risk_level': report.risk_level,
            'status': report.status,
            'timestamp': report.timestamp.isoformat(),
            'defects_detected': report.defects_detected,
            'user': report.user.username
        } for report in reports.items],
        'total': reports.total,
        'pages': reports.pages,
        'current_page': page
    })

@api.route('/reports/<int:report_id>', methods=['GET'])
@login_required
def get_report(report_id):
    """Get specific report details"""
    report = Report.query.get_or_404(report_id)
    
    # Check permissions
    if current_user.role == 'viewer' and report.user_id != current_user.id:
        return jsonify({'success': False, 'message': 'Access denied'}), 403
    
    return jsonify({
        'id': report.id,
        'tablet_name': report.tablet_name,
        'batch_number': report.batch_number,
        'agency': report.agency,
        'contaminant_type': report.contaminant_type,
        'composition': report.composition,
        'size_um': report.size_um,
        'recall_needed': report.recall_needed,
        'previous_cases': report.previous_cases,
        'immediate_actions': report.immediate_actions,
        'quality_impact': report.quality_impact,
        'training_needed': report.training_needed,
        'additional_notes': report.additional_notes,
        'content': report.content,
        'risk_level': report.risk_level,
        'status': report.status,
        'timestamp': report.timestamp.isoformat(),
        'image_filename': report.image_filename,
        'defects_detected': report.defects_detected,
        'user': report.user.username,
        'image_analysis': {
            'filename': report.image_analysis.filename,
            'total_defects': report.image_analysis.total_defects,
            'confidence_score': report.image_analysis.confidence_score,
            'analysis_summary': report.image_analysis.analysis_summary
        } if report.image_analysis else None,
        'revisions': [{
            'id': rev.id,
            'timestamp': rev.timestamp.isoformat(),
            'user_feedback': rev.user_feedback
        } for rev in report.revisions],
        'capa_actions': [{
            'id': capa.id,
            'action_type': capa.action_type,
            'description': capa.description,
            'priority': capa.priority,
            'status': capa.status,
            'due_date': capa.due_date.isoformat() if capa.due_date else None,
            'assigned_to': capa.assigned_to
        } for capa in report.capa_actions]
    })

@api.route('/reports', methods=['POST'])
@login_required
def create_report():
    """Create new report"""
    data = request.get_json()
    
    report = Report(
        user_id=current_user.id,
        tablet_name=data['tablet_name'],
        batch_number=data['batch_number'],
        agency=data['agency'],
        contaminant_type=data.get('contaminant_type'),
        composition=data.get('composition'),
        size_um=data.get('size_um'),
        recall_needed=data.get('recall_needed'),
        previous_cases=data.get('previous_cases'),
        immediate_actions=data.get('immediate_actions'),
        quality_impact=data.get('quality_impact'),
        training_needed=data.get('training_needed'),
        additional_notes=data.get('additional_notes'),
        content=data['content'],
        risk_level=data.get('risk_level'),
        image_filename=data.get('image_filename'),
        defects_detected=data.get('defects_detected', 0)
    )
    
    db.session.add(report)
    db.session.commit()
    
    log_audit('create_report', 'report', report.id)
    return jsonify({'success': True, 'report_id': report.id})

@api.route('/reports/<int:report_id>', methods=['PUT'])
@login_required
def update_report(report_id):
    """Update report"""
    report = Report.query.get_or_404(report_id)
    
    # Check permissions
    if current_user.role == 'viewer' and report.user_id != current_user.id:
        return jsonify({'success': False, 'message': 'Access denied'}), 403
    
    data = request.get_json()
    
    for field in ['tablet_name', 'batch_number', 'agency', 'content', 'risk_level', 'status']:
        if field in data:
            setattr(report, field, data[field])
    
    db.session.commit()
    
    log_audit('update_report', 'report', report.id)
    return jsonify({'success': True})

# Image Analysis API
@api.route('/images/upload', methods=['POST'])
@login_required
def upload_image():
    """Upload and analyze image"""
    if 'image' not in request.files:
        return jsonify({'success': False, 'message': 'No image provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'success': False, 'message': 'No file selected'}), 400
    
    if file:
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Analyze image
        defects_analysis = defect_detector.detect_defects(filepath)
        defect_summary = defect_detector.generate_defect_summary(defects_analysis)
        
        return jsonify({
            'success': True,
            'filename': filename,
            'filepath': filepath,
            'analysis': defects_analysis,
            'summary': defect_summary
        })
    
    return jsonify({'success': False, 'message': 'Invalid file'}), 400

@api.route('/images/analyze/<filename>', methods=['POST'])
@login_required
def analyze_image(filename):
    """Analyze existing image"""
    filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
    
    if not os.path.exists(filepath):
        return jsonify({'success': False, 'message': 'Image not found'}), 404
    
    defects_analysis = defect_detector.detect_defects(filepath)
    defect_summary = defect_detector.generate_defect_summary(defects_analysis)
    
    return jsonify({
        'success': True,
        'analysis': defects_analysis,
        'summary': defect_summary
    })

# CAPA Actions API
@api.route('/reports/<int:report_id>/capa', methods=['GET'])
@login_required
def get_capa_actions(report_id):
    """Get CAPA actions for a report"""
    report = Report.query.get_or_404(report_id)
    
    capa_actions = CAPAAction.query.filter_by(report_id=report_id).all()
    
    return jsonify({
        'capa_actions': [{
            'id': capa.id,
            'action_type': capa.action_type,
            'description': capa.description,
            'priority': capa.priority,
            'status': capa.status,
            'due_date': capa.due_date.isoformat() if capa.due_date else None,
            'completed_date': capa.completed_date.isoformat() if capa.completed_date else None,
            'assigned_to': capa.assigned_to,
            'department': capa.department,
            'effectiveness_score': capa.effectiveness_score
        } for capa in capa_actions]
    })

@api.route('/reports/<int:report_id>/capa', methods=['POST'])
@login_required
def create_capa_action(report_id):
    """Create CAPA action"""
    report = Report.query.get_or_404(report_id)
    data = request.get_json()
    
    capa = CAPAAction(
        report_id=report_id,
        user_id=current_user.id,
        action_type=data['action_type'],
        description=data['description'],
        priority=data.get('priority', 'medium'),
        assigned_to=data.get('assigned_to'),
        department=data.get('department'),
        due_date=datetime.fromisoformat(data['due_date']) if data.get('due_date') else None
    )
    
    db.session.add(capa)
    db.session.commit()
    
    log_audit('create_capa', 'capa_action', capa.id)
    return jsonify({'success': True, 'capa_id': capa.id})

@api.route('/capa/<int:capa_id>', methods=['PUT'])
@login_required
def update_capa_action(capa_id):
    """Update CAPA action"""
    capa = CAPAAction.query.get_or_404(capa_id)
    data = request.get_json()
    
    for field in ['status', 'description', 'priority', 'assigned_to', 'department']:
        if field in data:
            setattr(capa, field, data[field])
    
    if 'completed_date' in data and data['completed_date']:
        capa.completed_date = datetime.fromisoformat(data['completed_date'])
    
    if 'effectiveness_score' in data:
        capa.effectiveness_score = data['effectiveness_score']
        capa.verification_date = datetime.utcnow()
    
    db.session.commit()
    
    log_audit('update_capa', 'capa_action', capa.id)
    return jsonify({'success': True})

# Analytics API
@api.route('/analytics/dashboard', methods=['GET'])
@login_required
def get_dashboard_analytics():
    """Get dashboard analytics"""
    # Total reports
    total_reports = Report.query.count()
    
    # Reports by status
    status_counts = db.session.query(
        Report.status, db.func.count(Report.id)
    ).group_by(Report.status).all()
    
    # Reports by risk level
    risk_counts = db.session.query(
        Report.risk_level, db.func.count(Report.id)
    ).group_by(Report.risk_level).all()
    
    # Recent activity
    recent_reports = Report.query.order_by(Report.timestamp.desc()).limit(5).all()
    
    # CAPA statistics
    total_capa = CAPAAction.query.count()
    pending_capa = CAPAAction.query.filter_by(status='pending').count()
    completed_capa = CAPAAction.query.filter_by(status='completed').count()
    
    return jsonify({
        'total_reports': total_reports,
        'status_distribution': dict(status_counts),
        'risk_distribution': dict(risk_counts),
        'recent_reports': [{
            'id': report.id,
            'tablet_name': report.tablet_name,
            'risk_level': report.risk_level,
            'timestamp': report.timestamp.isoformat()
        } for report in recent_reports],
        'capa_stats': {
            'total': total_capa,
            'pending': pending_capa,
            'completed': completed_capa
        }
    })

@api.route('/analytics/trends', methods=['GET'])
@login_required
def get_trends():
    """Get trend analytics"""
    # Monthly report counts
    monthly_reports = db.session.query(
        db.func.date_trunc('month', Report.timestamp).label('month'),
        db.func.count(Report.id)
    ).group_by('month').order_by('month').all()
    
    # Defect trends
    defect_trends = db.session.query(
        Report.contaminant_type,
        db.func.count(Report.id)
    ).group_by(Report.contaminant_type).all()
    
    return jsonify({
        'monthly_reports': [{'month': str(month), 'count': count} for month, count in monthly_reports],
        'defect_trends': [{'type': defect_type, 'count': count} for defect_type, count in defect_trends]
    })

# Batch Management API
@api.route('/batches', methods=['GET'])
@login_required
def get_batches():
    """Get all batches"""
    batches = Batch.query.all()
    
    return jsonify({
        'batches': [{
            'id': batch.id,
            'batch_number': batch.batch_number,
            'tablet_name': batch.tablet_name,
            'manufacturing_date': batch.manufacturing_date.isoformat() if batch.manufacturing_date else None,
            'expiry_date': batch.expiry_date.isoformat() if batch.expiry_date else None,
            'quantity': batch.quantity,
            'defect_rate': batch.defect_rate,
            'risk_score': batch.risk_score,
            'status': batch.status
        } for batch in batches]
    })

@api.route('/batches', methods=['POST'])
@login_required
def create_batch():
    """Create new batch"""
    data = request.get_json()
    
    batch = Batch(
        batch_number=data['batch_number'],
        tablet_name=data['tablet_name'],
        manufacturing_date=datetime.fromisoformat(data['manufacturing_date']) if data.get('manufacturing_date') else None,
        expiry_date=datetime.fromisoformat(data['expiry_date']) if data.get('expiry_date') else None,
        quantity=data.get('quantity'),
        defect_rate=data.get('defect_rate', 0.0),
        risk_score=data.get('risk_score', 0.0)
    )
    
    db.session.add(batch)
    db.session.commit()
    
    log_audit('create_batch', 'batch', batch.id)
    return jsonify({'success': True, 'batch_id': batch.id})

# Export API
@api.route('/reports/<int:report_id>/export', methods=['GET'])
@login_required
def export_report(report_id):
    """Export report as JSON"""
    report = Report.query.get_or_404(report_id)
    
    export_data = {
        'report_id': report.id,
        'tablet_name': report.tablet_name,
        'batch_number': report.batch_number,
        'agency': report.agency,
        'timestamp': report.timestamp.isoformat(),
        'content': report.content,
        'risk_level': report.risk_level,
        'status': report.status,
        'image_analysis': {
            'filename': report.image_analysis.filename,
            'total_defects': report.image_analysis.total_defects,
            'analysis_summary': report.image_analysis.analysis_summary
        } if report.image_analysis else None,
        'capa_actions': [{
            'action_type': capa.action_type,
            'description': capa.description,
            'priority': capa.priority,
            'status': capa.status
        } for capa in report.capa_actions]
    }
    
    return jsonify(export_data)

# Health check
@api.route('/health', methods=['GET'])
def health_check():
    """API health check"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'version': '1.0.0'
    }) 