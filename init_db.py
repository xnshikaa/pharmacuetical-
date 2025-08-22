from app import app, db
from models import User, Report, ImageAnalysis, CAPAAction, AuditLog, Batch
from datetime import datetime

def init_database():
    """Initialize the database with tables and initial data"""
    with app.app_context():
        # Create all tables
        db.create_all()
        
        # Create admin user if it doesn't exist
        admin_user = User.query.filter_by(username='admin').first()
        if not admin_user:
            admin_user = User(
                username='admin',
                email='admin@pharma.com',
                role='admin'
            )
            admin_user.set_password('admin')
            db.session.add(admin_user)
            db.session.commit()
            print("Admin user created: username=admin, password=admin")
        
        # Create sample analyst user
        analyst_user = User.query.filter_by(username='analyst').first()
        if not analyst_user:
            analyst_user = User(
                username='analyst',
                email='analyst@pharma.com',
                role='analyst'
            )
            analyst_user.set_password('analyst')
            db.session.add(analyst_user)
            db.session.commit()
            print("Analyst user created: username=analyst, password=analyst")
        
        # Create sample viewer user
        viewer_user = User.query.filter_by(username='viewer').first()
        if not viewer_user:
            viewer_user = User(
                username='viewer',
                email='viewer@pharma.com',
                role='viewer'
            )
            viewer_user.set_password('viewer')
            db.session.add(viewer_user)
            db.session.commit()
            print("Viewer user created: username=viewer, password=viewer")
        
        print("Database initialized successfully!")
        print("Available users:")
        print("- admin/admin (Administrator)")
        print("- analyst/analyst (Analyst)")
        print("- viewer/viewer (Viewer)")

if __name__ == '__main__':
    init_database() 