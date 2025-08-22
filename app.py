from flask import Flask, render_template, request, session, redirect, url_for, flash, jsonify
import requests
import json
from functools import wraps
from datetime import datetime
import os
from werkzeug.utils import secure_filename
from image_processor import defect_detector
from blueprints.insight_engine import insight_engine_bp
from blueprints.pharma_gpt import pharma_gpt_bp
from blueprints.risk_console import risk_console_bp
from models import init_db
from blueprints.fda import fda_bp
from dotenv import load_dotenv
from blueprints.investigation import investigation_bp

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this to a secure secret key in production
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///pharma_app.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Configuration for file uploads
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Store reports in memory (in production, use a database)
reports = []

# Register advanced feature blueprints
app.register_blueprint(insight_engine_bp)
app.register_blueprint(pharma_gpt_bp)
app.register_blueprint(risk_console_bp)
app.register_blueprint(fda_bp)
app.register_blueprint(investigation_bp)

load_dotenv()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route("/")
def index():
    # Clear any existing session
    session.clear()
    return redirect(url_for('login'))

@app.route("/home")
@login_required
def home():
    return render_template("home.html")

@app.route("/login", methods=['GET', 'POST'])
def login():
    # If user is already logged in, redirect to home
    if 'user' in session:
        return redirect(url_for('home'))
        
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        # This is a simple example - in production, use proper authentication
        if username == "admin" and password == "admin":
            session['user'] = username
            return redirect(url_for('home'))
        return render_template("login.html", error="Invalid credentials")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()  # Clear all session data
    return redirect(url_for('login'))

@app.route("/dashboard")
@login_required
def dashboard():
    return render_template("dashboard.html")

@app.route("/history")
@login_required
def history():
    return render_template("history.html", reports=reports)

@app.route("/upload_image", methods=['GET', 'POST'])
@login_required
def upload_image():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'image' not in request.files:
            flash('No file selected')
            return render_template("upload_image.html")
        
        file = request.files['image']
        
        # If user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No file selected')
            return render_template("upload_image.html")
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            # Add timestamp to avoid filename conflicts
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Analyze the image for defects
            defects_analysis = defect_detector.detect_defects(filepath)
            defect_summary = defect_detector.generate_defect_summary(defects_analysis)
            
            # Store analysis results in session for report generation
            session['image_analysis'] = {
                'filepath': filepath,
                'filename': filename,
                'defects_analysis': defects_analysis,
                'defect_summary': defect_summary
            }
            
            flash('Image uploaded and analyzed successfully!')
            return render_template("image_analysis.html", 
                                analysis=defects_analysis, 
                                summary=defect_summary,
                                filename=filename)
        else:
            flash('Invalid file type. Please upload an image file.')
            return render_template("upload_image.html")
    
    return render_template("upload_image.html")

@app.route("/generate", methods=["POST"])
@login_required
def generate():
    try:
        data = request.form
        agency = data['agency']
        # Get image analysis if available
        image_analysis = session.get('image_analysis', {})
        image_info = ""
        defect_info = ""
        critical_error_info = ""
        if image_analysis:
            # Check for critical mixed tablet error
            if image_analysis.get('critical_error', False):
                critical_error_info = f"""
                🚨 CRITICAL QUALITY ISSUE DETECTED:
                {image_analysis.get('defect_summary', 'Multiple different tablets detected')}
                
                This is a serious pharmaceutical quality control failure requiring immediate action.
                """
            else:
                image_info = f"""
                Image Analysis Results:
                - Image File: {image_analysis.get('filename', 'N/A')}
                - Analysis Timestamp: {image_analysis.get('defects_analysis', {}).get('analysis_timestamp', 'N/A')}
                - Total Defects Detected: {image_analysis.get('defects_analysis', {}).get('total_defects', 0)}
                """
                defect_info = image_analysis.get('defect_summary', 'No image analysis available.')
        # Additional form data
        batch_number = data.get('batch_number', 'N/A')
        tablet_name = data.get('tablet_name', 'N/A')
        recall_needed = data.get('recall_needed', 'No')
        # CAPA information
        immediate_actions = data.get('immediate_actions', 'None reported')
        quality_impact = data.get('quality_impact', 'N/A')
        training_needed = data.get('training_needed', 'N/A')
        previous_cases = data.get('cases', 'N/A')
        additional_notes = data.get('notes', 'None')
        # --- FDA Integration ---
        FDA_API_KEY = os.getenv('OPENFDA_API_KEY')
        fda_base = 'https://api.fda.gov'
        def fda_get(endpoint, params):
            params = params.copy()
            if FDA_API_KEY:
                params['api_key'] = FDA_API_KEY
            params['limit'] = params.get('limit', 5)
            url = f"{fda_base}{endpoint}.json"
            r = requests.get(url, params=params)
            if r.status_code == 200:
                return r.json()
            return {'error': r.text, 'status': r.status_code}
        # Query FDA endpoints
        fda_recalls = fda_get('/drug/enforcement', {'search': f"product_description:{tablet_name}" + (f"+AND+recall_number:{batch_number}" if batch_number and batch_number != 'N/A' else '')})
        fda_events = fda_get('/drug/event', {'search': f"patient.drug.medicinalproduct:{tablet_name}"})
        fda_label = fda_get('/drug/label', {'search': f"openfda.brand_name:{tablet_name}"})
        # Summarize FDA data
        recall_alert = False
        recall_summary = "No recalls found."
        if 'results' in fda_recalls and len(fda_recalls['results']) > 0:
            recall_alert = True
            recall_summary = f"FDA Recall(s) found for {tablet_name} (Batch: {batch_number}):\n"
            for recall in fda_recalls['results']:
                recall_summary += f"- Recall #: {recall.get('recall_number', 'N/A')}, Reason: {recall.get('reason_for_recall', 'N/A')}, Status: {recall.get('status', 'N/A')}\n"
        events_summary = "No significant adverse events found."
        if 'results' in fda_events and len(fda_events['results']) > 0:
            events_summary = f"Adverse events reported for {tablet_name}: {len(fda_events['results'])} (see FDA for details)."
        label_summary = "No label info found."
        if 'results' in fda_label and len(fda_label['results']) > 0:
            label_summary = f"Label info available for {tablet_name}."
        # --- End FDA Integration ---
        # --- Investigation Section ---
        similar_investigations = []
        for r in reports:
            if r['tablet_name'].lower() == tablet_name.lower() and r['type'] == data.get('type', 'N/A'):
                similar_investigations.append({
                    'batch_number': r['batch_number'],
                    'immediate_actions': r.get('immediate_actions', ''),
                    'quality_impact': r.get('quality_impact', ''),
                    'notes': r.get('additional_notes', ''),
                    'timestamp': r['timestamp'],
                })
        investigation_summary = "No previous similar investigations found."
        if similar_investigations:
            investigation_summary = "Previous similar investigations:\n"
            for inv in similar_investigations:
                investigation_summary += f"- Batch: {inv['batch_number']} (Date: {inv['timestamp']}): Actions: {inv['immediate_actions']}, Quality Impact: {inv['quality_impact']}, Notes: {inv['notes']}\n"
        # --- End Investigation Section ---
        prompt = f"""
        You are a regulatory analyst at {agency}.
        {critical_error_info}
        Based on the following reported information:
        - Tablet Name: {tablet_name}
        - Batch Number: {batch_number}
        - Contaminant Type: {data.get('type', 'N/A')}
        - Composition: {data.get('composition', 'N/A')}
        - Size (µm): {data.get('size', 'N/A')}
        - Recall Needed: {recall_needed}
        - Previous Similar Cases: {previous_cases}
        - Immediate Actions Taken: {immediate_actions}
        - Quality System Impact: {quality_impact}
        - Training Requirements: {training_needed}
        - Additional Notes: {additional_notes}
        {image_info}
        Image Analysis Results:
        {defect_info}
        FDA Data:
        Recall Summary: {recall_summary}
        Adverse Events: {events_summary}
        Label Info: {label_summary}
        Investigation:
        {investigation_summary}
        {critical_error_info}
        
        {'🚨 CRITICAL ALERT: This report involves a critical quality issue with multiple different tablets detected. This requires immediate regulatory attention and batch quarantine.' if critical_error_info else ''}
        
        Generate a comprehensive contamination risk assessment report including:
        {'1. CRITICAL QUALITY ISSUE: This is a critical pharmaceutical quality control failure requiring immediate regulatory notification and batch quarantine. Emphasize the severity and immediate actions required.' if critical_error_info else '1. Classification of risk level (Low/Medium/High) with justification based on both manual input and image analysis.'}
        2. Predicted source of contamination (e.g., packaging, manufacturing, storage, etc.).
        3. Likely toxicity level and patient impact.
        4. Whether similar cases have been reported historically.
        5. Recommendation: Should this be reported to {agency} or can it be handled internally? How important is it to report to the agency, and how much time should be reported?
        6. CAPA (Corrective Action and Preventive Action) Analysis:
           - Immediate corrective actions required (considering actions already taken: {immediate_actions})
           - Root cause analysis recommendations
           - Preventive measures to avoid recurrence
           - Quality system improvements needed (based on impact level: {quality_impact})
           - Training requirements for staff (based on: {training_needed})
           - Documentation and record-keeping requirements
           - Timeline for CAPA implementation
           - Effectiveness monitoring plan
        7. Investigation: Provide a detailed investigation section, including:
           - The process of investigation for this case
           - How investigation was performed in previous similar cases (see above)
           - Best practices for pharmaceutical market complaint investigation
           - Next steps for investigation and documentation
        8. Provide the report in formal regulatory language as expected by {agency}.
        9. If image analysis detected defects, include specific recommendations based on the visual analysis.
        10. If FDA data indicates a recall or adverse event, include this in the risk assessment and recommendations.
        11. {'If this is a critical mixed tablet issue, emphasize the immediate regulatory requirements, patient safety implications, and mandatory reporting timeline.' if critical_error_info else ''}
        11. Please do not use markdown, bold (**), or any special formatting in the output.
        """
        response = requests.post("http://localhost:11434/api/generate",
            json={"model": "llama3", "prompt": prompt, "stream": False})
        if response.status_code != 200:
            return f"Error from Ollama API: {response.status_code} - {response.text}", 500
        try:
            response_data = response.json()
            print("Ollama Response:", json.dumps(response_data, indent=2))  # Debug print
            if "response" not in response_data:
                return f"Unexpected response format from Ollama: {response_data}", 500
            report = response_data["response"]
            if not report or report.strip() == "":
                return "Empty response received from Ollama", 500
            # Store the report
            report_data = {
                'id': len(reports) + 1,  # Simple ID generation
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'agency': agency,
                'tablet_name': tablet_name,
                'batch_number': batch_number,
                'type': data.get('type', 'N/A'),
                'composition': data.get('composition', 'N/A'),
                'size': data.get('size', 'N/A'),
                'recall_needed': recall_needed,
                'previous_cases': previous_cases,
                'immediate_actions': immediate_actions,
                'quality_impact': quality_impact,
                'training_needed': training_needed,
                'additional_notes': additional_notes,
                'content': report,
                'image_filename': image_analysis.get('filename', 'N/A') if image_analysis else 'N/A',
                'defects_detected': image_analysis.get('defects_analysis', {}).get('total_defects', 0) if image_analysis else 0,
                'revisions': [],  # Track revisions
                'fda_recalls': fda_recalls,
                'fda_events': fda_events,
                'fda_label': fda_label,
                'recall_alert': recall_alert,
                'recall_summary': recall_summary
            }
            reports.append(report_data)
            # Clear image analysis from session after report generation
            session.pop('image_analysis', None)
            return render_template("result.html", report=report, report_id=report_data['id'], recall_alert=recall_alert, recall_summary=recall_summary, fda_recalls=fda_recalls, fda_events=fda_events, fda_label=fda_label)
        except json.JSONDecodeError as e:
            print("Raw response:", response.text)  # Debug print
            return f"Error parsing Ollama response: {str(e)}", 500
    except requests.exceptions.ConnectionError:
        return "Could not connect to Ollama server. Please make sure Ollama is running.", 500
    except Exception as e:
        return f"An error occurred: {str(e)}", 500

@app.route("/revise_report", methods=["POST"])
@login_required
def revise_report():
    try:
        original_report = request.form['original_report']
        user_feedback = request.form['user_feedback']
        report_id = int(request.form['report_id'])
        
        # Find the report in our storage
        report_data = None
        for report in reports:
            if report['id'] == report_id:
                report_data = report
                break
        
        if not report_data:
            return "Report not found", 404
        
        # Create revision prompt with user feedback
        revision_prompt = f"""
        You are a regulatory analyst. Please revise the following report based on the user's feedback.

        ORIGINAL REPORT:
        {original_report}

        USER FEEDBACK:
        {user_feedback}

        Please provide a revised version of the report that addresses the user's feedback while maintaining the formal regulatory language and structure. The revised report should:
        1. Address all points mentioned in the user feedback
        2. Maintain the same professional tone and regulatory format
        3. Keep all essential information from the original report
        4. Make improvements based on the feedback provided
        5. Ensure CAPA (Corrective Action and Preventive Action) considerations are properly addressed
        6. Include appropriate regulatory compliance language

        Please provide only the revised report content without any additional explanations or markdown formatting.
        """

        response = requests.post("http://localhost:11434/api/generate",
            json={"model": "llama3", "prompt": revision_prompt, "stream": False})
        
        if response.status_code != 200:
            return f"Error from Ollama API: {response.status_code} - {response.text}", 500
            
        try:
            response_data = response.json()
            
            if "response" not in response_data:
                return f"Unexpected response format from Ollama: {response_data}", 500
                
            revised_report = response_data["response"]
            if not revised_report or revised_report.strip() == "":
                return "Empty response received from Ollama", 500
            
            # Store the revision
            revision_data = {
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'user_feedback': user_feedback,
                'original_content': report_data['content'],
                'revised_content': revised_report
            }
            
            # Update the report with the revision
            report_data['revisions'].append(revision_data)
            report_data['content'] = revised_report  # Update to latest version
            
            return render_template("result.html", report=revised_report, report_id=report_id, revision_success=True)
            
        except json.JSONDecodeError as e:
            print("Raw response:", response.text)  # Debug print
            return f"Error parsing Ollama response: {str(e)}", 500
            
    except requests.exceptions.ConnectionError:
        return "Could not connect to Ollama server. Please make sure Ollama is running.", 500
    except Exception as e:
        return f"An error occurred: {str(e)}", 500

@app.route("/insight_engine")
@login_required
def insight_engine_dashboard():
    return render_template("advanced_templates/insight_engine.html")

@app.route("/pharma_gpt")
@login_required
def pharma_gpt_dashboard():
    return render_template("advanced_templates/pharma_gpt.html")

@app.route("/risk_console")
@login_required
def risk_console_dashboard():
    return render_template("advanced_templates/risk_console.html")

@app.route("/fda_dashboard")
@login_required
def fda_dashboard():
    return render_template("advanced_templates/fda_dashboard.html")

@app.route("/investigation_tasks")
@login_required
def investigation_tasks():
    return render_template("advanced_templates/investigation_tasks.html")

if __name__ == "__main__":
    try:
        init_db(app)
        print("✅ Database initialized successfully!")
        print("🚀 Starting Flask application...")
        app.run(debug=True, port=8081)
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        print("Please check the error details above and fix any issues.")
