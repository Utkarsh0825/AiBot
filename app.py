import os
import uuid
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, session, send_from_directory, jsonify
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import PyPDF2
from docx import Document
import mammoth
import requests
from dotenv import load_dotenv  # To load variables from a .env file

# Load environment variables from .env file (for development)
load_dotenv()

# In app.py, after load_dotenv() and before initializing SQLAlchemy:
if "DATABASE_URL" in os.environ:
    app.config["SQLALCHEMY_DATABASE_URI"] = os.environ["DATABASE_URL"]
else:
    app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URI", "sqlite:///aibot.db")


app = Flask(__name__)

# Configure the app using environment variables
app.secret_key = os.environ.get("SECRET_KEY", "default_secret_key")  # Replace with secure key in production
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get("DATABASE_URI", "sqlite:///aibot.db")
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

db = SQLAlchemy(app)

# ------------------ MODELS ------------------
class User(db.Model):
    id         = db.Column(db.Integer, primary_key=True)
    first_name = db.Column(db.String(50), nullable=False)
    last_name  = db.Column(db.String(50), nullable=False)
    email      = db.Column(db.String(150), unique=True, nullable=False)
    password   = db.Column(db.String(200), nullable=False)
    resumes    = db.relationship('Resume', backref='user', lazy=True)

class Resume(db.Model):
    id                = db.Column(db.Integer, primary_key=True)
    original_filename = db.Column(db.String(150), nullable=False)
    file_path         = db.Column(db.String(300), nullable=False)
    timestamp         = db.Column(db.DateTime, default=datetime.utcnow)
    user_id           = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

class InterviewAnswer(db.Model):
    id        = db.Column(db.Integer, primary_key=True)
    user_id   = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    question  = db.Column(db.String(300), nullable=False)
    answer    = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

with app.app_context():
    db.create_all()

# ------------------ HUGGING FACE API SETUP ------------------
HF_API_TOKEN = os.environ.get("HF_API_TOKEN", "")
HF_MODEL_URL = os.environ.get("HF_MODEL_URL", "https://api-inference.huggingface.co/models/google/flan-t5-large")
HEADERS = {
    "Authorization": f"Bearer {HF_API_TOKEN}",
    "Content-Type": "application/json"
}

def query_hf_api(prompt, max_length=600):
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_length,
            "temperature": 0.7,
            "top_k": 50,
            "top_p": 0.95,
            "do_sample": True
        }
    }
    response = requests.post(HF_MODEL_URL, headers=HEADERS, json=payload)
    if response.status_code == 200:
        data = response.json()
        print("DEBUG: Raw model output:", data)
        return data[0]["generated_text"]
    else:
        print("Error calling HF API:", response.status_code, response.text)
        return ""

def generate_dynamic_questions(resume_text, difficulty):
    prompt = (
        f"Resume:\n{resume_text}\n\n"
        "You are a professional interviewer conducting a real, engaging interview. "
        "Based on the resume above, generate a flowing, natural conversation script that includes probing, human-like interview questions. "
        "Focus on detailed questions about projects, challenges, achievements, and lessons learned. "
        f"Tailor the tone for a {difficulty} interview (easy: friendly and general; medium: balanced; hard: technical and challenging). "
        "Generate up to 20 unique questions, each on a new line."
    )
    raw_output = query_hf_api(prompt, max_length=600)
    conversation = raw_output.replace(prompt, "").strip()
    lines = conversation.split("\n")
    questions = [line.strip() for line in lines if line.strip()]
    return questions[:20]

# ------------------ ROUTES ------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        first_name = request.form.get('first_name')
        last_name  = request.form.get('last_name')
        email      = request.form.get('email')
        password   = request.form.get('password')
        confirm    = request.form.get('confirm')

        if password != confirm:
            flash("Passwords do not match.", "danger")
            return redirect(url_for('signup'))

        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            flash("Email already used. Please log in.", "warning")
            return redirect(url_for('login'))

        hashed_password = generate_password_hash(password)
        new_user = User(
            first_name=first_name,
            last_name=last_name,
            email=email,
            password=hashed_password
        )
        db.session.add(new_user)
        db.session.commit()
        flash("Account created successfully! Please log in.", "success")
        return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email    = request.form.get('email')
        password = request.form.get('password')
        user = User.query.filter_by(email=email).first()
        if not user:
            flash("Email not found. Please sign up.", "warning")
            return redirect(url_for('signup'))
        if not check_password_hash(user.password, password):
            flash("Invalid password.", "danger")
            return redirect(url_for('login'))
        session['user_id'] = user.id
        flash(f"Logged in successfully! Welcome, {user.first_name} {user.last_name}!", "success")
        return redirect(url_for('dashboard'))
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    flash("Logged out successfully.", "info")
    return redirect(url_for('index'))

@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    if 'user_id' not in session:
        flash("Please log in to access the dashboard.", "warning")
        return redirect(url_for('login'))
    user = User.query.get(session['user_id'])
    if request.method == 'POST':
        file = request.files.get('resume')
        if not file or file.filename == '':
            flash("No file selected.", "danger")
            return redirect(url_for('dashboard'))
        filename = secure_filename(file.filename)
        unique_filename = str(uuid.uuid4()) + "_" + filename
        file_save_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        try:
            file.save(file_save_path)
            new_resume = Resume(
                original_filename=filename,
                file_path=unique_filename,
                user_id=session['user_id']
            )
            db.session.add(new_resume)
            db.session.commit()
            flash("Resume uploaded successfully!", "success")
        except Exception as e:
            flash(f"Error processing resume: {e}", "danger")
        return redirect(url_for('dashboard'))
    return render_template('dashboard.html', user=user)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/delete_resume/<int:resume_id>')
def delete_resume(resume_id):
    if 'user_id' not in session:
        flash("Please log in to delete resumes.", "warning")
        return redirect(url_for('login'))
    resume = Resume.query.get_or_404(resume_id)
    if resume.user_id != session['user_id']:
        flash("Unauthorized access.", "danger")
        return redirect(url_for('dashboard'))
    file_full_path = os.path.join(app.config['UPLOAD_FOLDER'], resume.file_path)
    if os.path.exists(file_full_path):
        os.remove(file_full_path)
    db.session.delete(resume)
    db.session.commit()
    flash("Resume deleted successfully.", "success")
    return redirect(url_for('dashboard'))

@app.route('/view_resume/<int:resume_id>')
def view_resume(resume_id):
    if 'user_id' not in session:
        flash("Please log in to view resumes.", "warning")
        return redirect(url_for('login'))
    resume = Resume.query.get_or_404(resume_id)
    if resume.user_id != session['user_id']:
        flash("Unauthorized access.", "danger")
        return redirect(url_for('dashboard'))
    ext = resume.original_filename.lower()
    file_full_path = os.path.join(app.config['UPLOAD_FOLDER'], resume.file_path)
    if ext.endswith('.pdf'):
        return render_template('view_pdf.html', resume=resume)
    elif ext.endswith('.docx'):
        try:
            with open(file_full_path, "rb") as docx_file:
                result = mammoth.convert_to_html(docx_file)
                html_content = result.value
            return render_template('view_docx.html', resume=resume, html_content=html_content)
        except Exception as e:
            flash(f"Error converting DOCX: {e}", "danger")
            return redirect(url_for('dashboard'))
    else:
        flash("Unsupported file format.", "danger")
        return redirect(url_for('dashboard'))

@app.route('/interview_setup', methods=['GET', 'POST'])
def interview_setup():
    if 'user_id' not in session:
        flash("Please log in to continue.", "warning")
        return redirect(url_for('login'))
    if request.method == 'POST':
        difficulty = request.form.get('difficulty')
        session['difficulty'] = difficulty
        flash(f"Difficulty set to {difficulty}.", "info")
        return redirect(url_for('interview_session'))
    return render_template('interview_setup.html')

@app.route('/interview_session')
def interview_session():
    if 'user_id' not in session:
        flash("Please log in to continue.", "warning")
        return redirect(url_for('login'))
    questions = [
        "Tell me about yourself.",
        "Could you describe a challenging project you worked on?",
        "What is your greatest strength?",
        "What do you consider your biggest achievement?",
        "Why do you want this role?",
        "Describe a time you failed and what you learned.",
        "What are your long-term career goals?"
    ]
    user = User.query.get(session['user_id'])
    return render_template('interview_session.html', questions=questions, user=user)

@app.route('/generate_questions/<int:resume_id>')
def generate_questions_for_resume(resume_id):
    if 'user_id' not in session:
        flash("Please log in to generate questions.", "warning")
        return redirect(url_for('login'))
    resume = Resume.query.get_or_404(resume_id)
    if resume.user_id != session['user_id']:
        flash("Unauthorized access to this resume.", "danger")
        return redirect(url_for('dashboard'))
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], resume.file_path)
    resume_text = ""
    if resume.original_filename.lower().endswith(".pdf"):
        with open(file_path, "rb") as f:
            pdf_reader = PyPDF2.PdfReader(f)
            for page in pdf_reader.pages:
                resume_text += page.extract_text() + "\n"
    elif resume.original_filename.lower().endswith(".docx"):
        doc = Document(file_path)
        for para in doc.paragraphs:
            resume_text += para.text + "\n"
    else:
        resume_text = "No parse logic for this file type"
    difficulty = session.get('difficulty', 'medium')
    questions = generate_dynamic_questions(resume_text, difficulty)
    if not questions:
        return jsonify({"error": "Dynamic question generation failed. Please try again."}), 500
    return jsonify({"questions": questions})

@app.route('/store_answer', methods=['POST'])
def store_answer():
    if 'user_id' not in session:
        return jsonify({"error": "Unauthorized"}), 401
    data = request.get_json()
    question = data.get('question', '')
    answer = data.get('answer', '')
    new_answer = InterviewAnswer(
        user_id=session['user_id'],
        question=question,
        answer=answer
    )
    db.session.add(new_answer)
    db.session.commit()
    return jsonify({"status": "success"})

@app.route('/stop_interview')
def stop_interview():
    flash("Interview stopped by user.", "info")
    return redirect(url_for('dashboard'))

if __name__ == '__main__':
    app.run(debug=True)
