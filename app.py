from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
import cv2
import pytesseract
import numpy as np
from PIL import Image
import re

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    """Preprocess image for better OCR"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply thresholding
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def extract_text(image_path):
    """Extract text from image using OCR"""
    try:
        processed_img = preprocess_image(image_path)
        text = pytesseract.image_to_string(processed_img)
        return text
    except Exception as e:
        return f"Error extracting text: {str(e)}"

def parse_math_problem(text):
    """Parse mathematical expressions from text"""
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Extract equations (basic pattern matching)
    equations = re.findall(r'[0-9x\+\-\*\/\=\(\)\^\s]+', text)
    
    return {
        'full_text': text,
        'equations': equations,
        'problem_type': detect_problem_type(text)
    }

def detect_problem_type(text):
    """Detect type of JEE problem"""
    text_lower = text.lower()
    if any(word in text_lower for word in ['integrate', 'integration', 'differentiate', 'derivative']):
        return 'Calculus'
    elif any(word in text_lower for word in ['solve', 'equation', 'roots']):
        return 'Algebra'
    elif any(word in text_lower for word in ['force', 'velocity', 'acceleration', 'motion']):
        return 'Physics - Mechanics'
    elif any(word in text_lower for word in ['reaction', 'chemical', 'organic', 'molecule']):
        return 'Chemistry'
    else:
        return 'General Mathematics'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Extract text from image
        extracted_text = extract_text(filepath)
        
        # Parse the problem
        parsed_data = parse_math_problem(extracted_text)
        
        # Clean up uploaded file
        os.remove(filepath)
        
        return jsonify({
            'success': True,
            'extracted_text': parsed_data['full_text'],
            'problem_type': parsed_data['problem_type'],
            'equations': parsed_data['equations']
        })
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/solve', methods=['POST'])
def solve_problem():
    """Placeholder for solving logic - can integrate with SymPy or other solvers"""
    data = request.json
    equation = data.get('equation', '')
    
    # This is a placeholder - you can integrate actual solving logic here
    return jsonify({
        'solution': 'Solution functionality coming soon! Equation detected: ' + equation,
        'steps': ['Step 1: Parse equation', 'Step 2: Solve', 'Step 3: Display result']
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
