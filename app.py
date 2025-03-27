from flask import Flask, render_template, request, jsonify, send_file
import os
from werkzeug.utils import secure_filename
import torch
from torchvision import transforms, models
from PIL import Image
import io
import base64
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.base import MIMEBase
from email import encoders

# Email settings (update with your details)
SMTP_SERVER = 'smtp.gmail.com'
SMTP_PORT = 587
SENDER_EMAIL = 'dodl21132.cs@rmkec.ac.in'  # Replace with your email
SENDER_PASSWORD = 'deeh hyqh zwxk hxwh'  # Replace with your email password
RECIPIENT_EMAIL = 'arak21101.cs@rmkec.ac.in'  # Replace with the recipient's email

def send_email(subject, body, recipient_email=RECIPIENT_EMAIL, image=None, report_path=None):
    try:
        # Create the email message object
        msg = MIMEMultipart()
        msg['From'] = SENDER_EMAIL
        msg['To'] = recipient_email
        msg['Subject'] = subject

        # Attach the body text
        msg.attach(MIMEText(body, 'plain'))

        # Attach the image if provided
        if image:
            img = MIMEImage(image, name="detected_image.jpg")
            msg.attach(img)

        # Attach the report file if provided
        if report_path:
            with open(report_path, 'rb') as report_file:
                report = MIMEBase('application', 'octet-stream')
                report.set_payload(report_file.read())
                encoders.encode_base64(report)
                report.add_header('Content-Disposition', 'attachment', filename='report.txt')
                msg.attach(report)

        # Send the email using the Gmail SMTP server
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()  # Encrypt the connection
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.sendmail(SENDER_EMAIL, recipient_email, msg.as_string())
            print("Email sent successfully!")

    except Exception as e:
        print(f"Error sending email: {str(e)}")


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load model once when starting the server
def load_model(model_path='models/ResNet18_best.pth'):
    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 7)
    
    try:
        model.load_state_dict(torch.load(model_path))
        print(f"Loaded trained model from {model_path}")
    except FileNotFoundError:
        print(f"Warning: Using pretrained model")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    return model

model = load_model()

# Disease details
disease_details = {
    'opacity': 'Opacity refers to the clouding of the lens of the eye, which can affect vision.',
    'diabetic retinopathy': 'Diabetic retinopathy is a diabetes complication that affects the eyes.',
    'glaucoma': 'Glaucoma is a group of eye conditions that damage the optic nerve.',
    'macular edema': 'Macular edema is the buildup of fluid in the macula, leading to vision loss.',
    'macular degeneration': 'Macular degeneration is an age-related condition that affects central vision.',
    'retinal vascular occlusion': 'Retinal vascular occlusion occurs when blood flow to the retina is blocked.',
    'normal': 'No diseases detected; the retina appears healthy.'
}

def predict_image(image):
    disease_labels = list(disease_details.keys())
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Transform image
    image_tensor = transform(image).unsqueeze(0)
    device = next(model.parameters()).device
    image_tensor = image_tensor.to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.sigmoid(outputs)
        predictions = (probabilities > 0.5).float()
    
    # Get detected diseases
    detected_diseases = []
    for i, (prob, pred) in enumerate(zip(probabilities[0], predictions[0])):
        if pred == 1:
            detected_diseases.append({
                'name': disease_labels[i],
                'probability': f"{prob.item():.2%}",
                'details': disease_details[disease_labels[i]]
            })
    
    return detected_diseases

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    try:
        # Read and process image
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Get predictions
        predictions = predict_image(image)
        
        # Convert image to base64 for display
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        # Save report
        report_path = os.path.join(app.config['UPLOAD_FOLDER'], 'report.txt')
        with open(report_path, 'w') as report_file:
            report_file.write("Detected Diseases:\n")
            if predictions:
                for pred in predictions:
                    report_file.write(f"- {pred['name']}: {pred['probability']}\n")
                    report_file.write(f"  Details: {pred['details']}\n")
            else:
                report_file.write("No diseases detected\n")
        
                # Send the email with the report and the image
        subject = "Retina Disease Detection Report"
        body = "Dear User, here is the detection report for your retina image. Please find the report and image attached."
        send_email(subject, body, image=buffered.getvalue(), report_path=report_path)
        
        return jsonify({
            'success': True,
            'predictions': predictions,
            'image': img_str,
            'report_url': '/download_report'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/download_report')
def download_report():
    report_path = os.path.join(app.config['UPLOAD_FOLDER'], 'report.txt')
    return send_file(report_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True) 