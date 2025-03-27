import torch
from torchvision import transforms, models
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

def load_model(model_path='models/ResNet18_best.pth'):
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Initialize model
    model = models.resnet50(pretrained=True)  # Changed to True to use pretrained weights
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 7)  # 7 disease classes
    
    # Try to load trained weights if they exist
    try:
        model.load_state_dict(torch.load(model_path))
        print(f"Loaded trained model from {model_path}")
    except FileNotFoundError:
        print(f"Warning: No trained model found at {model_path}")
        print("Using pretrained ResNet50 model. Please train the model first for better results.")
    
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    return model

def predict_image(image_path, model):
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return []
        
    # Disease labels
    disease_labels = ['opacity', 'diabetic retinopathy', 'glaucoma', 'macular edema',
                     'macular degeneration', 'retinal vascular occlusion', 'normal']
    
    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    try:
        # Load and transform image
        image = Image.open(image_path)
        image_tensor = transform(image).unsqueeze(0)
        
        # Get device
        device = next(model.parameters()).device
        image_tensor = image_tensor.to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.sigmoid(outputs)
            predictions = (probabilities > 0.5).float()
        
        # Show image with predictions
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title('Input Image')
        plt.axis('off')
        
        # Show predictions
        plt.subplot(1, 2, 2)
        detected_diseases = []
        for i, (prob, pred) in enumerate(zip(probabilities[0], predictions[0])):
            if pred == 1:
                detected_diseases.append(f"{disease_labels[i]}: {prob.item():.2%}")
        
        plt.text(0.1, 0.5, '\n'.join(detected_diseases) if detected_diseases else 'No diseases detected',
                fontsize=12, verticalalignment='center')
        plt.axis('off')
        plt.show()
        
        return detected_diseases
        
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return []

def main():
    # Load model
    model = load_model()
    
    while True:
        # Get image path from user
        image_path = input("\nEnter the path to your retinal image (or 'q' to quit): ")
        
        if image_path.lower() == 'q':
            break
            
        # Make prediction
        detected_diseases = predict_image(image_path, model)
        
        # Print results
        print("\nDetected Diseases:")
        if detected_diseases:
            for disease in detected_diseases:
                print(f"- {disease}")
        else:
            print("No diseases detected")

if __name__ == "__main__":
    main() 