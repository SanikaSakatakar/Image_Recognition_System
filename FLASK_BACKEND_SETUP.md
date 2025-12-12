# Flask Backend Setup Guide for VisionAI

This guide provides complete instructions for setting up your Flask backend to work with the VisionAI image recognition app.

## Quick Start

The VisionAI frontend is currently running in **Demo Mode** with mock responses. Follow these steps to connect your real Flask backend:

### Step 1: Set Up Flask Backend

Create a new directory for your Flask backend:

\`\`\`bash
mkdir visionai-backend
cd visionai-backend
\`\`\`

### Step 2: Install Dependencies

\`\`\`bash
pip install flask flask-cors pillow numpy tensorflow
# OR for PyTorch
pip install flask flask-cors pillow numpy torch torchvision
\`\`\`

### Step 3: Create Flask App

Create `app.py` with the following code:

\`\`\`python
from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
from io import BytesIO
from PIL import Image
import numpy as np

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load your ML model here
# For TensorFlow:
# import tensorflow as tf
# model = tf.keras.models.load_model('path/to/your/model')

# For PyTorch:
# import torch
# model = torch.load('path/to/your/model')
# model.eval()

@app.route('/analyze', methods=['POST'])
def analyze_image():
    try:
        # Get the base64 image from request
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400
        
        # Decode base64 image
        image_data = data['image']
        if ',' in image_data:
            image_data = image_data.split(',')[1]  # Remove data:image/xxx;base64, prefix
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes))
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Preprocess image for your model
        # Example for ResNet/VGG models (224x224)
        image = image.resize((224, 224))
        img_array = np.array(image) / 255.0  # Normalize
        
        # For TensorFlow models:
        # img_array = np.expand_dims(img_array, axis=0)
        # predictions = model.predict(img_array)
        
        # For PyTorch models:
        # from torchvision import transforms
        # transform = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # ])
        # img_tensor = transform(image).unsqueeze(0)
        # with torch.no_grad():
        #     predictions = model(img_tensor)
        
        # MOCK RESPONSE FOR TESTING (Replace with real model predictions)
        import random
        labels = ['Cat', 'Dog', 'Car', 'Building', 'Tree', 'Person', 'Bicycle', 'Flower']
        main_label = random.choice(labels)
        confidence = random.uniform(75, 98)
        
        # Generate top predictions
        top_predictions = [
            {'label': main_label, 'confidence': confidence},
            {'label': random.choice([l for l in labels if l != main_label]), 'confidence': random.uniform(50, 70)},
            {'label': random.choice([l for l in labels if l != main_label]), 'confidence': random.uniform(30, 50)},
        ]
        
        return jsonify({
            'success': True,
            'label': main_label,
            'confidence': confidence,
            'predictions': top_predictions,
            'details': top_predictions
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'message': 'VisionAI Flask backend is running'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
\`\`\`

### Step 4: Run Flask Backend

\`\`\`bash
python app.py
\`\`\`

Your Flask backend should now be running at `http://localhost:5000`

### Step 5: Connect Frontend to Backend

#### Option A: Local Development (Same Machine)

1. In v0, click the **Vars** section in the left sidebar
2. Add environment variable:
   - Key: `FLASK_API_URL`
   - Value: `http://localhost:5000/analyze`
3. Save and refresh your app

#### Option B: Deploy Flask Backend (Recommended)

Deploy your Flask backend to make it accessible from anywhere:

**Using Railway:**
\`\`\`bash
# Install Railway CLI
npm install -g @railway/cli

# Login and deploy
railway login
railway init
railway up
\`\`\`

**Using Render:**
1. Push your Flask code to GitHub
2. Go to https://render.com
3. Create new Web Service
4. Connect your GitHub repo
5. Set start command: `gunicorn app:app`
6. Deploy

**Using Vercel (Serverless):**
\`\`\`bash
npm install -g vercel
vercel
\`\`\`

After deployment, copy the production URL and add it to your environment variables:
- Key: `FLASK_API_URL`
- Value: `https://your-backend.railway.app/analyze`

## Using Pre-trained Models

### TensorFlow/Keras Example

\`\`\`python
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

# Load pre-trained ResNet50
model = ResNet50(weights='imagenet')

@app.route('/analyze', methods=['POST'])
def analyze_image():
    # ... image loading code ...
    
    # Preprocess for ResNet50
    image = image.resize((224, 224))
    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    # Predict
    predictions = model.predict(img_array)
    decoded = decode_predictions(predictions, top=5)[0]
    
    # Format response
    top_predictions = [
        {'label': label, 'confidence': float(conf * 100)}
        for (_, label, conf) in decoded
    ]
    
    return jsonify({
        'success': True,
        'label': top_predictions[0]['label'],
        'confidence': top_predictions[0]['confidence'],
        'predictions': top_predictions
    })
\`\`\`

### PyTorch Example

\`\`\`python
import torch
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights

# Load pre-trained ResNet50
model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
model.eval()

# Get class labels
weights = ResNet50_Weights.IMAGENET1K_V2
categories = weights.meta["categories"]

# Image preprocessing
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@app.route('/analyze', methods=['POST'])
def analyze_image():
    # ... image loading code ...
    
    # Preprocess
    img_tensor = preprocess(image).unsqueeze(0)
    
    # Predict
    with torch.no_grad():
        output = model(img_tensor)
    
    # Get probabilities
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    
    # Get top 5 predictions
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    
    top_predictions = [
        {'label': categories[top5_catid[i]], 'confidence': float(top5_prob[i] * 100)}
        for i in range(5)
    ]
    
    return jsonify({
        'success': True,
        'label': top_predictions[0]['label'],
        'confidence': top_predictions[0]['confidence'],
        'predictions': top_predictions
    })
\`\`\`

## API Response Format

Your Flask backend should return JSON in this format:

### Success Response:
\`\`\`json
{
  "success": true,
  "label": "Golden Retriever",
  "confidence": 94.5,
  "predictions": [
    {"label": "Golden Retriever", "confidence": 94.5},
    {"label": "Labrador", "confidence": 78.2},
    {"label": "Dog", "confidence": 65.8}
  ]
}
\`\`\`

### Error Response:
\`\`\`json
{
  "error": "Error message here"
}
\`\`\`

## Troubleshooting

### CORS Issues
Make sure Flask-CORS is installed and enabled:
\`\`\`python
from flask_cors import CORS
CORS(app)
\`\`\`

### Connection Refused
- Check Flask is running: `curl http://localhost:5000/health`
- Verify `FLASK_API_URL` environment variable is set correctly
- Make sure Flask is listening on `0.0.0.0` not `127.0.0.1`

### Slow Predictions
- Use GPU if available (CUDA for PyTorch, TensorFlow GPU)
- Implement model caching to avoid reloading on each request
- Consider using quantized models for faster inference

### Memory Issues
- Limit request size in Flask:
\`\`\`python
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
\`\`\`

## Production Deployment Checklist

- [ ] Remove `debug=True` from Flask
- [ ] Use production WSGI server (gunicorn, waitress)
- [ ] Add request validation and rate limiting
- [ ] Implement proper error handling
- [ ] Add logging
- [ ] Use environment variables for configuration
- [ ] Enable HTTPS
- [ ] Add authentication if needed

## Next Steps

1. Replace mock predictions with your actual ML model
2. Deploy Flask backend to production
3. Update `FLASK_API_URL` environment variable
4. Test with real images
5. Monitor performance and errors

For more help, check the console logs in the v0 preview for detailed error messages.
