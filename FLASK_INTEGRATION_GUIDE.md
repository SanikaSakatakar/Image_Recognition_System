# Flask Backend Integration Guide

Complete guide for integrating a custom Flask backend with your VisionAI app for advanced image recognition capabilities.

## Why Use a Flask Backend?

While the TensorFlow.js client-side model works great, you might want a Flask backend for:

- **Custom Models**: Use your own trained models (ResNet, EfficientNet, custom CNNs)
- **Specialized Tasks**: Medical imaging, face recognition, object detection
- **Larger Models**: More powerful models that don't fit in browsers
- **Pre/Post Processing**: Custom image preprocessing or result formatting
- **Database Integration**: Store results, user data, or training logs
- **GPU Acceleration**: Use powerful server GPUs for faster processing

## Step-by-Step Setup

### Step 1: Create Flask Application

Create a new directory for your backend:

\`\`\`bash
mkdir flask-backend
cd flask-backend
\`\`\`

### Step 2: Install Dependencies

Create `requirements.txt`:

\`\`\`txt
flask==3.0.0
flask-cors==4.0.0
pillow==10.1.0
tensorflow==2.15.0
numpy==1.24.3
\`\`\`

Or for PyTorch:

\`\`\`txt
flask==3.0.0
flask-cors==4.0.0
pillow==10.1.0
torch==2.1.0
torchvision==0.16.0
numpy==1.24.3
\`\`\`

Install:

\`\`\`bash
pip install -r requirements.txt
\`\`\`

### Step 3: Create Flask Server

#### Option A: Using TensorFlow/Keras

\`\`\`python
# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io
import base64
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

app = Flask(__name__)
CORS(app)  # Enable CORS for Next.js frontend

# Load pre-trained model
print("Loading ResNet50 model...")
model = ResNet50(weights='imagenet')
print("Model loaded successfully!")

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'model': 'ResNet50'})

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # Get image from request
        data = request.json
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({'error': 'No image provided'}), 400
        
        # Decode base64 image
        image_bytes = base64.b64decode(image_data.split(',')[1])
        image = Image.open(io.BytesIO(image_bytes))
        
        # Preprocess image
        image = image.convert('RGB')
        image = image.resize((224, 224))
        image_array = np.array(image)
        image_array = np.expand_dims(image_array, axis=0)
        image_array = preprocess_input(image_array)
        
        # Make prediction
        predictions = model.predict(image_array)
        decoded = decode_predictions(predictions, top=5)[0]
        
        # Format response
        results = {
            'label': decoded[0][1].replace('_', ' ').title(),
            'confidence': float(decoded[0][2] * 100),
            'predictions': [
                {
                    'label': pred[1].replace('_', ' ').title(),
                    'confidence': float(pred[2] * 100)
                }
                for pred in decoded
            ],
            'mode': 'real',
            'message': 'Classified using ResNet50'
        }
        
        return jsonify(results)
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
\`\`\`

#### Option B: Using PyTorch

\`\`\`python
# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io
import base64
import torch
from torchvision import models, transforms

app = Flask(__name__)
CORS(app)

# Load pre-trained model
print("Loading ResNet50 model...")
model = models.resnet50(pretrained=True)
model.eval()
print("Model loaded successfully!")

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load ImageNet class labels
with open('imagenet_classes.txt', 'r') as f:
    categories = [s.strip() for s in f.readlines()]

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'model': 'ResNet50-PyTorch'})

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.json
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({'error': 'No image provided'}), 400
        
        # Decode and preprocess image
        image_bytes = base64.b64decode(image_data.split(',')[1])
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image_tensor = transform(image).unsqueeze(0)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        
        # Get top 5 predictions
        top5_prob, top5_catid = torch.topk(probabilities, 5)
        
        results = {
            'label': categories[top5_catid[0]],
            'confidence': float(top5_prob[0] * 100),
            'predictions': [
                {
                    'label': categories[catid],
                    'confidence': float(prob * 100)
                }
                for prob, catid in zip(top5_prob, top5_catid)
            ],
            'mode': 'real',
            'message': 'Classified using ResNet50-PyTorch'
        }
        
        return jsonify(results)
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
\`\`\`

### Step 4: Test Locally

Run the Flask server:

\`\`\`bash
python app.py
\`\`\`

Test the health endpoint:

\`\`\`bash
curl http://localhost:5000/health
\`\`\`

Expected response:
\`\`\`json
{"status": "healthy", "model": "ResNet50"}
\`\`\`

### Step 5: Deploy to Production

#### Option 1: Railway (Recommended)

1. Create account at [railway.app](https://railway.app)
2. Install Railway CLI:
   \`\`\`bash
   npm install -g @railway/cli
   \`\`\`
3. Login and deploy:
   \`\`\`bash
   railway login
   railway init
   railway up
   \`\`\`
4. Get your public URL from Railway dashboard

#### Option 2: Render

1. Create account at [render.com](https://render.com)
2. Create new Web Service
3. Connect your GitHub repository
4. Set build command: `pip install -r requirements.txt`
5. Set start command: `gunicorn app:app`
6. Deploy and get public URL

#### Option 3: Fly.io

1. Install Fly CLI:
   \`\`\`bash
   curl -L https://fly.io/install.sh | sh
   \`\`\`
2. Login and launch:
   \`\`\`bash
   fly auth login
   fly launch
   fly deploy
   \`\`\`
3. Get public URL from Fly dashboard

### Step 6: Configure Next.js App

In your Vercel project or v0 Vars section, add:

\`\`\`bash
FLASK_API_URL=https://your-flask-api.railway.app
\`\`\`

**Important**: Use HTTPS, not HTTP!

### Step 7: Verify Integration

1. Upload an image in your VisionAI app
2. Check browser console for logs
3. Verify "Real AI" badge appears (not "Demo")
4. Confirm predictions match Flask model

## Advanced Configurations

### Using Custom Trained Models

Replace the model loading with your custom model:

\`\`\`python
# Load your custom model
model = tf.keras.models.load_model('path/to/your/model.h5')

# Or for PyTorch
model = YourCustomModel()
model.load_state_dict(torch.load('path/to/model.pth'))
\`\`\`

### Adding Authentication

Protect your API with API keys:

\`\`\`python
from functools import wraps
from flask import request, jsonify

def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        if api_key != 'your-secret-api-key':
            return jsonify({'error': 'Invalid API key'}), 401
        return f(*args, **kwargs)
    return decorated_function

@app.route('/analyze', methods=['POST'])
@require_api_key
def analyze():
    # ... existing code
\`\`\`

Then update Next.js API route to include the key:

\`\`\`typescript
const response = await fetch(`${flaskUrl}/analyze`, {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
    "X-API-Key": process.env.FLASK_API_KEY || "",
  },
  body: JSON.stringify({ image }),
})
\`\`\`

### Database Integration

Store analysis results in PostgreSQL:

\`\`\`python
import psycopg2
from datetime import datetime

# Database connection
conn = psycopg2.connect(
    database="visionai",
    user="your_user",
    password="your_password",
    host="your_host"
)

@app.route('/analyze', methods=['POST'])
def analyze():
    # ... existing prediction code
    
    # Store results
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO analyses (timestamp, label, confidence) VALUES (%s, %s, %s)",
        (datetime.now(), results['label'], results['confidence'])
    )
    conn.commit()
    
    return jsonify(results)
\`\`\`

## Troubleshooting

### CORS Errors

Add Flask-CORS with explicit origins:

\`\`\`python
CORS(app, resources={
    r"/*": {
        "origins": ["https://your-nextjs-app.vercel.app"],
        "methods": ["POST", "GET"],
        "allow_headers": ["Content-Type"]
    }
})
\`\`\`

### Memory Issues

Limit image size or use batch processing:

\`\`\`python
# Limit max image size
MAX_IMAGE_SIZE = (1024, 1024)
image.thumbnail(MAX_IMAGE_SIZE, Image.Resampling.LANCZOS)
\`\`\`

### Slow Predictions

Use GPU acceleration:

\`\`\`python
# TensorFlow
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# PyTorch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
\`\`\`

### Timeout Errors

Increase timeout in Next.js API route:

\`\`\`typescript
export const maxDuration = 60; // 60 seconds

export async function POST(request: NextRequest) {
  // ... existing code
}
\`\`\`

## Performance Tips

1. **Model Optimization**: Use quantized or pruned models for faster inference
2. **Caching**: Cache predictions for identical images
3. **Async Processing**: Use Celery for long-running tasks
4. **CDN**: Serve model files from CDN
5. **Load Balancing**: Use multiple workers for high traffic

## Security Best Practices

1. **Input Validation**: Validate image size and format
2. **Rate Limiting**: Prevent API abuse
3. **API Keys**: Require authentication
4. **HTTPS Only**: Never use HTTP in production
5. **Error Handling**: Don't expose internal errors

## Cost Considerations

- **Railway**: Free tier available, ~$5-20/month for production
- **Render**: Free tier available, ~$7-25/month for production
- **Fly.io**: Free tier available, pay-as-you-go
- **GPU Instances**: $50-500/month for dedicated GPUs

## Conclusion

With a Flask backend, you can use any AI model, framework, or custom training data while keeping the beautiful VisionAI interface. The client-side TensorFlow.js model works great for general use, but a Flask backend unlocks unlimited possibilities for specialized image recognition tasks.
