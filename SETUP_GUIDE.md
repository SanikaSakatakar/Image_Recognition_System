# Complete Setup Guide for VisionAI Pro

## Step-by-Step Backend Setup

### 1. Install Python Dependencies

\`\`\`bash
cd backend
pip install -r requirements.txt
\`\`\`

### 2. Choose Your ML Framework

#### Option A: Using Pre-trained Models (Recommended for Testing)

**TensorFlow with ImageNet:**
\`\`\`python
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

# Load pre-trained model
model = ResNet50(weights='imagenet')

# In your analyze endpoint:
image = image.resize((224, 224))
image_array = preprocess_input(np.array(image))
image_array = np.expand_dims(image_array, axis=0)

predictions = model.predict(image_array)
decoded = decode_predictions(predictions, top=5)[0]

results = [
    {"label": label, "confidence": float(confidence) * 100}
    for (_, label, confidence) in decoded
]
\`\`\`

**PyTorch with torchvision:**
\`\`\`python
import torch
from torchvision import models, transforms

# Load pre-trained model
model = models.resnet50(pretrained=True)
model.eval()

# Preprocessing
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# In your analyze endpoint:
input_tensor = preprocess(image)
input_batch = input_tensor.unsqueeze(0)

with torch.no_grad():
    output = model(input_batch)
    
probabilities = torch.nn.functional.softmax(output[0], dim=0)
\`\`\`

#### Option B: Using Your Custom Model

Replace the model loading section with your trained model:

\`\`\`python
# Load your custom model
model = load_model('path/to/your/model.h5')  # TensorFlow
# or
model = torch.load('path/to/your/model.pth')  # PyTorch
\`\`\`

### 3. Start Flask Server

\`\`\`bash
python app.py
\`\`\`

Server will start on `http://localhost:5000`

### 4. Test Backend API

\`\`\`bash
curl -X POST http://localhost:5000/analyze \
  -H "Content-Type: application/json" \
  -d '{"image": "data:image/jpeg;base64,YOUR_BASE64_IMAGE"}'
\`\`\`

## Frontend Configuration

### 1. Set Environment Variables

In your Vercel project or v0 workspace, add:

\`\`\`
FLASK_API_URL=http://localhost:5000
\`\`\`

For production, update to your deployed Flask URL:
\`\`\`
FLASK_API_URL=https://your-flask-api.com
\`\`\`

### 2. Test Connection

1. Upload an image in the UI
2. Click "Analyze with AI"
3. Check browser console for connection logs
4. Verify results appear in the history panel

## Common Issues and Solutions

### Issue: CORS Errors

**Solution**: Make sure flask-cors is installed and configured:
\`\`\`python
from flask_cors import CORS
CORS(app)
\`\`\`

### Issue: Large Images Causing Timeout

**Solution**: Add image size limits or compression:
\`\`\`python
# Resize large images before processing
max_size = (1024, 1024)
image.thumbnail(max_size, Image.Resampling.LANCZOS)
\`\`\`

### Issue: Model Loading is Slow

**Solution**: Load model once at startup, not per request:
\`\`\`python
# Load model globally, not inside the endpoint
model = load_model_once()

@app.route('/analyze', methods=['POST'])
def analyze_image():
    # Use pre-loaded model
    predictions = model.predict(image_array)
\`\`\`

### Issue: Memory Errors with Large Batches

**Solution**: Process images one at a time or implement batch size limits

## Production Deployment

### Deploy Flask Backend

**Option 1: Railway**
\`\`\`bash
railway login
railway init
railway up
\`\`\`

**Option 2: Render**
1. Connect your GitHub repository
2. Create new Web Service
3. Set build command: `pip install -r requirements.txt`
4. Set start command: `gunicorn app:app`

**Option 3: Google Cloud Run**
\`\`\`bash
gcloud run deploy visionai-backend \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
\`\`\`

### Deploy Next.js Frontend

1. Click "Publish" in v0
2. Add production environment variables in Vercel
3. Deploy

## Performance Optimization

### Backend Optimization

1. **Use Gunicorn** for production:
\`\`\`bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
\`\`\`

2. **Enable Response Caching** for repeated images

3. **Optimize Model**: Use TensorFlow Lite or ONNX for faster inference

### Frontend Optimization

1. **Image Compression**: Compress images before sending
2. **Debouncing**: Prevent rapid repeated requests
3. **Progressive Loading**: Show results as they complete

## Next Steps

- Add user authentication
- Implement result persistence (database)
- Add more detailed analytics
- Support video analysis
- Add model selection (multiple AI models)
- Implement rate limiting
- Add webhook notifications

## Support

For issues or questions:
- Check the browser console for detailed error messages
- Review Flask server logs
- Verify API response format matches expected structure
