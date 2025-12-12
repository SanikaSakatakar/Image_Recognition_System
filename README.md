# VisionAI - Real-Time Image Recognition Platform

A modern image recognition application with **real AI** powered by TensorFlow.js and Google's MobileNet v2 neural network. No backend required - everything runs in your browser!

## Features

- **Real AI Classification**: Uses Google MobileNet v2 trained on 1.4M images (1000 categories)
- **Client-Side Processing**: Images never leave your device - complete privacy
- **Batch Upload**: Upload and analyze multiple images simultaneously
- **Instant Results**: Get predictions in 1-2 seconds per image
- **Top 5 Predictions**: See confidence scores for multiple possibilities
- **History Management**: View, download, and manage analysis history
- **Modern UI**: Beautiful dark mode interface with glassmorphism and animations
- **Statistics Dashboard**: Track total analyses, success rate, and average confidence
- **Offline Capable**: Works offline after first model download

## How It Works

VisionAI uses **TensorFlow.js** to run a real convolutional neural network directly in your browser:

1. **First Visit**: Downloads ~5MB MobileNet v2 model (cached for future use)
2. **Upload**: Select images from your device
3. **Analysis**: Neural network processes images with 53 convolutional layers
4. **Results**: Get accurate predictions from 1000+ trained categories

**Categories Include**: Animals, vehicles, food, household objects, nature, electronics, and more!

## Getting Started

### No Setup Required!

The app works immediately with built-in AI:
1. Upload any image (JPG, PNG, WEBP)
2. Click "Analyze with AI"
3. Get real predictions instantly

### Model Loading

On first use, the app downloads the MobileNet model (~5MB):
- Takes 2-3 seconds on fast connections
- Cached permanently in browser
- Future visits are instant

## Accuracy

- **Top-1 Accuracy**: ~71% (correct answer first)
- **Top-5 Accuracy**: ~90% (correct in top 5 predictions)
- **Trained on**: ImageNet dataset (1.4M images, 1000 classes)

### Works Best With:
✅ Common animals and pets
✅ Vehicles and transportation
✅ Food and beverages
✅ Household objects
✅ Nature scenes
✅ Well-lit, clear images

### May Struggle With:
❌ Abstract art or heavily edited images
❌ Very rare or niche objects
❌ Blurry or low-resolution images
❌ Complex scenes with many objects

## Advanced: Custom Flask Backend (Optional)

Want to use your own trained models? Follow these steps:

### 1. Create Flask API

See `FLASK_INTEGRATION_GUIDE.md` for complete setup instructions.

### 2. Set Environment Variable

In the **Vars section** of v0, add:
\`\`\`
FLASK_API_URL=https://your-flask-api.railway.app
\`\`\`

### 3. Deploy Flask Backend

Deploy to Railway, Render, or Fly.io (see integration guide)

The app will automatically switch from TensorFlow.js to your Flask backend!

## Usage

### Basic Workflow

1. **Upload Images**: Click upload area or drag and drop
2. **Multiple Selection**: Add as many images as you want
3. **Analyze**: Click "Analyze X Images" button
4. **View Results**: See predictions with confidence scores
5. **Export Data**: Download results as JSON
6. **Clear History**: Remove old analyses

### Understanding Results

Each result shows:
- **Label**: Main prediction (highest confidence)
- **Confidence**: Accuracy percentage (0-100%)
- **Top 5 Predictions**: Alternative classifications
- **Timestamp**: When analysis was performed

**Confidence Levels**:
- 🟢 80-100%: Very confident
- 🟡 60-79%: Moderately confident
- 🟠 0-59%: Less confident (image may be ambiguous)

## Privacy & Security

Your images are 100% private:
- ✅ All processing happens in your browser
- ✅ Images never uploaded to servers
- ✅ No data collection or tracking
- ✅ Works completely offline (after model download)

## Tech Stack

- **Frontend**: Next.js 16, React 19, TypeScript
- **AI**: TensorFlow.js 4.22, MobileNet v2.1.1
- **Styling**: Tailwind CSS v4, shadcn/ui components
- **Neural Network**: 53-layer CNN trained on ImageNet
- **GPU**: WebGL acceleration (automatic)

## Performance

- **Model Size**: ~5MB (one-time download)
- **Inference Time**: 50-200ms per image
- **GPU Accelerated**: Uses WebGL for 10-100x speedup
- **Memory Efficient**: Processes images sequentially
- **Cache Persistent**: Model stored in IndexedDB

## Documentation

- **HOW_IT_WORKS.md**: Technical deep dive into the AI architecture
- **FLASK_INTEGRATION_GUIDE.md**: Complete guide for custom backends

## Browser Support

Requires modern browser with:
- WebGL support (for GPU acceleration)
- ES6+ JavaScript
- IndexedDB (for model caching)

Tested on:
- ✅ Chrome 90+
- ✅ Firefox 88+
- ✅ Safari 14+
- ✅ Edge 90+

## Troubleshooting

### Model Loading Failed

**Issue**: "Unable to load image recognition model"

**Solution**: 
- Refresh the page to retry download
- Check internet connection
- Clear browser cache and reload

### Low Accuracy

**Issue**: Wrong predictions or low confidence scores

**Solution**:
- Use well-lit, clear images
- Ensure subject is in focus
- Try different angles
- Check if object is in 1000 ImageNet categories

### Slow Performance

**Issue**: Analysis takes too long

**Solution**:
- Enable GPU acceleration in browser settings
- Close other tabs to free memory
- Use smaller image files (<5MB)

## Deployment

### Deploy to Vercel

1. Click "Publish" button in v0
2. Project deploys automatically
3. Share the URL with anyone!

No environment variables needed for basic functionality.

## Advanced Features

### Custom Models

Replace MobileNet with other TensorFlow.js models:
- ResNet
- EfficientNet
- Custom trained models

See technical documentation for implementation details.

### Flask Backend Integration

For specialized AI tasks:
- Medical image analysis
- Face recognition
- Object detection
- Custom trained models (PyTorch, TensorFlow)

Follow the Flask Integration Guide for complete setup.

## License

MIT

## Credits

- **Model**: Google MobileNet v2
- **Framework**: TensorFlow.js
- **Training Data**: ImageNet (Stanford University)
- **UI Components**: shadcn/ui
