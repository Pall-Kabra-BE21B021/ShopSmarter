import os
import numpy as np
import torch
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image
import clip
import logging
from datetime import datetime

# Configuration
PRODUCT_IMAGES_DIR = 'product_images'
UPLOADS_DIR = 'uploads'
FEATURES_FILE = 'features.npy'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize Flask
app = Flask(__name__)
CORS(app)
os.makedirs(UPLOADS_DIR, exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ShopSmarter")

# Load CLIP model
model, preprocess = clip.load("ViT-B/32", device=DEVICE)

def initialize_features():
    """Automatically generate features on first run or when images change"""
    image_files = [f for f in os.listdir(PRODUCT_IMAGES_DIR) 
                  if f.lower().endswith(('jpg', 'jpeg', 'png'))]
    
    if not image_files:
        raise ValueError("No product images found in 'product_images' directory")
    
    features = []
    for img_file in image_files:
        img_path = os.path.join(PRODUCT_IMAGES_DIR, img_file)
        image = preprocess(Image.open(img_path)).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            features.append(model.encode_image(image).cpu().numpy()[0])
    
    features = np.stack(features)
    np.save(FEATURES_FILE, features)
    return image_files, features

# Load or create features
try:
    if os.path.exists(FEATURES_FILE):
        product_files = [f for f in os.listdir(PRODUCT_IMAGES_DIR) 
                        if f.lower().endswith(('jpg', 'jpeg', 'png'))]
        features = np.load(FEATURES_FILE)
        
        # Validate feature-product alignment
        if len(product_files) != features.shape[0]:
            raise ValueError("Product images and features mismatch")
    else:
        product_files, features = initialize_features()
except Exception as e:
    logger.error(f"Feature initialization failed: {str(e)}")
    raise

@app.route('/api/recommend', methods=['POST'])
def handle_recommendation():
    """Multimodal recommendation endpoint"""
    start_time = datetime.now()
    
    # Process inputs
    image_file = request.files.get('image')
    text_query = request.form.get('query', '').strip()
    
    # Validate inputs
    if not image_file and not text_query:
        return jsonify({"error": "No input provided"}), 400
    
    # Process image
    image_feat = None
    if image_file:
        if not image_file.filename.lower().endswith(('jpg', 'jpeg', 'png')):
            return jsonify({"error": "Invalid image format"}), 400
            
        upload_path = os.path.join(UPLOADS_DIR, image_file.filename)
        image_file.save(upload_path)
        
        try:
            image = preprocess(Image.open(upload_path)).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                image_feat = model.encode_image(image).cpu().numpy()[0]
        except Exception as e:
            logger.error(f"Image processing failed: {str(e)}")
            return jsonify({"error": "Invalid image file"}), 400

    # Process text
    text_feat = None
    if text_query:
        try:
            text = clip.tokenize([text_query]).to(DEVICE)
            with torch.no_grad():
                text_feat = model.encode_text(text).cpu().numpy()[0]
        except Exception as e:
            logger.error(f"Text processing failed: {str(e)}")
            return jsonify({"error": "Text processing error"}), 400

    # Calculate similarities
    if image_feat is not None and text_feat is not None:
        scores = 0.7 * (features @ image_feat) + 0.3 * (features @ text_feat)
    elif image_feat is not None:
        scores = features @ image_feat
    else:
        scores = features @ text_feat

    # Get top 5 results
    top_indices = np.argsort(scores)[::-1][:5]
    results = [{
        "id": str(i),
        "filename": product_files[idx],
        "score": float(scores[idx]),
        "type": "product"
    } for i, idx in enumerate(top_indices)]
    
    logger.info(f"Processed request in {(datetime.now()-start_time).total_seconds():.2f}s")
    return jsonify({"results": results})

@app.route('/api/products/<filename>')
def serve_product(filename):
    """Serve product images"""
    return send_from_directory(PRODUCT_IMAGES_DIR, filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
