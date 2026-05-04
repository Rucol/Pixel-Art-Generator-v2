import json
import base64
import os
import pathlib
from io import BytesIO
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image

app = FastAPI(title="Pixel Art Generator API")

# --- PATH CONFIGURATION ---
BASE_DIR = pathlib.Path(__file__).parent.resolve()
MODEL_PATH = BASE_DIR / "model_data" / "generator_model.keras"
TAGS_PATH = BASE_DIR / "model_data" / "tags_dictionary.json"

NOISE_DIM = 256

# --- DATA LOADING ---
TAGS_LIST = []
NUM_CLASSES = 0

try:
    if TAGS_PATH.exists():
        with open(TAGS_PATH, 'r', encoding='utf-8') as f:
            TAGS_LIST = json.load(f)
        NUM_CLASSES = len(TAGS_LIST)
        print(f"✅ Success: Loaded {NUM_CLASSES} tags from {TAGS_PATH}")
    else:
        print(f"❌ ERROR: Tags dictionary not found at {TAGS_PATH}")
except Exception as e:
    print(f"❌ ERROR while reading tags: {e}")

# --- MODEL LOADING ---
GEN = None
try:
    if MODEL_PATH.exists():
        print(f"🚀 Loading model from: {MODEL_PATH}")
        # compile=False is used because we only need the model for inference
        GEN = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
        print("✅ Model loaded successfully!")
    else:
        print(f"❌ ERROR: Model file not found at {MODEL_PATH}")
except Exception as e:
    print(f"❌ CRITICAL ERROR during model initialization: {e}")

# --- API SCHEMAS ---
class GenerateRequest(BaseModel):
    tags: list[int]

# --- API ENDPOINTS ---

@app.get("/tags")
async def get_tags():
    """Returns the full list of available character tags for the UI."""
    return TAGS_LIST

@app.post("/generate")
async def generate_sprite(req: GenerateRequest):
    """
    Generates a pixel art sprite based on provided tag indices.
    Applies background removal and returns a base64 encoded PNG.
    """
    if GEN is None:
        raise HTTPException(status_code=500, detail="Model not initialized.")

    try:
        # 1. Prepare label vector (One-hot encoding for selected tags)
        label_vector = np.zeros((1, NUM_CLASSES), dtype=np.float32)
        for tag_idx in req.tags:
            if 0 <= tag_idx < NUM_CLASSES:
                label_vector[0, tag_idx] = 1.0
        
        # 2. Generate random latent noise
        noise = tf.random.normal([1, NOISE_DIM])
        
        # 3. Model Inference
        prediction = GEN([noise, label_vector], training=False)
        
        # 4. Image Processing (Denormalization: [-1, 1] -> [0, 255])
        img_array = (prediction[0].numpy() * 127.5 + 127.5).astype(np.uint8)
        img = Image.fromarray(img_array).convert("RGBA")
        
        # 5. Background Removal (Chroma keying for dark pixels)
        datas = img.getdata()
        newData = []
        for item in datas:
            # Check if pixel is near-black and set to transparent
            if item[0] < 40 and item[1] < 40 and item[2] < 40: 
                newData.append((0, 0, 0, 0))
            else:
                newData.append(item)
        img.putdata(newData)
        
        # 6. Final scaling for display
        img = img.resize((256, 256), Image.NEAREST)
        
        # 7. Convert to Base64 for web delivery
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return {"image": img_str}
        
    except Exception as e:
        print(f"❌ Generation Error: {e}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

# --- STATIC FILES SERVING ---
# Must be mounted last to avoid overriding API routes
app.mount("/", StaticFiles(directory=BASE_DIR / "static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    # Start the server on localhost
    uvicorn.run(app, host="127.0.0.1", port=8000)