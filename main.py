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

app = FastAPI()

# --- KONFIGURACJA ŚCIEŻEK ---
# Ustalenie ścieżki do folderu, w którym znajduje się main.py
BASE_DIR = pathlib.Path(__file__).parent.resolve()
MODEL_PATH = BASE_DIR / "model_data" / "generator_model.keras"
TAGS_PATH = BASE_DIR / "model_data" / "tags_dictionary.json"

NOISE_DIM = 256
# --- ŁADOWANIE DANYCH ---

TAGS_LIST = []
NUM_CLASSES = 0

try:
    if TAGS_PATH.exists():
        with open(TAGS_PATH, 'r', encoding='utf-8') as f:
            TAGS_LIST = json.load(f)
        NUM_CLASSES = len(TAGS_LIST)
        print(f"✅ Sukces: Załadowano {NUM_CLASSES} tagów z {TAGS_PATH}")
    else:
        print(f"❌ BŁĄD: Nie znaleziono pliku tagów w {TAGS_PATH}")
except Exception as e:
    print(f"❌ BŁĄD podczas odczytu tagów: {e}")

# --- ŁADOWANIE MODELU ---

GEN = None
try:
    if MODEL_PATH.exists():
        print(f"🚀 Ładowanie modelu z: {MODEL_PATH}")
        GEN = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
        print("✅ Model załadowany pomyślnie!")
    else:
        print(f"❌ BŁĄD: Nie znaleziono pliku modelu w {MODEL_PATH}")
except Exception as e:
    print(f"❌ BŁĄD krytyczny modelu: {e}")

# --- API ---

class GenerateRequest(BaseModel):
    tags: list[int]

@app.get("/tags")
async def get_tags():
    # Zwracamy listę tagów - to jest kluczowe dla działania menu!
    return TAGS_LIST

@app.post("/generate")
async def generate_sprite(req: GenerateRequest):
    if GEN is None:
        raise HTTPException(status_code=500, detail="Model nie został zainicjalizowany.")

    try:
        # 1. Tworzymy wektor o długości 2564 (tyle ile widział model)
        label_vector = np.zeros((1, NUM_CLASSES), dtype=np.float32)
        for tag_idx in req.tags:
            if 0 <= tag_idx < NUM_CLASSES:
                label_vector[0, tag_idx] = 1.0
        
        # 2. ZMIANA: Szum musi mieć 256, a nie 100!
        noise = tf.random.normal([1, NOISE_DIM]) # NOISE_DIM mamy ustawione na 256 wyżej
        
        # 3. Przekazujemy do modelu
        prediction = GEN([noise, label_vector], training=False)
        
        # Reszta kodu przetwarzania (img_array, wycinanie tła) pozostaje bez zmian
        img_array = (prediction[0].numpy() * 127.5 + 127.5).astype(np.uint8)
        img = Image.fromarray(img_array)
        img = img.convert("RGBA")
        
        # ... (twój kod wycinania tła i wysyłania base64) ...
        
        # Wycinanie tła
        datas = img.getdata()
        newData = []
        for item in datas:
            if item[0] < 40 and item[1] < 40 and item[2] < 40: 
                newData.append((0, 0, 0, 0))
            else:
                newData.append(item)
        img.putdata(newData)
        img = img.resize((256, 256), Image.NEAREST)
        
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return {"image": img_str}
    except Exception as e:
        print(f"❌ Błąd generowania: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# MONTAŻ PLIKÓW STATYCZNYCH
# Musi być na samym końcu!
app.mount("/", StaticFiles(directory=BASE_DIR / "static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)