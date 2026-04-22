# Pixel-Art AI Generator ⚔️🤖

A generative artificial intelligence application that creates custom RPG-style pixel-art characters using a **Conditional GAN (cGAN)**. Users can select specific character traits (tags), and the AI "forges" a unique sprite based on those inputs.


## 🌟 Features

* **Custom Character Forging:** Select gender, hair style, armor type, weapons, and more.
* **Intelligent Tag System:** Built-in logic to prevent conflicting traits (e.g., no simultaneous male/female tags).
* **Smart Defaults:** If a user omits essential traits, the system automatically fills the gaps to ensure a coherent character generation.
* **Real-time Interaction:** Fast generation using a FastAPI backend and a retro-themed web interface.
* **Automatic Background Removal:** Generated sprites are processed to have transparent backgrounds, ready for game engines.

## 🛠️ Technical Stack

* **Deep Learning:** TensorFlow / Keras (Conditional GAN architecture)
* **Backend:** FastAPI (Python)
* **Frontend:** Vanilla JavaScript, CSS3, HTML5
* **Image Processing:** Pillow (PIL)
* **Data Handling:** NumPy, JSON

## 🚀 Installation & Setup

### Prerequisites
* Python 3.10 !
* Git

### 1. Clone the Repository
```bash
git clone [https://github.com/Rucol/Pixel-Art-Generator-v2.git](https://github.com/Rucol/Pixel-Art-Generator-v2.git)
cd Pixel-Art-Generator-v2
```

### Create and Activate Virtual Environment
```bash
python -m venv venv
```
# On windows:
```bash
venv\Scripts\activate
```
# On Linux/MacOS
```bash
source venv/bin/activate
```


