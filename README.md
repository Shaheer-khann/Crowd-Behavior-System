# 🎯 Crowd Behavior Analysis Dashboard

An AI-powered web application that classifies crowd behavior from aerial drone footage using a **CNN-LSTM hybrid deep learning model**. Upload a video and get instant predictions across four behavior categories — Normal, Panic, Aggressive, or Idle.

---

## 📸 Overview

The system extracts frames from uploaded drone footage, runs a sliding-window CNN-LSTM inference pipeline, and returns a majority-voted crowd behavior prediction with a detailed confidence breakdown.

---

## 🧠 Model Architecture

| Layer | Details |
|---|---|
| **Backbone** | MobileNetV2 (frozen, ImageNet weights) |
| **Temporal Wrapper** | TimeDistributed |
| **Pooling** | GlobalAveragePooling2D |
| **Sequence Model** | LSTM (64 units) |
| **Dense** | 128 → ReLU |
| **Output** | 4 → Softmax |
| **Input Shape** | `(1, 16, 112, 112, 3)` |
| **Test Accuracy** | 100% |

---

## 📊 Supported Behavior Classes

| Class | Description | Severity |
|---|---|---|
| 🟢 **Normal** | Calm, orderly crowd movement at a typical pace | Safe |
| 🔴 **Panic** | Rapid, erratic dispersal — fear or emergency response | Critical |
| 🟠 **Aggressive** | Hostile interactions, dense clustering, localized conflict | Danger |
| 🔵 **Idle** | Stationary crowd with minimal movement — queuing or waiting | Monitor |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- pip
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/crowd-behavior-analysis.git
cd crowd-behavior-analysis
```

### 2. Create a Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Add the Trained Model

The model file (`model.h5`) is **not included** in this repository due to its size (~13 MB). You have two options:

**Option A — Use the pre-trained model:**
- Download `model.h5` from the shared link *(add your link here)*
- Place it in the **project root** directory

**Option B — Train from scratch:**
- Open `Code.ipynb` in Jupyter Notebook
- Follow the steps to prepare data and train the model
- The notebook will save `model.h5` to the project root automatically

### 5. Run the Server

```bash
python app.py
```

Open your browser and navigate to **http://localhost:5000**

---

## 📁 Project Structure

```
Crowd Behavior System/
│
├── app.py                              # Flask backend — API routes, frame extraction, inference
├── model.h5                       # ⚠️ Trained model weights (gitignored — add manually)
├── requirements.txt                    # Python dependencies
├── README.md                           # This file
├── .gitignore                          # Git exclusion rules
│
├── templates/
│   └── index.html                      # Frontend dashboard (HTML + Bootstrap 5)
│
├── static/
│   ├── style.css                       # Custom dark-theme styling & animations
│   ├── script.js                       # Frontend logic (drag-drop, API fetch, results display)
│   └── logo-1.png                      # App favicon / logo
│
├── uploads/                            # ⚠️ Temp storage for uploaded videos (auto-cleaned, gitignored)
│
├── Code.ipynb             # Model training notebook (data prep → train → evaluate → save)
└── Scripts of renaming and frame making.ipynb  # Dataset preprocessing — video → frames, file renaming
```

> **Note:** `test_frames/`, `test2/`, and `crowd_behavior_data/` are excluded from Git via `.gitignore`.

---

## 📓 Notebooks

### `Code.ipynb`
The main model training pipeline:
- Loads and preprocesses the dataset from `crowd_behavior_data/`
- Builds the CNN-LSTM architecture (MobileNetV2 backbone + LSTM)
- Trains, validates, and evaluates the model
- Saves the best weights as `model.h5`

### `Scripts of renaming and frame making.ipynb`
Dataset preparation utilities:
- Extracts frames from raw video files
- Renames and organizes frame files into class folders
- Prepares the dataset structure expected by the training notebook

---

## 🔌 API Reference

### `GET /`
Serves the web dashboard.

---

### `GET /api/status`
Health check endpoint.

**Response:**
```json
{
  "status": "online",
  "service": "Crowd Behavior Analysis API",
  "model": "model.h5",
  "classes": ["normal", "panic", "aggressive", "idle"]
}
```

---

### `POST /predict`
Upload a video file for behavior classification.

**Request:** `multipart/form-data` with a `video` field.

**Supported formats:** MP4, AVI, MOV, MKV, WEBM — max **100 MB**

**Minimum requirement:** At least **16 frames** must be extractable from the video.

**Success Response `200`:**
```json
{
  "success": true,
  "total_frames_extracted": 1672,
  "total_windows_analysed": 208,
  "dominant_prediction": "normal",
  "dominant_count": 185,
  "prediction_summary": {
    "normal": 185,
    "panic": 3,
    "aggressive": 8,
    "idle": 12
  },
  "window_details": [ "... up to 50 entries ..." ],
  "window_details_truncated": true
}
```

**Error Response `422` (video too short):**
```json
{
  "success": false,
  "error": "Video is too short. Need at least 16 frames but only got 5."
}
```

**Error Response `400` (bad request):**
```json
{
  "success": false,
  "error": "Unsupported file type. Allowed: mp4, avi, mov, mkv, webm"
}
```

---

## ⚙️ Configuration

All settings can be overridden via environment variables — no code changes needed:

| Variable | Default | Description |
|---|---|---|
| `MODEL_PATH` | `model.h5` | Path to the trained Keras model file |
| `STRIDE` | `8` | Frames to advance between sliding windows |
| `TARGET_FPS` | `8` | Max FPS to sample during frame extraction |
| `MAX_WINDOW_DETAILS` | `50` | Max window entries returned in the JSON response |
| `FLASK_ENV` | `development` | Set to `production` to use Waitress WSGI server |
| `PORT` | `5000` | Server port (production mode only) |

---

## 🏗️ Production Deployment

To run with the production-grade **Waitress** WSGI server:

**Windows (PowerShell):**
```powershell
$env:FLASK_ENV = "production"
python app.py
```

**Linux / macOS:**
```bash
FLASK_ENV=production python app.py
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Backend** | Flask, Flask-CORS |
| **ML / Inference** | TensorFlow 2.x / Keras, OpenCV, NumPy |
| **Frontend** | HTML5, Bootstrap 5, Vanilla JavaScript |
| **WSGI Server** | Waitress (production) |
| **Model** | CNN-LSTM (MobileNetV2 + LSTM) |
| **Fonts** | Inter, Space Grotesk (Google Fonts) |

---

## ⚠️ Known Limitations

- The model was trained on a specific aerial drone dataset — accuracy may vary on footage from very different camera angles or heights.
- Videos shorter than 16 sampled frames will be rejected.
- The `window_details` in the JSON response is capped at 50 entries to keep payload size manageable.
- The model file must be placed manually (not included in the repo due to file size).

---

