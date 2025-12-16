# HALEGANADA-OCR
# 🧠 Kannada OCR - Train on the Fly

> A smart, interactive OCR system that learns Kannada characters as you teach them - no datasets required!

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

---

## 🌟 What Makes This Special?

This isn't your typical OCR system. Instead of needing thousands of labeled images upfront, this model **learns from you in real-time**. Draw a Kannada character, tell it what it is, and boom - it remembers! The more you teach it, the smarter it gets.

**Key Features:**
- ✨ **Zero-Shot Learning**: Start teaching immediately, no pre-training needed
- 🔄 **Continuous Learning**: Every session builds on previous knowledge
- 🎨 **Interactive Training**: Draw characters directly in your browser
- 💾 **Auto-Save**: Never lose progress, even if you hit Ctrl+C
- 🔗 **Variant Support**: Teach multiple forms of the same letter
- 📊 **Full Tracking**: JSON logs and saved images for every training sample
- 🚀 **Fast Training**: Leverages pre-trained ResNet50 for quick learning

---

## 🎯 Perfect For

- **Language Learners**: Practice writing Kannada while building a personalized OCR
- **Researchers**: Experiment with online learning and few-shot classification
- **Developers**: Build custom OCR for any script or handwriting style
- **Educators**: Create interactive tools for teaching writing systems
- **Anyone**: Who wants a character recognition system tailored to their handwriting!

---

## 🏗️ Architecture

```
Your Drawing (224×224×3)
        ↓
ResNet50 Feature Extraction (frozen) → 2048 features
        ↓
Linear Projection → 512 dimensions
        ↓
Transformer Layer 1 (Self-Attention + FFN)
        ↓
Transformer Layer 2 (Self-Attention + FFN)
        ↓
Transformer Layer 3 (Self-Attention + FFN)
        ↓
Fully Connected Layers (512 → 1024 → 512)
        ↓
Decoder Head (512 → vocab_size)
        ↓
Your Character! 🎉
```

**Why This Architecture?**
- **ResNet50 (Frozen)**: Already knows basic shapes and patterns from millions of images
- **Transformers**: Excel at capturing relationships and important features
- **Only ~3-5M trainable params**: Fast training, works on CPU or GPU
- **Dynamic Vocabulary**: Grows automatically as you teach new characters

---

## 📋 Requirements

```bash
Python 3.8+
torch >= 2.0.0
torchvision >= 0.15.0
Flask >= 2.0.0
Pillow >= 9.0.0
numpy >= 1.21.0
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/yourusername/kannada-ocr.git
cd kannada-ocr
pip install -r requirements.txt
```

### 2. Run the Server

```bash
python app.py
```

You'll see:
```
======================================================================
🧠 KANNADA OCR - TRAIN ON THE FLY (CONTINUOUS LEARNING)
======================================================================
Device: cuda
Architecture: Input → ResNet50 (frozen) → Transformer → FC → Decoder
Features: Auto-save, Resume training, Graceful shutdown, JSON tracking
======================================================================

✅ NEW TRAINING SESSION
   Starting fresh with vocab_size=333

======================================================================
📁 Data Storage:
   Model: saved_model/kannada_r50_transformer.pt
   JSON Log: saved_model/training_log.json
   Summary: saved_model/training_summary.json
   Images: saved_model/training_images
======================================================================

🚀 Server starting at http://localhost:5000
======================================================================
```

### 3. Open Your Browser

Navigate to **http://localhost:5000**

### 4. Start Teaching!

1. **Draw** a Kannada character in the canvas
2. **Type** what character it is (e.g., "ಅ")
3. **Click Train** - model learns in ~2 seconds
4. **Draw** again and **Predict** to test it!

---

## 🎨 How to Use

### Training Mode

**Teach a New Character:**
```
1. Draw "ಅ" on the canvas
2. Type "ಅ" in the input box
3. Click "Train"
4. Model learns and auto-saves every 10 samples
```

**Teach a Variant:**
```
1. Draw "ಆ" (another form of "ಅ")
2. Type "ಆ" in the input box
3. Check "This is a variant of" and select "ಅ"
4. Click "Train"
5. Now both forms map to the same character!
```

### Prediction Mode

**Test Your Model:**
```
1. Draw a character you've trained
2. Click "Predict"
3. See the prediction, confidence %, and top 3 guesses
4. If it has variants, you'll see all forms listed
```

### Managing Your Data

**View Stats:**
```bash
GET http://localhost:5000/stats
```

**Manual Save:**
```bash
POST http://localhost:5000/save
```

**Export Everything:**
```bash
GET http://localhost:5000/export_data
```

**Reset (Delete Everything):**
```bash
POST http://localhost:5000/reset
```

---

## 💾 What Gets Saved?

### File Structure
```
saved_model/
├── kannada_r50_transformer.pt      # Model weights & optimizer state
├── metadata.pkl                     # Character mappings & variants
├── history.pkl                      # Training history (Python object)
├── training_log.json               # Detailed log (human-readable)
├── training_summary.json           # Statistics & analytics
└── training_images/                # All training images organized by character
    ├── ಅ/
    │   ├── 1_20241216_143022.png
    │   └── 2_20241216_143045.png
    ├── ಆ/
    │   └── 3_20241216_143105.png
    └── ...
```

### training_log.json
Every training sample is logged:
```json
{
  "sample_id": 1,
  "timestamp": "2024-12-16T14:30:22.123456",
  "character": "ಅ",
  "canonical_form": "ಅ",
  "character_index": 0,
  "loss": 2.456,
  "is_variant": false,
  "image_path": "saved_model/training_images/ಅ/1_20241216_143022.png",
  "total_chars": 1,
  "vocab_size": 333
}
```

### training_summary.json
High-level statistics:
```json
{
  "total_characters": 45,
  "total_samples": 230,
  "character_counts": {"ಅ": 15, "ಆ": 12},
  "average_losses": {"ಅ": 0.234, "ಆ": 0.198},
  "variant_groups": 8,
  "training_stats": {
    "min_loss": 0.045,
    "max_loss": 3.456,
    "avg_loss": 0.567
  }
}
```

---

## 🔄 Continuous Learning

**Session 1:**
```bash
python app.py
# Teach 10 characters
# Ctrl+C (auto-saves before exit)
```

**Session 2 (Next Day):**
```bash
python app.py
# ✅ Loads all 10 characters automatically
# Teach 10 more characters
# Model now knows 20 characters total!
```

**The Magic:**
- All knowledge persists between sessions
- Model continues learning from where it left off
- Optimizer state preserved (maintains learning momentum)
- No need to retrain from scratch - ever!

---

## 🛠️ API Endpoints

### POST /train
Train on a single character sample.

**Request:**
```json
{
  "image": "data:image/png;base64,...",
  "char": "ಅ",
  "variant_of": null  // Optional: link to canonical form
}
```

**Response:**
```json
{
  "success": true,
  "sample_id": 1,
  "loss": 2.456,
  "chars_learned": 1,
  "total_samples": 1,
  "trained_as": "ಅ",
  "vocab_size": 333,
  "saved": false
}
```

### POST /predict
Recognize a drawn character.

**Request:**
```json
{
  "image": "data:image/png;base64,..."
}
```

**Response:**
```json
{
  "success": true,
  "predicted_char": "ಅ",
  "confidence": 95.6,
  "char_idx": 0,
  "variant_info": {
    "canonical": "ಅ",
    "variants": ["ಅ", "ಆ"]
  },
  "top_3": [
    {"char": "ಅ", "confidence": 95.6},
    {"char": "ಇ", "confidence": 3.2},
    {"char": "ಈ", "confidence": 1.2}
  ]
}
```

### POST /link_variants
Manually link two characters as variants.

**Request:**
```json
{
  "char1": "ಅ",
  "char2": "ಆ"
}
```

### GET /stats
Get current training statistics.

### GET /export_data
Export complete training data as JSON.

### POST /save
Manually trigger a save.

### POST /reset
⚠️ Delete everything and start fresh.

---

## 🧪 Advanced Features

### Character Variants

Kannada (and most scripts) have multiple forms of the same character. This system handles them elegantly:

```python
# Train base form
train("ಕ")  # standalone ka

# Train variants - they all map to "ಕ"
train("ಕಾ", variant_of="ಕ")  # ka + aa matra
train("ಕಿ", variant_of="ಕ")  # ka + i matra
train("ಕೀ", variant_of="ಕ")  # ka + ii matra

# Now all forms are recognized as "ಕ"!
```

### Dynamic Vocabulary Growth

The model automatically expands as you teach new characters:

```
Start: vocab_size = 333
Teach 50 characters → vocab_size = 383
Teach 100 more → vocab_size = 483
```

Old knowledge is preserved when vocabulary expands!

### Auto-Save Strategy

- **Every 10 samples**: Automatic checkpoint
- **On Ctrl+C**: Graceful shutdown with save
- **On exit**: atexit handler ensures save
- **Manual**: POST /save anytime

---

## 📊 Analyzing Your Training Data

### Load JSON Log in Python

```python
import json

with open('saved_model/training_log.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Find high-loss samples
high_loss = [s for s in data if s['loss'] > 1.0]

# Count samples per character
from collections import Counter
char_counts = Counter(s['character'] for s in data)
print(char_counts)
```

### Visualize Learning Progress

```python
import matplotlib.pyplot as plt
import json

with open('saved_model/training_log.json', 'r') as f:
    data = json.load(f)

# Plot loss over time
losses = [s['loss'] for s in data]
plt.plot(losses)
plt.xlabel('Sample')
plt.ylabel('Loss')
plt.title('Training Progress')
plt.show()
```

---

## 🎓 Tips for Best Results

### Training Tips

1. **Consistency is Key**: Try to draw characters similarly each time
2. **Start Simple**: Begin with distinct characters before variants
3. **More Samples = Better**: 5-10 samples per character is ideal
4. **Teach Variants**: Group similar forms together for robust recognition
5. **Review Stats**: Check average losses to see which characters need more training

### Drawing Tips

1. **Center Your Character**: Keep it in the middle of the canvas
2. **Size Matters**: Fill the canvas reasonably, but leave some margin
3. **Clear Strokes**: Draw confidently with clear, connected strokes
4. **Consistent Style**: Maintain similar handwriting style throughout

### Performance Tips

1. **GPU**: CUDA-enabled GPU will train faster (but CPU works fine!)
2. **Batch Teaching**: Teach 10-20 characters, then test them all
3. **Regular Saves**: Let auto-save run every 10 samples
4. **Clean Canvas**: Always clear before drawing new characters

---

## 🐛 Troubleshooting

### Model not loading?
```bash
# Check if files exist
ls saved_model/

# If corrupted, reset and start fresh
curl -X POST http://localhost:5000/reset
```

### High losses?
- Draw characters more consistently
- Add more training samples (5-10 per character)
- Check if similar characters are confusing the model

### Ctrl+C not saving?
- Make sure you're running with `debug=False`
- Check terminal output for save confirmation
- Use manual save: `POST /save` before closing

### Out of memory?
- Model uses ~500-800MB GPU memory
- Reduce batch size (already 1)
- Close other GPU-intensive apps
- Use CPU mode (set `device = torch.device('cpu')`)

---

## 🔬 Technical Details

### Model Architecture
- **Input**: 224×224×3 RGB images
- **Backbone**: ResNet50 (25M params, frozen)
- **Trainable**: ~3-5M parameters
- **Transformer Layers**: 3 layers, 8 attention heads
- **Hidden Dim**: 512
- **Feed-Forward Dim**: 2048
- **Dropout**: 0.1 (transformers), 0.3 (FC layers)

### Training Details
- **Optimizer**: AdamW (lr=0.001, weight_decay=0.01)
- **Loss**: CrossEntropyLoss
- **Epochs per Sample**: 5
- **Batch Size**: 1 (online learning)
- **Training Time**: ~2 seconds per sample (GPU)

### Data Preprocessing
- Resize to 256×256
- Center crop to 224×224
- Normalize: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
- Convert to RGB (3 channels)





