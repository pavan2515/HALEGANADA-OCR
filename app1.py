"""
Kannada OCR - Train on the Fly with Character Variants Support
Architecture: Input -> ResNet50 (frozen) -> Transformer -> FC Layers -> Decoder CLS (vocab size)
Features: Auto-save, Resume training, Graceful shutdown with Ctrl+C, JSON tracking, Image storage
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from flask import Flask, render_template, request, jsonify
import numpy as np
from PIL import Image
import io
import base64
import os
import pickle
import signal
import sys
import atexit
import json
from datetime import datetime

app = Flask(__name__)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Global variables
model = None
char_map = {}
reverse_char_map = {}
char_variants = {}  # Maps canonical char -> set of variants
canonical_map = {}  # Maps any char -> its canonical form
next_idx = 0
training_history = []
optimizer = None
unsaved_changes = False  # Track if we need to save

# Paths
SAVE_DIR = 'saved_model'
IMAGES_DIR = os.path.join(SAVE_DIR, 'training_images')
MODEL_PATH = os.path.join(SAVE_DIR, 'kannada_r50_transformer.pt')
METADATA_PATH = os.path.join(SAVE_DIR, 'metadata.pkl')
HISTORY_PATH = os.path.join(SAVE_DIR, 'history.pkl')
JSON_LOG_PATH = os.path.join(SAVE_DIR, 'training_log.json')
SUMMARY_PATH = os.path.join(SAVE_DIR, 'training_summary.json')

def ensure_dirs():
    """Create necessary directories"""
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)

def save_training_image(image_data, char, sample_id):
    """Save training image to disk"""
    try:
        # Create character-specific directory
        char_dir = os.path.join(IMAGES_DIR, char)
        os.makedirs(char_dir, exist_ok=True)
        
        # Decode and save image
        image_data_clean = image_data.split(',')[1] if ',' in image_data else image_data
        image_bytes = base64.b64decode(image_data_clean)
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Save with timestamp
        image_path = os.path.join(char_dir, f'{sample_id}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        image.save(image_path)
        
        return image_path
    except Exception as e:
        print(f"Error saving image: {e}")
        return None

def update_json_log(entry):
    """Append entry to JSON log file"""
    try:
        # Load existing log
        if os.path.exists(JSON_LOG_PATH):
            with open(JSON_LOG_PATH, 'r', encoding='utf-8') as f:
                log_data = json.load(f)
        else:
            log_data = []
        
        # Append new entry
        log_data.append(entry)
        
        # Save updated log
        with open(JSON_LOG_PATH, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)
        
        return True
    except Exception as e:
        print(f"Error updating JSON log: {e}")
        return False

def update_training_summary():
    """Update summary statistics in JSON"""
    try:
        # Calculate statistics
        char_counts = {}
        for entry in training_history:
            char = entry.get('canonical', entry.get('char'))
            char_counts[char] = char_counts.get(char, 0) + 1
        
        # Recent losses per character
        char_losses = {}
        for entry in reversed(training_history[-100:]):
            char = entry.get('canonical', entry.get('char'))
            if char not in char_losses:
                char_losses[char] = []
            char_losses[char].append(entry.get('loss', 0))
        
        avg_losses = {char: np.mean(losses) for char, losses in char_losses.items()}
        
        # Variant information
        variant_info = {}
        for canonical, variants in char_variants.items():
            if len(variants) > 1:
                variant_info[canonical] = {
                    'variants': list(variants),
                    'count': len(variants)
                }
        
        summary = {
            'last_updated': datetime.now().isoformat(),
            'total_characters': len(char_map),
            'total_samples': len(training_history),
            'vocab_size': model.decoder_cls.out_features if model else 0,
            'variant_groups': len(variant_info),
            'character_counts': char_counts,
            'average_losses': avg_losses,
            'variant_details': variant_info,
            'character_list': list(char_map.keys()),
            'training_stats': {
                'min_loss': min([e.get('loss', float('inf')) for e in training_history]) if training_history else 0,
                'max_loss': max([e.get('loss', 0) for e in training_history]) if training_history else 0,
                'avg_loss': np.mean([e.get('loss', 0) for e in training_history]) if training_history else 0
            }
        }
        
        with open(SUMMARY_PATH, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        return True
    except Exception as e:
        print(f"Error updating summary: {e}")
        return False

class TransformerLayer(nn.Module):
    """Transformer Encoder Layer"""
    def __init__(self, d_model=512, nhead=8, dim_ff=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_ff, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x):
        # Self attention
        x2 = self.norm1(x)
        x2, _ = self.self_attn(x2, x2, x2)
        x = x + self.dropout1(x2)
        
        # Feed forward
        x2 = self.norm2(x)
        x2 = self.linear2(self.dropout(F.gelu(self.linear1(x2))))
        x = x + self.dropout2(x2)
        
        return x

class KannadaOCR(nn.Module):
    """
    Architecture:
    Input (224x224x3) -> ResNet50 (frozen) -> Transformer Layers -> FC Layers -> Decoder CLS (vocab_size)
    """
    def __init__(self, vocab_size=333, n_transformer_layers=3):
        super().__init__()
        
        # 1. ResNet50 backbone (frozen)
        resnet = models.resnet50(pretrained=True)
        self.resnet50 = nn.Sequential(*list(resnet.children())[:-1])
        
        # Freeze ResNet50
        for param in self.resnet50.parameters():
            param.requires_grad = False
        
        resnet_dim = 2048
        transformer_dim = 512
        
        # 2. Projection to transformer dimension
        self.input_projection = nn.Linear(resnet_dim, transformer_dim)
        
        # 3. Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerLayer(d_model=transformer_dim, nhead=8, dim_ff=2048, dropout=0.1)
            for _ in range(n_transformer_layers)
        ])
        
        # 4. Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(transformer_dim, 1024),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Dropout(0.3)
        )
        
        # 5. Decoder CLS head (vocab size) - can grow dynamically
        self.decoder_cls = nn.Linear(512, vocab_size)
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Extract features with frozen ResNet50
        with torch.no_grad():
            features = self.resnet50(x)
        
        features = features.view(batch_size, -1)
        
        # Project to transformer dimension
        x = self.input_projection(features)
        x = x.unsqueeze(1)
        
        # Pass through transformer layers
        for transformer in self.transformer_layers:
            x = transformer(x)
        
        x = x.squeeze(1)
        
        # Fully connected layers
        x = self.fc_layers(x)
        
        # Decoder CLS head
        logits = self.decoder_cls(x)
        
        return logits
    
    def expand_vocab(self, new_vocab_size):
        """Dynamically expand vocabulary size"""
        old_vocab_size = self.decoder_cls.out_features
        
        if new_vocab_size <= old_vocab_size:
            return  # No expansion needed
        
        print(f"📈 Expanding vocabulary: {old_vocab_size} → {new_vocab_size}")
        
        # Create new decoder layer
        old_decoder = self.decoder_cls
        new_decoder = nn.Linear(512, new_vocab_size).to(device)
        
        # Copy old weights
        with torch.no_grad():
            new_decoder.weight[:old_vocab_size] = old_decoder.weight
            new_decoder.bias[:old_vocab_size] = old_decoder.bias
        
        self.decoder_cls = new_decoder

def create_model(vocab_size=333):
    """Create model with specified vocab size"""
    model = KannadaOCR(vocab_size=vocab_size, n_transformer_layers=3)
    model = model.to(device)
    return model

def save_checkpoint():
    """Save model checkpoint"""
    global model, optimizer, char_map, reverse_char_map, next_idx, training_history
    global char_variants, canonical_map, unsaved_changes
    
    try:
        ensure_dirs()
        
        print("💾 Saving checkpoint...")
        
        # Save model and optimizer
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
            'vocab_size': model.decoder_cls.out_features
        }, MODEL_PATH)
        
        # Save metadata
        with open(METADATA_PATH, 'wb') as f:
            pickle.dump({
                'char_map': char_map,
                'reverse_char_map': reverse_char_map,
                'char_variants': char_variants,
                'canonical_map': canonical_map,
                'next_idx': next_idx
            }, f)
        
        # Save history
        with open(HISTORY_PATH, 'wb') as f:
            pickle.dump(training_history, f)
        
        # Update JSON summary
        update_training_summary()
        
        unsaved_changes = False
        print(f"✅ Saved: {len(char_map)} chars, {len(training_history)} samples, vocab_size={model.decoder_cls.out_features}")
        return True
    except Exception as e:
        print(f"❌ Save failed: {e}")
        return False

def load_checkpoint():
    """Load model checkpoint and resume training"""
    global model, optimizer, char_map, reverse_char_map, next_idx, training_history
    global char_variants, canonical_map, unsaved_changes
    
    try:
        if os.path.exists(MODEL_PATH) and os.path.exists(METADATA_PATH):
            print("📂 Loading checkpoint...")
            
            # Load metadata first
            with open(METADATA_PATH, 'rb') as f:
                meta = pickle.load(f)
                char_map = meta['char_map']
                reverse_char_map = meta['reverse_char_map']
                char_variants = meta.get('char_variants', {})
                canonical_map = meta.get('canonical_map', {})
                next_idx = meta['next_idx']
            
            # Load checkpoint to get vocab size
            checkpoint = torch.load(MODEL_PATH, map_location=device)
            vocab_size = checkpoint.get('vocab_size', max(333, next_idx))
            
            # Create model with correct vocab size
            model = create_model(vocab_size)
            
            # Load model weights
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # Create and load optimizer
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=0.001,
                weight_decay=0.01
            )
            if checkpoint['optimizer_state_dict']:
                try:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                except:
                    print("⚠️  Optimizer state mismatch, using fresh optimizer")
            
            # Load history
            if os.path.exists(HISTORY_PATH):
                with open(HISTORY_PATH, 'rb') as f:
                    training_history = pickle.load(f)
            
            unsaved_changes = False
            
            variant_count = sum(1 for v in char_variants.values() if len(v) > 1)
            print(f"✅ Loaded: {len(char_map)} chars, {len(training_history)} samples")
            print(f"   Vocab size: {vocab_size}, Variant groups: {variant_count}")
            return True
        else:
            print("📝 No checkpoint found, creating new model")
            model = create_model()
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=0.001,
                weight_decay=0.01
            )
            return False
    except Exception as e:
        print(f"❌ Load failed: {e}")
        print("   Creating new model instead")
        model = create_model()
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=0.001,
            weight_decay=0.01
        )
        return False

# Image preprocessing
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def preprocess_image(image_data):
    """Convert base64 to tensor"""
    try:
        image_data = image_data.split(',')[1] if ',' in image_data else image_data
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        tensor = preprocess(image).unsqueeze(0)
        return tensor.to(device)
    except Exception as e:
        print(f"Preprocess error: {e}")
        raise

def update_char_map(char):
    """Add character to mapping and expand model if needed"""
    global char_map, reverse_char_map, next_idx, model, unsaved_changes
    
    if char not in char_map:
        char_map[char] = next_idx
        reverse_char_map[next_idx] = char
        next_idx += 1
        unsaved_changes = True
        
        # Expand model vocabulary if needed
        current_vocab_size = model.decoder_cls.out_features
        if next_idx > current_vocab_size:
            new_vocab_size = max(current_vocab_size + 50, next_idx + 10)
            model.expand_vocab(new_vocab_size)
    
    return char_map[char]

def link_variants(char1, char2):
    """Link two characters as variants of the same letter"""
    global char_variants, canonical_map, unsaved_changes
    
    # Determine canonical form
    if char1 in canonical_map:
        canonical = canonical_map[char1]
    elif char2 in canonical_map:
        canonical = canonical_map[char2]
    else:
        canonical = char1
    
    # Update mappings
    canonical_map[char1] = canonical
    canonical_map[char2] = canonical
    
    # Add to variants set
    if canonical not in char_variants:
        char_variants[canonical] = set()
    char_variants[canonical].add(char1)
    char_variants[canonical].add(char2)
    
    unsaved_changes = True
    return canonical

@app.route('/')
def index():
    return render_template('index_bw.html')

@app.route('/train', methods=['POST'])
def train():
    """Train on the fly - continues learning from existing knowledge"""
    global model, optimizer, training_history, unsaved_changes
    
    try:
        data = request.json
        image_data = data.get('image')
        char = data.get('char')
        variant_of = data.get('variant_of')
        
        if not image_data or not char or len(char) != 1:
            return jsonify({'success': False, 'error': 'Need exactly one character'}), 400
        
        # Generate sample ID
        sample_id = len(training_history) + 1
        
        # Save training image
        image_path = save_training_image(image_data, char, sample_id)
        
        # Preprocess
        img_tensor = preprocess_image(image_data)
        
        # Handle variants
        if variant_of:
            canonical = link_variants(char, variant_of)
            char_idx = update_char_map(canonical)
            training_char = canonical
        else:
            if char in canonical_map:
                canonical = canonical_map[char]
                char_idx = update_char_map(canonical)
                training_char = canonical
            else:
                char_idx = update_char_map(char)
                training_char = char
        
        target = torch.tensor([char_idx], dtype=torch.long).to(device)
        
        # Train (continues from current state)
        model.train()
        losses = []
        
        for epoch in range(5):
            optimizer.zero_grad()
            logits = model(img_tensor)
            loss = F.cross_entropy(logits, target)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        
        avg_loss = np.mean(losses)
        
        # Create history entry
        history_entry = {
            'sample_id': sample_id,
            'char': char,
            'canonical': training_char,
            'loss': avg_loss,
            'char_idx': char_idx,
            'timestamp': datetime.now().isoformat(),
            'image_path': image_path,
            'is_variant': variant_of is not None,
            'variant_of': variant_of
        }
        
        # Update history
        training_history.append(history_entry)
        
        # Update JSON log
        json_entry = {
            'sample_id': sample_id,
            'timestamp': history_entry['timestamp'],
            'character': char,
            'canonical_form': training_char,
            'character_index': char_idx,
            'loss': float(avg_loss),
            'is_variant': variant_of is not None,
            'variant_of': variant_of,
            'image_saved': image_path is not None,
            'image_path': image_path,
            'total_chars': len(char_map),
            'total_samples': len(training_history),
            'vocab_size': model.decoder_cls.out_features
        }
        update_json_log(json_entry)
        
        unsaved_changes = True
        
        # Auto-save every 10 samples
        saved = False
        if len(training_history) % 10 == 0:
            saved = save_checkpoint()
        
        # Get variant info
        variant_info = {}
        if training_char in char_variants:
            variant_info = {
                'canonical': training_char,
                'variants': list(char_variants[training_char])
            }
        
        return jsonify({
            'success': True,
            'sample_id': sample_id,
            'loss': avg_loss,
            'chars_learned': len(char_map),
            'total_samples': len(training_history),
            'saved': saved,
            'trained_as': training_char,
            'variant_info': variant_info,
            'vocab_size': model.decoder_cls.out_features,
            'image_saved': image_path is not None
        })
        
    except Exception as e:
        print(f"Train error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Predict character"""
    global model
    
    try:
        data = request.json
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({'success': False, 'error': 'No image'}), 400
        
        # Preprocess
        img_tensor = preprocess_image(image_data)
        
        # Predict
        model.eval()
        with torch.no_grad():
            logits = model(img_tensor)
            probs = F.softmax(logits, dim=-1)
        
        # Top prediction
        pred_idx = torch.argmax(probs, dim=-1).item()
        confidence = probs[0, pred_idx].item() * 100
        pred_char = reverse_char_map.get(pred_idx, '?')
        
        # Get variant info
        variant_info = None
        if pred_char in char_variants:
            variant_info = {
                'canonical': pred_char,
                'variants': list(char_variants[pred_char])
            }
        
        # Top 3
        top_k = min(3, len(char_map))
        top_probs, top_indices = torch.topk(probs[0], top_k)
        top_3 = []
        for idx, prob in zip(top_indices.tolist(), top_probs.tolist()):
            if idx in reverse_char_map:
                char = reverse_char_map[idx]
                variants = list(char_variants.get(char, [])) if char in char_variants else None
                top_3.append({
                    'char': char,
                    'confidence': prob * 100,
                    'variants': variants
                })
        
        return jsonify({
            'success': True,
            'predicted_char': pred_char,
            'confidence': confidence,
            'char_idx': int(pred_idx),
            'variant_info': variant_info,
            'top_3': top_3
        })
        
    except Exception as e:
        print(f"Predict error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/link_variants', methods=['POST'])
def link_variants_endpoint():
    """Manually link two characters as variants"""
    try:
        data = request.json
        char1 = data.get('char1')
        char2 = data.get('char2')
        
        if not char1 or not char2:
            return jsonify({'success': False, 'error': 'Need both characters'}), 400
        
        canonical = link_variants(char1, char2)
        saved = save_checkpoint()
        
        return jsonify({
            'success': True,
            'canonical': canonical,
            'variants': list(char_variants[canonical]),
            'saved': saved
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/save', methods=['POST'])
def manual_save():
    """Manual save endpoint"""
    try:
        saved = save_checkpoint()
        return jsonify({
            'success': saved,
            'message': 'Checkpoint saved' if saved else 'Save failed'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/stats', methods=['GET'])
def stats():
    """Get stats"""
    variant_summary = {}
    for canonical, variants in char_variants.items():
        if len(variants) > 1:
            variant_summary[canonical] = list(variants)
    
    return jsonify({
        'chars_learned': len(char_map),
        'total_samples': len(training_history),
        'vocab_size': model.decoder_cls.out_features,
        'char_map': char_map,
        'variant_groups': variant_summary,
        'training_history': training_history[-10:],
        'unsaved_changes': unsaved_changes
    })

@app.route('/export_data', methods=['GET'])
def export_data():
    """Export all training data as JSON"""
    try:
        export = {
            'export_timestamp': datetime.now().isoformat(),
            'model_info': {
                'vocab_size': model.decoder_cls.out_features,
                'total_characters': len(char_map),
                'total_samples': len(training_history)
            },
            'characters': char_map,
            'variants': {k: list(v) for k, v in char_variants.items()},
            'training_history': training_history
        }
        return jsonify(export)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/reset', methods=['POST'])
def reset():
    """Reset everything"""
    global model, optimizer, char_map, reverse_char_map, next_idx, training_history
    global char_variants, canonical_map, unsaved_changes
    
    try:
        model = create_model()
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=0.001,
            weight_decay=0.01
        )
        char_map = {}
        reverse_char_map = {}
        char_variants = {}
        canonical_map = {}
        next_idx = 0
        training_history = []
        unsaved_changes = False
        
        # Delete files
        for path in [MODEL_PATH, METADATA_PATH, HISTORY_PATH, JSON_LOG_PATH, SUMMARY_PATH]:
            if os.path.exists(path):
                os.remove(path)
        
        # Clear images directory
        if os.path.exists(IMAGES_DIR):
            import shutil
            shutil.rmtree(IMAGES_DIR)
            os.makedirs(IMAGES_DIR)
        
        print("🔄 Model reset complete")
        return jsonify({'success': True, 'message': 'Reset complete'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

def cleanup_on_exit():
    """Save before exit"""
    global unsaved_changes
    if unsaved_changes:
        print("\n" + "="*70)
        print("💾 Saving changes before exit...")
        save_checkpoint()
        print("="*70)

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print("\n" + "="*70)
    print("🛑 Shutdown signal received (Ctrl+C)")
    print("="*70)
    cleanup_on_exit()
    print("✅ Goodbye!")
    sys.exit(0)

# Register cleanup handlers
atexit.register(cleanup_on_exit)
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

if __name__ == '__main__':
    print("=" * 70)
    print("🧠 KANNADA OCR - TRAIN ON THE FLY (CONTINUOUS LEARNING)")
    print("=" * 70)
    print(f"Device: {device}")
    print("Architecture: Input → ResNet50 (frozen) → Transformer → FC → Decoder")
    print("Features: Auto-save, Resume training, Graceful shutdown, JSON tracking")
    print("=" * 70)
    
    # Load existing model or create new
    loaded = load_checkpoint()
    
    if loaded:
        variant_count = sum(1 for v in char_variants.values() if len(v) > 1)
        print(f"\n✅ RESUMED TRAINING SESSION")
        print(f"   Characters learned: {len(char_map)}")
        print(f"   Training samples: {len(training_history)}")
        print(f"   Variant groups: {variant_count}")
        print(f"   Vocab size: {model.decoder_cls.out_features}")
    else:
        print(f"\n✅ NEW TRAINING SESSION")
        print(f"   Starting fresh with vocab_size=333")
    
    print("\n" + "=" * 70)
    print("📁 Data Storage:")
    print(f"   Model: {MODEL_PATH}")
    print(f"   JSON Log: {JSON_LOG_PATH}")
    print(f"   Summary: {SUMMARY_PATH}")
    print(f"   Images: {IMAGES_DIR}")
    print("=" * 70)
    print("\n💡 Tips:")
    print("   • Press Ctrl+C to stop server (auto-saves before exit)")
    print("   • Model auto-saves every 10 training samples")
    print("   • All progress is preserved between sessions")
    print("   • Images and JSON logs track every training sample")
    print("=" * 70)
    print("\n🚀 Server starting at http://localhost:5000")
    print("=" * 70 + "\n")
    
    app.run(debug=False, host='0.0.0.0', port=5000, use_reloader=False)