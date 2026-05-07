#!/usr/bin/env python3
"""
Plant Piano Web App - Backend Server
Flask + SocketIO for real-time plant music
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import os
import numpy as np
import joblib
from scipy.io import wavfile
from scipy import signal as sp_signal
import base64
import io
import threading
import time

app = Flask(__name__)
app.config['SECRET_KEY'] = 'plant-piano-secret'
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Global state
current_model = None
current_scaler = None
current_features = None
models_cache = {}
processing_active = False

# Musical notes - Extended scale
NOTES = {
    'C3': 130.81,
    'D3': 146.83,
    'E3': 164.81,
    'G3': 196.00,
    'A3': 220.00,
    'C4': 261.63,
    'D4': 293.66,
    'E4': 329.63,
    'G4': 392.00,
    'A4': 440.00,
    'C5': 523.25
}

def load_model(model_name):
    """Load a trained model"""
    global current_model, current_scaler, current_features, models_cache
    
    if model_name in models_cache:
        data = models_cache[model_name]
    else:
        model_path = f'output/models/{model_name}_model.pkl'
        if not os.path.exists(model_path):
            return False
        
        data = joblib.load(model_path)
        models_cache[model_name] = data
    
    current_model = data['model']
    current_scaler = data['scaler']
    current_features = data['feature_names']
    
    return True

def extract_features(window):
    """Extract features from audio window"""
    features = {}
    features['rms'] = np.sqrt(np.mean(window**2))
    features['peak_to_peak'] = np.ptp(window)
    features['std'] = np.std(window)
    features['percentile_90'] = np.percentile(np.abs(window), 90)
    features['mean_abs'] = np.mean(np.abs(window))
    return features

def amplitude_to_note(amplitude):
    """
    Map amplitude to musical note (extended scale)
    
    Amplitude ranges:
    < 0.004:  C3  (very gentle)
    0.004-0.008:  D3  (gentle)
    0.008-0.012:  E3  (light)
    0.012-0.020:  G3  (medium-light)
    0.020-0.028:  A3  (medium)
    0.028-0.035:  C4  (medium-firm)
    0.035-0.045:  D4  (firm)
    0.045-0.055:  E4  (strong)
    0.055-0.070:  G4  (very strong)
    0.070-0.090:  A4  (hard)
    > 0.090:  C5  (very hard)
    """
    if amplitude < 0.004:
        return 'C3'
    elif amplitude < 0.008:
        return 'D3'
    elif amplitude < 0.012:
        return 'E3'
    elif amplitude < 0.020:
        return 'G3'
    elif amplitude < 0.028:
        return 'A3'
    elif amplitude < 0.035:
        return 'C4'
    elif amplitude < 0.045:
        return 'D4'
    elif amplitude < 0.055:
        return 'E4'
    elif amplitude < 0.070:
        return 'G4'
    elif amplitude < 0.090:
        return 'A4'
    else:
        return 'C5'

def process_wav_file(file_data, threshold=0.7, model_name='random_forest'):
    """Process uploaded WAV file"""
    
    print(f"\n>>> Starting process_wav_file")
    print(f"    Model: {model_name}")
    print(f"    Threshold: {threshold}")
    
    # Load model
    print(f">>> Loading model...")
    if not load_model(model_name):
        print(f"    ❌ Model not found: {model_name}")
        return {'error': 'Model not found'}
    print(f"    ✓ Model loaded")
    
    try:
        print(f">>> Decoding audio data...")
        # Decode base64 audio
        if ',' in file_data:
            audio_bytes = base64.b64decode(file_data.split(',')[1])
        else:
            audio_bytes = base64.b64decode(file_data)
        print(f"    ✓ Decoded {len(audio_bytes)} bytes")
        
        print(f">>> Loading WAV from bytes...")
        # Load audio
        with io.BytesIO(audio_bytes) as audio_buffer:
            sr, audio = wavfile.read(audio_buffer)
        print(f"    ✓ Loaded audio: {sr} Hz, {audio.shape}, {audio.dtype}")
        
        print(f">>> Converting to float...")
        # Convert to float
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float32) / 2147483648.0
        print(f"    ✓ Converted, range: [{np.min(audio):.6f}, {np.max(audio):.6f}]")
        
        print(f">>> Handling stereo...")
        # Handle stereo
        if len(audio.shape) > 1:
            print(f"    Converting stereo to mono")
            audio = np.mean(audio, axis=1)
        print(f"    ✓ Shape: {audio.shape}")
        
        print(f">>> Resampling...")
        # Downsample to 10kHz if needed
        target_sr = 10000
        if sr != target_sr:
            print(f"    Resampling from {sr} to {target_sr} Hz")
            num_samples = int(len(audio) * target_sr / sr)
            audio = sp_signal.resample(audio, num_samples)
            sr = target_sr
        print(f"    ✓ Sample rate: {sr} Hz, length: {len(audio)/sr:.1f}s")
        
        print(f">>> Removing DC offset...")
        # Remove DC
        audio = audio - np.mean(audio)
        print(f"    ✓ Mean: {np.mean(audio):.9f}")
        
        print(f">>> Processing windows...")
        # Process with sliding window
        window_size = int(5.0 * sr)
        hop_size = int(0.5 * sr)
        
        detections = []
        waveform_data = []
        
        total_windows = (len(audio) - window_size) // hop_size
        print(f"    Expected windows: {total_windows}")
        
        for i, start_idx in enumerate(range(0, len(audio) - window_size, hop_size)):
            window = audio[start_idx:start_idx + window_size]
            
            # Extract features
            features = extract_features(window)
            
            # Predict
            X = np.array([[features[f] for f in current_features]])
            X_scaled = current_scaler.transform(X)
            probability = current_model.predict_proba(X_scaled)[0, 1]
            
            time_sec = start_idx / sr
            
            # Store waveform sample
            waveform_data.append({
                'time': float(time_sec),
                'amplitude': float(features['peak_to_peak']),
                'probability': float(probability)
            })
            
            if probability > threshold:
                note = amplitude_to_note(features['peak_to_peak'])
                detections.append({
                    'time': float(time_sec),
                    'probability': float(probability),
                    'amplitude': float(features['peak_to_peak']),
                    'note': note,
                    'frequency': float(NOTES[note])
                })
            
            # Progress logging
            if i % 50 == 0:
                print(f"    Progress: {i}/{total_windows} ({i*100//total_windows}%)", end='\r')
        
        print(f"\n    ✓ Processed {total_windows} windows")
        print(f"    Raw detections: {len(detections)}")
        
        print(f">>> Filtering detections...")
        # Remove duplicate detections (too close together)
        filtered_detections = []
        last_time = -999
        for det in detections:
            if det['time'] - last_time >= 5.0:  # 1 second cooldown
                filtered_detections.append(det)
                last_time = det['time']
        
        print(f"    ✓ Filtered detections: {len(filtered_detections)}")
        
        result = {
            'success': True,
            'duration': float(len(audio) / sr),
            'detections': filtered_detections,
            'waveform': waveform_data[::10],  # Downsample for visualization
            'total_detections': int(len(filtered_detections))
        }
        
        print(f">>> SUCCESS! Returning result")
        return result
        
    except Exception as e:
        print(f">>> ❌ EXCEPTION in process_wav_file:")
        print(f"    {str(e)}")
        import traceback
        traceback.print_exc()
        return {'error': str(e)}

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/api/models')
def get_models():
    """Get list of available models"""
    models_dir = 'output/models'
    models = []
    
    if os.path.exists(models_dir):
        for file in os.listdir(models_dir):
            if file.endswith('_model.pkl'):
                model_name = file.replace('_model.pkl', '')
                models.append({
                    'name': model_name,
                    'display_name': model_name.replace('_', ' ').title()
                })
    
    return jsonify(models)

@app.route('/api/process', methods=['POST'])
def process_audio():
    """Process uploaded audio file"""
    try:
        print("\n" + "="*70)
        print("NEW PROCESSING REQUEST")
        print("="*70)
        
        data = request.json
        print(f"Received request with:")
        print(f"  - Threshold: {data.get('threshold', 0.7)}")
        print(f"  - Model: {data.get('model', 'random_forest')}")
        print(f"  - Audio data length: {len(data.get('audio', ''))}")
        
        # Start processing in a separate thread to allow progress updates
        result = process_wav_file(
            data['audio'],
            threshold=data.get('threshold', 0.7),
            model_name=data.get('model', 'random_forest')
        )
        
        print(f"\nProcessing result:")
        if 'error' in result:
            print(f"  ❌ Error: {result['error']}")
        else:
            print(f"  ✓ Success!")
            print(f"  - Duration: {result.get('duration', 0):.1f}s")
            print(f"  - Detections: {result.get('total_detections', 0)}")
        print("="*70 + "\n")
        
        return jsonify(result)
        
    except Exception as e:
        print(f"\n❌ EXCEPTION in process_audio:")
        print(f"  Error: {str(e)}")
        import traceback
        traceback.print_exc()
        print("="*70 + "\n")
        
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print('Client connected')
    emit('connection_response', {'status': 'connected'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print('Client disconnected')

@socketio.on('start_live')
def handle_start_live(data):
    """Start live audio processing"""
    global processing_active
    processing_active = True
    
    # Load model
    model_name = data.get('model', 'random_forest')
    if not load_model(model_name):
        emit('error', {'message': 'Model not found'})
        return
    
    emit('live_started', {'status': 'started'})

@socketio.on('stop_live')
def handle_stop_live():
    """Stop live audio processing"""
    global processing_active
    processing_active = False
    emit('live_stopped', {'status': 'stopped'})

@socketio.on('audio_chunk')
def handle_audio_chunk(data):
    """Process live audio chunk"""
    if not processing_active:
        return
    
    try:
        # Decode audio chunk
        audio_bytes = base64.b64decode(data['audio'].split(',')[1])
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Process (simplified for real-time)
        features = extract_features(audio_array)
        
        X = np.array([[features[f] for f in current_features]])
        X_scaled = current_scaler.transform(X)
        probability = current_model.predict_proba(X_scaled)[0, 1]
        
        threshold = data.get('threshold', 0.7)
        
        result = {
            'probability': float(probability),
            'amplitude': float(features['peak_to_peak']),
            'detected': bool(probability > threshold)
        }
        
        if result['detected']:
            result['note'] = amplitude_to_note(features['peak_to_peak'])
            result['frequency'] = NOTES[result['note']]
        
        emit('audio_result', result)
        
    except Exception as e:
        emit('error', {'message': str(e)})

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    print("""
╔══════════════════════════════════════════════════════════════════════════╗
║                    🌱 PLANT PIANO WEB SERVER 🎹                          ║
╚══════════════════════════════════════════════════════════════════════════╝

Starting server...
Open your browser to: http://localhost:5000

    """)
    
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
