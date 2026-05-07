#!/usr/bin/env python3
"""
Debug script to test WAV file processing
"""

import os
import sys
import numpy as np
import joblib
from scipy.io import wavfile
from scipy import signal as sp_signal

def test_processing(wav_file, model_path='output/models/random_forest_model.pkl'):
    """Test the processing pipeline"""
    
    print(f"\n{'='*70}")
    print("TESTING PROCESSING PIPELINE")
    print(f"{'='*70}\n")
    
    # Step 1: Load model
    print("Step 1: Loading model...")
    try:
        model_data = joblib.load(model_path)
        model = model_data['model']
        scaler = model_data['scaler']
        feature_names = model_data['feature_names']
        print(f"✓ Model loaded successfully")
        print(f"  Features: {feature_names}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Step 2: Load audio
    print("\nStep 2: Loading audio file...")
    try:
        sr, audio = wavfile.read(wav_file)
        print(f"✓ Audio loaded")
        print(f"  Sample rate: {sr} Hz")
        print(f"  Data type: {audio.dtype}")
        print(f"  Shape: {audio.shape}")
        print(f"  Duration: {len(audio)/sr:.1f} seconds")
    except Exception as e:
        print(f"❌ Error loading audio: {e}")
        return
    
    # Step 3: Convert to float
    print("\nStep 3: Converting to float...")
    try:
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float32) / 2147483648.0
        print(f"✓ Converted to float32")
        print(f"  Range: [{np.min(audio):.6f}, {np.max(audio):.6f}]")
    except Exception as e:
        print(f"❌ Error converting: {e}")
        return
    
    # Step 4: Handle stereo
    print("\nStep 4: Checking channels...")
    try:
        if len(audio.shape) > 1:
            print(f"  Stereo detected, converting to mono")
            audio = np.mean(audio, axis=1)
        else:
            print(f"  Already mono")
        print(f"✓ Shape after: {audio.shape}")
    except Exception as e:
        print(f"❌ Error handling stereo: {e}")
        return
    
    # Step 5: Resample
    print("\nStep 5: Resampling...")
    try:
        target_sr = 10000
        if sr != target_sr:
            print(f"  Resampling from {sr} to {target_sr} Hz")
            num_samples = int(len(audio) * target_sr / sr)
            audio = sp_signal.resample(audio, num_samples)
            sr = target_sr
        print(f"✓ Sample rate: {sr} Hz")
        print(f"  Length: {len(audio)} samples ({len(audio)/sr:.1f}s)")
    except Exception as e:
        print(f"❌ Error resampling: {e}")
        return
    
    # Step 6: Remove DC
    print("\nStep 6: Removing DC offset...")
    try:
        audio = audio - np.mean(audio)
        print(f"✓ DC removed")
        print(f"  New mean: {np.mean(audio):.9f} (should be ~0)")
    except Exception as e:
        print(f"❌ Error removing DC: {e}")
        return
    
    # Step 7: Process windows
    print("\nStep 7: Processing windows...")
    try:
        window_size = int(5.0 * sr)
        hop_size = int(0.5 * sr)
        
        print(f"  Window size: {window_size} samples (5.0s)")
        print(f"  Hop size: {hop_size} samples (0.5s)")
        print(f"  Expected windows: {(len(audio) - window_size) // hop_size}")
        
        detections = []
        threshold = 0.7
        
        for i, start_idx in enumerate(range(0, len(audio) - window_size, hop_size)):
            window = audio[start_idx:start_idx + window_size]
            
            # Extract features
            features = {
                'rms': np.sqrt(np.mean(window**2)),
                'peak_to_peak': np.ptp(window),
                'std': np.std(window),
                'percentile_90': np.percentile(np.abs(window), 90),
                'mean_abs': np.mean(np.abs(window))
            }
            
            # Predict
            X = np.array([[features[f] for f in feature_names]])
            X_scaled = scaler.transform(X)
            probability = model.predict_proba(X_scaled)[0, 1]
            
            time_sec = start_idx / sr
            
            if probability > threshold:
                detections.append({
                    'time': time_sec,
                    'probability': probability,
                    'amplitude': features['peak_to_peak']
                })
            
            # Progress
            if i % 20 == 0:
                print(f"  Processed {i} windows... (time: {time_sec:.1f}s)", end='\r')
        
        print(f"\n✓ Processed all windows")
        print(f"  Total detections: {len(detections)}")
        
        if len(detections) > 0:
            print(f"\n  First 5 detections:")
            for det in detections[:5]:
                print(f"    {det['time']:.1f}s - Prob: {det['probability']:.1%} - Amp: {det['amplitude']:.5f}")
        
    except Exception as e:
        print(f"❌ Error processing windows: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"\n{'='*70}")
    print("✅ PROCESSING COMPLETED SUCCESSFULLY!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
    else:
        test_file = "\dataset\holy basil\touch_02.wav"
    
    if not os.path.exists(test_file):
        print(f"❌ File not found: {test_file}")
        print("\nUsage: python debug_processing.py <path_to_wav_file>")
        sys.exit(1)
    
    test_processing(test_file)
