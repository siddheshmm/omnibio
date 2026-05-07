#!/usr/bin/env python3
"""
Plant Piano - Test Mode
Test the piano on your recorded WAV files
"""

import os
import sys
import joblib
import numpy as np
from scipy.io import wavfile
from scipy import signal as sp_signal
import time

# Try to import sound library
try:
    import pygame
    pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)  # channels=2 for stereo
    SOUND_ENGINE = 'pygame'
    print("✓ Using pygame for audio")
except:
    try:
        import sounddevice as sd
        SOUND_ENGINE = 'sounddevice'
        print("✓ Using sounddevice for audio")
    except:
        print("⚠️  No audio library available")
        SOUND_ENGINE = None


def generate_note_sound(frequency, duration=0.5):
    """Generate a piano-like note"""
    sample_rate = 44100
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Piano-like sound with harmonics
    fundamental = np.sin(2 * np.pi * frequency * t)
    harmonic2 = 0.5 * np.sin(2 * np.pi * frequency * 2 * t)
    harmonic3 = 0.25 * np.sin(2 * np.pi * frequency * 3 * t)
    wave = fundamental + harmonic2 + harmonic3
    
    # Envelope
    envelope = np.exp(-3 * t)
    wave = wave * envelope
    wave = wave / np.max(np.abs(wave))
    
    # Convert to stereo for pygame (duplicate to both channels)
    if SOUND_ENGINE == 'pygame':
        wave_stereo = np.column_stack((wave, wave))
        return wave_stereo
    
    return wave


def play_note(frequency, volume=1.0):
    """Play a note"""
    if SOUND_ENGINE is None:
        return
    
    wave = generate_note_sound(frequency) * volume
    
    try:
        if SOUND_ENGINE == 'pygame':
            wave_int = (wave * 32767).astype(np.int16)
            sound = pygame.sndarray.make_sound(wave_int)
            sound.play()
        else:
            sd.play(wave, 44100)
    except Exception as e:
        print(f"Audio error: {e}")


def test_on_wav_file(model_path, wav_file, threshold=0.7):
    """Test piano on a single WAV file"""
    
    print(f"\n{'='*70}")
    print(f"Testing: {os.path.basename(wav_file)}")
    print(f"{'='*70}\n")
    
    # Load model
    model_data = joblib.load(model_path)
    model = model_data['model']
    scaler = model_data['scaler']
    feature_names = model_data['feature_names']
    
    # Load audio
    sr, audio = wavfile.read(wav_file)
    
    # Convert to float
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    
    # Handle stereo
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
    
    # Downsample to 10kHz if needed
    target_sr = 10000
    if sr != target_sr:
        num_samples = int(len(audio) * target_sr / sr)
        audio = sp_signal.resample(audio, num_samples)
        sr = target_sr
    
    # Remove DC
    audio = audio - np.mean(audio)
    
    print(f"Duration: {len(audio)/sr:.1f} seconds")
    print(f"Sample rate: {sr} Hz")
    print(f"Amplitude range: [{np.min(audio):.6f}, {np.max(audio):.6f}]")
    print()
    
    # Amplitude to Frequency mapping constants for continuous pitch
    MIN_AMP = 0.005
    MAX_AMP = 0.060
    MIN_FREQ = 261.63  # C4
    MAX_FREQ = 493.88  # B4

    def amplitude_to_frequency(amp):
        """Maps signal amplitude directly to a frequency for continuous pitch change."""
        clamped_amp = np.clip(amp, MIN_AMP, MAX_AMP)
        # Linearly interpolate amplitude to the frequency range
        freq = np.interp(clamped_amp, [MIN_AMP, MAX_AMP], [MIN_FREQ, MAX_FREQ])
        return freq
    
    # Process with sliding window
    window_size = int(5.0 * sr)  # 5 seconds
    hop_size = int(0.5 * sr)     # 0.5 second hop
    
    detections = []
    
    for start_idx in range(0, len(audio) - window_size, hop_size):
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
    
    # Print results
    if len(detections) > 0:
        print(f"✅ Detected {len(detections)} touch event(s):\n")
        
        last_time = -999
        for det in detections:
            # Skip if too close to previous (cooldown)
            if det['time'] - last_time < 5.0:
                continue
            
            last_time = det['time']
            
            # Directly map amplitude to frequency for continuous tones
            freq = amplitude_to_frequency(det['amplitude'])
            
            print(f"  🎵 {det['time']:5.1f}s | Freq: {freq:.0f} Hz | "
                  f"Prob: {det['probability']:.1%} | Amp: {det['amplitude']:.5f}")
            
            # Play the note
            play_note(freq, volume=min(1.0, det['amplitude'] * 30))
            time.sleep(0.6)  # Wait for note to finish
    else:
        print(f"❌ No touches detected (threshold: {threshold})")
        print(f"   Try lowering the threshold or check if this is a control recording.")
    
    print(f"\n{'='*70}\n")


def main():
    """Main test program"""
    
    print("""
╔══════════════════════════════════════════════════════════════════════════╗
║                    🌱 PLANT PIANO - TEST MODE 🎹                         ║
║                    Test on your recorded files                           ║
╚══════════════════════════════════════════════════════════════════════════╝
    """)
    
    MODEL_PATH = "output/models/random_forest_model.pkl"
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ ERROR: Model not found at {MODEL_PATH}")
        print("Please run train_model.py first!")
        return
    
    print("Available test files:")
    print()
    
    # Look for WAV files
    data_dir = "Dataset/Holy_Basil"
    if os.path.exists(data_dir):
        wav_files = [f for f in os.listdir(data_dir) if f.endswith('.wav')]
        for i, f in enumerate(wav_files[:10], 1):
            print(f"  [{i}] {f}")
        print()
        
        choice = input("Enter file number (or full path): ").strip()
        
        if choice.isdigit() and 1 <= int(choice) <= len(wav_files):
            wav_file = os.path.join(data_dir, wav_files[int(choice) - 1])
        else:
            wav_file = choice
    else:
        wav_file = input("Enter path to WAV file: ").strip()
    
    if not os.path.exists(wav_file):
        print(f"❌ File not found: {wav_file}")
        return
    
    # Get threshold
    # Purpose: 
    print()
    threshold_input = input("Detection threshold (0.5-0.9, default 0.7): ").strip()
    threshold = float(threshold_input) if threshold_input else 0.7
    
    # Test!
    test_on_wav_file(MODEL_PATH, wav_file, threshold)


if __name__ == "__main__":
    main()
