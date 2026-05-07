#!/usr/bin/env python3
"""
Plant Piano - Live Inference Mode
Reads data directly from a connected SpikerBox and plays notes in real-time.
"""

import time
import joblib
import numpy as np
import serial
import serial.tools.list_ports

# --- Audio Setup ---
try:
    import pygame
    pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
    SOUND_ENGINE = 'pygame'
    print("✓ Using pygame for audio playback.")
except ImportError:
    try:
        import sounddevice as sd
        SOUND_ENGINE = 'sounddevice'
        print("✓ Using sounddevice for audio playback.")
    except ImportError:
        SOUND_ENGINE = None
        print("⚠️ No audio library found (pygame or sounddevice). Will run without sound.")

# --- Helper Functions from other scripts ---

def find_spikerbox():
    """Find Plant SpikerBox COM port."""
    ports = serial.tools.list_ports.comports()
    for port in ports:
        if "Plant" in port.description or "Interface 0" in port.description:
            return port.device
    return None

def extract_features(window):
    """Extract 5 simple features from a window."""
    return {
        'rms': np.sqrt(np.mean(window**2)),
        'peak_to_peak': np.ptp(window),
        'std': np.std(window),
        'percentile_90': np.percentile(np.abs(window), 90),
        'mean_abs': np.mean(np.abs(window))
    }

def generate_note_sound(frequency, duration=0.5):
    """Generate a piano-like note."""
    sample_rate = 44100
    t = np.linspace(0, duration, int(sample_rate * duration))
    fundamental = np.sin(2 * np.pi * frequency * t)
    harmonic2 = 0.5 * np.sin(2 * np.pi * frequency * 2 * t)
    harmonic3 = 0.25 * np.sin(2 * np.pi * frequency * 3 * t)
    wave = (fundamental + harmonic2 + harmonic3) * np.exp(-3 * t)
    wave /= np.max(np.abs(wave))
    if SOUND_ENGINE == 'pygame':
        return np.column_stack((wave, wave))
    return wave

def play_note(frequency, volume=1.0):
    """Play a note."""
    if SOUND_ENGINE is None:
        return
    wave = generate_note_sound(frequency) * volume
    try:
        if SOUND_ENGINE == 'pygame':
            sound = pygame.sndarray.make_sound((wave * 32767).astype(np.int16))
            sound.play()
        else:
            sd.play(wave, 44100)
    except Exception as e:
        print(f"Audio error: {e}")

# --- Main Inference Function ---

def run_live_inference():
    """Finds SpikerBox, loads model, and runs real-time classification."""
    
    # --- 1. Configuration ---
    MODEL_PATH = "output/models/random_forest_model.pkl"
    THRESHOLD = 0.7  # Confidence threshold for touch detection
    WINDOW_SECONDS = 5.0
    HOP_SECONDS = 0.5  # How often to run a new prediction
    SAMPLE_RATE = 10000 # Must match the rate used for training
    COOLDOWN_SECONDS = 2.0 # Wait time between playing notes

    # --- 2. Load Model ---
    print(f"Loading model: {MODEL_PATH}")
    try:
        model_data = joblib.load(MODEL_PATH)
        model = model_data['model']
        scaler = model_data['scaler']
        feature_names = model_data['feature_names']
    except FileNotFoundError:
        print(f"❌ ERROR: Model not found at {MODEL_PATH}. Please run train_model.py first.")
        return

    # --- 3. Find and Open SpikerBox ---
    print("Searching for Plant SpikerBox...")
    port = find_spikerbox()
    if not port:
        print("❌ ERROR: Plant SpikerBox not found. Please check connection.")
        return
    print(f"✓ Found Plant SpikerBox on {port}. Opening connection...")
    
    try:
        ser = serial.Serial(port, baudrate=230400, timeout=1)
    except serial.SerialException as e:
        print(f"❌ ERROR: Could not open port {port}. Is another program using it? Details: {e}")
        return

    # --- 4. Real-time Loop ---
    print("
" + "="*50)
    print("LIVE INFERENCE RUNNING (Ctrl+C to stop)")
    print("="*50 + "
")

    byte_buffer = b''
    audio_buffer = np.array([], dtype=np.float32)
    window_samples = int(WINDOW_SECONDS * SAMPLE_RATE)
    
    last_prediction_time = 0
    last_note_time = 0

    try:
        while True:
            # Read data from serial and add to byte buffer
            if ser.in_waiting > 0:
                byte_buffer += ser.read(ser.in_waiting)
            
            # Process complete 2-byte chunks from the buffer
            num_samples_in_buffer = len(byte_buffer) // 2
            if num_samples_in_buffer > 0:
                bytes_to_process = num_samples_in_buffer * 2
                process_data = byte_buffer[:bytes_to_process]
                byte_buffer = byte_buffer[bytes_to_process:]
                
                # Convert to float and append to audio buffer
                samples = np.frombuffer(process_data, dtype=np.int16)
                new_audio = samples.astype(np.float32) / 32768.0
                audio_buffer = np.append(audio_buffer, new_audio)

            # Keep audio buffer from growing indefinitely
            # Keep slightly more than a window's worth of data
            if len(audio_buffer) > window_samples * 1.5:
                audio_buffer = audio_buffer[-window_samples:]

            # --- 5. Run Prediction periodically ---
            current_time = time.time()
            if current_time - last_prediction_time > HOP_SECONDS and len(audio_buffer) >= window_samples:
                last_prediction_time = current_time
                
                # Get the latest window of audio
                window = audio_buffer[-window_samples:]
                
                # a. Extract features
                features = extract_features(window)
                
                # b. Scale features
                X = np.array([[features[f] for f in feature_names]])
                X_scaled = scaler.transform(X)
                
                # c. Predict
                touch_prob = model.predict_proba(X_scaled)[0, 1]
                
                # d. Make decision and act
                if touch_prob > THRESHOLD and (current_time - last_note_time) > COOLDOWN_SECONDS:
                    last_note_time = current_time
                    
                    # Continuous frequency mapping
                    MIN_AMP, MAX_AMP = 0.005, 0.060
                    MIN_FREQ, MAX_FREQ = 261.63, 493.88 # C4 to B4
                    clamped_amp = np.clip(features['peak_to_peak'], MIN_AMP, MAX_AMP)
                    frequency = np.interp(clamped_amp, [MIN_AMP, MAX_AMP], [MIN_FREQ, MAX_FREQ])

                    print(f"TOUCH DETECTED! Prob: {touch_prob:.1%} | Amp: {features['peak_to_peak']:.5f} | Freq: {frequency:.0f} Hz")
                    play_note(frequency, volume=min(1.0, features['peak_to_peak'] * 30))
                else:
                    # Optional: print status when no touch is detected
                    print(f"Status: No Touch | Prob: {touch_prob:.1%}          ", end='')

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("
Stopping live inference.")
    finally:
        ser.close()
        print("Serial port closed.")

if __name__ == "__main__":
    run_live_inference()
