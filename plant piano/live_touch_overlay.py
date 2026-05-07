#!/usr/bin/env python3
"""
Plant Touch Detector - Real-Time Overlay
Works alongside Spike Recorder to show live touch detection
"""

import sys
import numpy as np
import joblib
import serial
import serial.tools.list_ports
from collections import deque
import time
import threading
from datetime import datetime

# Try to import GUI library
try:
    from tkinter import Tk, Label, Button, Frame, StringVar, IntVar
    from tkinter import ttk
    GUI_AVAILABLE = True
except:
    print("⚠️ tkinter not available, using terminal mode only")
    GUI_AVAILABLE = False

# Try to import audio
try:
    import pygame
    pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
    AUDIO_AVAILABLE = True
except:
    print("⚠️ pygame not available, no audio playback")
    AUDIO_AVAILABLE = False


class PlantTouchDetector:
    """Real-time plant touch detector with overlay display"""
    
    def __init__(self, model_path='output/models/random_forest_model.pkl', 
                 threshold=0.7, cooldown=1.0):
        
        self.threshold = threshold
        self.cooldown = cooldown
        
        # Load model
        print("Loading model...")
        model_data = joblib.load(model_path)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        print(f"✓ Model loaded: {model_path}")
        
        # Detection state
        self.running = False
        self.serial_port = None
        self.buffer = deque(maxlen=50000)  # 5 seconds at 10kHz
        self.last_detection_time = 0
        self.total_detections = 0
        self.session_start = None
        
        # Latest detection info
        self.last_note = None
        self.last_amplitude = 0
        self.last_probability = 0
        self.current_amplitude = 0
        
        # Musical notes
        self.notes = {
            'C3': 130.81, 'D3': 146.83, 'E3': 164.81, 'G3': 196.00, 'A3': 220.00,
            'C4': 261.63, 'D4': 293.66, 'E4': 329.63, 'G4': 392.00, 'A4': 440.00,
            'C5': 523.25
        }
        
        # Generate sounds if available
        if AUDIO_AVAILABLE:
            self._generate_sounds()
    
    def _generate_sounds(self):
        """Generate piano-like sounds for each note"""
        self.note_sounds = {}
        sample_rate = 44100
        duration = 0.5
        
        for note_name, frequency in self.notes.items():
            t = np.linspace(0, duration, int(sample_rate * duration))
            
            # Create harmonics
            wave1 = np.sin(2 * np.pi * frequency * t)
            wave2 = 0.3 * np.sin(2 * np.pi * frequency * 2 * t)
            wave3 = 0.15 * np.sin(2 * np.pi * frequency * 3 * t)
            wave = wave1 + wave2 + wave3
            
            # Envelope
            envelope = np.exp(-3 * t)
            wave = wave * envelope
            wave = wave / np.max(np.abs(wave))
            
            # Convert to stereo int16
            wave_stereo = np.column_stack((wave, wave))
            wave_int = (wave_stereo * 0.2 * 32767).astype(np.int16)
            
            self.note_sounds[note_name] = pygame.sndarray.make_sound(wave_int)
    
    def find_spikerbox(self):
        """Find Plant SpikerBox COM port"""
        ports = serial.tools.list_ports.comports()
        
        for port in ports:
            if "Plant" in port.description or "Interface 0" in port.description:
                return port.device
        
        return None
    
    def extract_features(self, window):
        """Extract features from audio window"""
        features = {}
        features['rms'] = np.sqrt(np.mean(window**2))
        features['peak_to_peak'] = np.ptp(window)
        features['std'] = np.std(window)
        features['percentile_90'] = np.percentile(np.abs(window), 90)
        features['mean_abs'] = np.mean(np.abs(window))
        return features
    
    def amplitude_to_note(self, amplitude):
        """Map amplitude to musical note"""
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
    
    def play_note(self, note):
        """Play a musical note"""
        if AUDIO_AVAILABLE and note in self.note_sounds:
            self.note_sounds[note].play()
    
    def detection_loop(self):
        """Main detection loop - runs in background thread"""
        
        port = self.find_spikerbox()
        if not port:
            print("❌ Plant SpikerBox not found!")
            self.running = False
            return
        
        try:
            # Open serial connection
            self.serial_port = serial.Serial(
                port=port,
                baudrate=230400,
                timeout=0.1
            )
            
            print(f"✓ Connected to {port}")
            print("🟢 Detection active - touch your plant!")
            
            byte_buffer = b''
            
            while self.running:
                # Read available data
                if self.serial_port.in_waiting > 0:
                    byte_buffer += self.serial_port.read(self.serial_port.in_waiting)
                    
                    # Process complete samples
                    num_samples = len(byte_buffer) // 2
                    if num_samples > 0:
                        bytes_to_process = num_samples * 2
                        process_data = byte_buffer[:bytes_to_process]
                        byte_buffer = byte_buffer[bytes_to_process:]
                        
                        # Convert to float
                        samples = np.frombuffer(process_data, dtype=np.int16)
                        audio = samples.astype(np.float32) / 32768.0
                        
                        # Add to buffer
                        self.buffer.extend(audio)
                        
                        # Update current amplitude for display
                        if len(audio) > 0:
                            self.current_amplitude = np.mean(np.abs(audio))
                        
                        # Process when buffer is full
                        if len(self.buffer) == 50000:
                            window = np.array(self.buffer)
                            window = window - np.mean(window)  # DC removal
                            
                            # Extract features
                            features = self.extract_features(window)
                            
                            # Predict
                            X = np.array([[features[f] for f in self.feature_names]])
                            X_scaled = self.scaler.transform(X)
                            probability = self.model.predict_proba(X_scaled)[0, 1]
                            
                            # Check for touch
                            current_time = time.time()
                            if probability > self.threshold and \
                               (current_time - self.last_detection_time) > self.cooldown:
                                
                                # TOUCH DETECTED!
                                self.last_detection_time = current_time
                                self.total_detections += 1
                                
                                amplitude = features['peak_to_peak']
                                note = self.amplitude_to_note(amplitude)
                                
                                self.last_note = note
                                self.last_amplitude = amplitude
                                self.last_probability = probability
                                
                                # Log to terminal
                                timestamp = datetime.now().strftime("%H:%M:%S")
                                print(f"\n🎵 [{timestamp}] TOUCH! {note} ({self.notes[note]:.0f} Hz)")
                                print(f"   Amplitude: {amplitude:.5f} | Confidence: {probability:.1%}")
                                
                                # Play sound
                                self.play_note(note)
                
                time.sleep(0.01)
        
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            if self.serial_port and self.serial_port.is_open:
                self.serial_port.close()
                print("\n✓ Serial port closed")
    
    def start(self):
        """Start detection"""
        if self.running:
            print("Already running!")
            return
        
        self.running = True
        self.session_start = datetime.now()
        self.total_detections = 0
        
        # Start detection thread
        self.thread = threading.Thread(target=self.detection_loop, daemon=True)
        self.thread.start()
    
    def stop(self):
        """Stop detection"""
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=2)
        
        # Print session stats
        if self.session_start:
            duration = (datetime.now() - self.session_start).total_seconds()
            print(f"\n{'='*60}")
            print("SESSION SUMMARY")
            print(f"{'='*60}")
            print(f"Duration: {duration:.1f} seconds")
            print(f"Total touches detected: {self.total_detections}")
            if duration > 0:
                print(f"Detection rate: {self.total_detections / (duration / 60):.1f} touches/minute")
            print(f"{'='*60}\n")


class OverlayGUI:
    """Overlay window showing detection status"""
    
    def __init__(self, detector):
        self.detector = detector
        
        self.root = Tk()
        self.root.title("🌱 Plant Touch Detector")
        self.root.geometry("500x350")
        self.root.configure(bg='#2C3E50')
        
        # Make window stay on top
        self.root.attributes('-topmost', True)
        
        # Status variables
        self.status_text = StringVar(value="⚪ Stopped")
        self.detections_text = StringVar(value="Touches: 0")
        self.last_touch_text = StringVar(value="Last: None")
        self.amplitude_text = StringVar(value="Signal: 0.000000")
        
        self._create_widgets()
        self._update_display()
    
    def _create_widgets(self):
        """Create GUI widgets"""
        
        # Header
        header = Label(
            self.root,
            text="🌱 Plant Touch Detector 🎹",
            font=("Arial", 20, "bold"),
            bg='#2ECC71',
            fg='white',
            pady=15
        )
        header.pack(fill='x')
        
        # Status display
        status_frame = Frame(self.root, bg='#34495E', padx=20, py=20)
        status_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Status
        status_label = Label(
            status_frame,
            textvariable=self.status_text,
            font=("Arial", 18, "bold"),
            bg='#34495E',
            fg='white'
        )
        status_label.pack(pady=5)
        
        # Detections count
        det_label = Label(
            status_frame,
            textvariable=self.detections_text,
            font=("Arial", 14),
            bg='#34495E',
            fg='#ECF0F1'
        )
        det_label.pack(pady=5)
        
        # Last touch
        touch_label = Label(
            status_frame,
            textvariable=self.last_touch_text,
            font=("Arial", 14),
            bg='#34495E',
            fg='#ECF0F1'
        )
        touch_label.pack(pady=5)
        
        # Current amplitude
        amp_label = Label(
            status_frame,
            textvariable=self.amplitude_text,
            font=("Arial", 12),
            bg='#34495E',
            fg='#95A5A6'
        )
        amp_label.pack(pady=5)
        
        # Buttons
        button_frame = Frame(self.root, bg='#2C3E50')
        button_frame.pack(fill='x', padx=10, pady=10)
        
        self.start_btn = Button(
            button_frame,
            text="▶️ Start",
            command=self.start_detection,
            font=("Arial", 12, "bold"),
            bg='#27AE60',
            fg='white',
            padx=20,
            pady=10,
            relief='raised',
            bd=3
        )
        self.start_btn.pack(side='left', padx=5)
        
        self.stop_btn = Button(
            button_frame,
            text="⏸️ Stop",
            command=self.stop_detection,
            font=("Arial", 12, "bold"),
            bg='#E74C3C',
            fg='white',
            padx=20,
            pady=10,
            relief='raised',
            bd=3,
            state='disabled'
        )
        self.stop_btn.pack(side='left', padx=5)
        
        # Info
        info = Label(
            self.root,
            text="Keep this window visible while using Spike Recorder",
            font=("Arial", 9),
            bg='#2C3E50',
            fg='#95A5A6'
        )
        info.pack(pady=5)
    
    def start_detection(self):
        """Start detection"""
        self.detector.start()
        self.status_text.set("🟢 Monitoring...")
        self.start_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
    
    def stop_detection(self):
        """Stop detection"""
        self.detector.stop()
        self.status_text.set("⚪ Stopped")
        self.start_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
    
    def _update_display(self):
        """Update display with latest detection info"""
        
        if self.detector.running:
            # Update detection count
            self.detections_text.set(f"Touches: {self.detector.total_detections}")
            
            # Update last touch
            if self.detector.last_note:
                self.last_touch_text.set(
                    f"Last: {self.detector.last_note} "
                    f"({self.detector.last_probability:.0%} confidence)"
                )
            
            # Update current amplitude
            self.amplitude_text.set(f"Signal: {self.detector.current_amplitude:.6f}")
        
        # Schedule next update
        self.root.after(100, self._update_display)
    
    def run(self):
        """Run the GUI"""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()
    
    def on_closing(self):
        """Handle window closing"""
        if self.detector.running:
            self.detector.stop()
        self.root.destroy()


def main():
    """Main entry point"""
    
    print("""
╔══════════════════════════════════════════════════════════════════════════╗
║           🌱 PLANT TOUCH DETECTOR - REAL-TIME OVERLAY 🎹                 ║
║                                                                          ║
║  This app works ALONGSIDE Spike Recorder:                               ║
║    1. Open Spike Recorder and start viewing your plant signal           ║
║    2. Run this app to see real-time touch detection                     ║
║    3. Touch your plant and watch both!                                  ║
╚══════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Check for model
    model_path = 'output/models/random_forest_model.pkl'
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print("Please run train_model.py first!")
        return
    
    # Create detector
    detector = PlantTouchDetector(
        model_path=model_path,
        threshold=0.7,
        cooldown=1.0
    )
    
    if GUI_AVAILABLE:
        # Run GUI version
        print("\n✓ Starting GUI overlay...")
        print("📌 Position the overlay window where you can see it alongside Spike Recorder\n")
        
        gui = OverlayGUI(detector)
        gui.run()
    else:
        # Terminal version
        print("\n✓ Running in terminal mode (no GUI available)")
        print("Press Ctrl+C to stop\n")
        
        detector.start()
        
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n\n🛑 Stopping...")
            detector.stop()


if __name__ == "__main__":
    import os
    main()
