#!/usr/bin/env python3
"""
SpikerBox Serial Reader
Reads real-time data from Plant SpikerBox via USB/COM port
"""

import serial
import serial.tools.list_ports
import numpy as np
import time

def find_spikerbox():
    """Find Plant SpikerBox COM port"""
    ports = serial.tools.list_ports.comports()
    
    for port in ports:
        # Look for "Plant SpikerBox" or "Interface 0"
        if "Plant" in port.description or "Interface 0" in port.description:
            return port.device
    
    return None

def test_spikerbox():
    """Test SpikerBox connection"""
    print("\n" + "="*70)
    print("SPIKERBOX CONNECTION TEST")
    print("="*70 + "\n")
    
    # Find SpikerBox
    port = find_spikerbox()
    
    if port:
        print(f"✓ Found Plant SpikerBox on {port}")
        print("\nAttempting to read data...")
        
        try:
            ser = serial.Serial(
                port=port,
                baudrate=230400,
                timeout=1
            )
            
            print("✓ Port opened successfully")
            print("Reading samples (Ctrl+C to stop)...\n")
            
            byte_buffer = b''
            sample_count = 0
            for i in range(100):  # Read 100 chunks
                if ser.in_waiting > 0:
                    # Append new data to our buffer
                    byte_buffer += ser.read(ser.in_waiting)
                    
                    # Find how many complete 2-byte samples we have
                    num_samples = len(byte_buffer) // 2
                    
                    if num_samples > 0:
                        # Determine the number of bytes to process
                        bytes_to_process = num_samples * 2
                        
                        # Get the data to process and the leftover bytes
                        process_data = byte_buffer[:bytes_to_process]
                        byte_buffer = byte_buffer[bytes_to_process:] # Keep the remainder
                        
                        # Convert the complete data to samples
                        samples = np.frombuffer(process_data, dtype=np.int16)
                        audio = samples.astype(np.float32) / 32768.0
                        
                        sample_count += len(audio)
                        
                        print(f"  Read {len(audio)} samples. Total: {sample_count}. Range: [{np.min(audio):.6f}, {np.max(audio):.6f}]")

                time.sleep(0.05) # Shorter sleep to be more responsive
            
            ser.close()
            print(f"\n✓ Test complete! Received {sample_count} samples")
            
        except Exception as e:
            print(f"❌ Error: {e}")
    else:
        print("❌ Plant SpikerBox not found!")
        print("\nAvailable ports:")
        for p in serial.tools.list_ports.comports():
            print(f"  {p.device}: {p.description}")

if __name__ == "__main__":
    test_spikerbox()
