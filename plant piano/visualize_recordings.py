#!/usr/bin/env python3
"""
Visualization Script: Inspect All Recordings
Creates 3 plots showing all touch recordings grouped by timing
Plus 1 plot for all control recordings
"""

import os
import numpy as np
import pandas as pd
from scipy.io import wavfile
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def load_and_preprocess(filepath):
    """Load WAV file and do minimal preprocessing"""
    sr, audio = wavfile.read(filepath)
    
    # Convert to float
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483648.0
    
    # Handle stereo
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
    
    # Simple DC removal
    audio = audio - np.mean(audio)
    
    return sr, audio


def create_touch_group_plot(metadata_df, data_dir, group_name, output_file):
    """
    Create a plot with subplots for all recordings in a touch group
    """
    # Filter for this group
    group_df = metadata_df[metadata_df['touch_group'] == group_name].copy()
    
    if len(group_df) == 0:
        print(f"No recordings found for group: {group_name}")
        return
    
    n_recordings = len(group_df)
    
    # Create figure with subplots
    n_cols = 2
    n_rows = (n_recordings + n_cols - 1) // n_cols
    
    fig = plt.figure(figsize=(16, 4 * n_rows))
    fig.suptitle(f'Touch Group: {group_name.upper()} (N={n_recordings})', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    for idx, (_, row) in enumerate(group_df.iterrows()):
        filepath = os.path.join(data_dir, row['filename'])
        
        try:
            sr, audio = load_and_preprocess(filepath)
            time = np.arange(len(audio)) / sr
            
            # Create subplot
            ax = plt.subplot(n_rows, n_cols, idx + 1)
            
            # Plot signal
            ax.plot(time, audio, linewidth=0.5, alpha=0.8, color='#2E86AB')
            
            # Mark touch time
            if pd.notna(row['touch_time_sec']):
                touch_time = row['touch_time_sec']
                ax.axvline(touch_time, color='red', linestyle='--', 
                          linewidth=2, alpha=0.7, label=f'Touch at {touch_time}s')
                
                # Add shaded region around touch (±3 seconds)
                ax.axvspan(max(0, touch_time - 3), 
                          min(time[-1], touch_time + 3),
                          alpha=0.15, color='red')
            
            # Calculate signal statistics
            std_before = np.std(audio[:int(touch_time * sr)]) if pd.notna(row['touch_time_sec']) else np.std(audio)
            std_after = np.std(audio[int(touch_time * sr):]) if pd.notna(row['touch_time_sec']) else 0
            peak_to_peak = np.ptp(audio)
            
            # Title with stats
            title = f"{row['filename']}\n"
            title += f"Touch: {touch_time:.1f}s | " if pd.notna(row['touch_time_sec']) else ""
            title += f"P2P: {peak_to_peak:.5f} | Std: {np.std(audio):.5f}"
            ax.set_title(title, fontsize=9, fontweight='bold')
            
            ax.set_xlabel('Time (seconds)', fontsize=8)
            ax.set_ylabel('Amplitude', fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7, loc='upper right')
            
            # Zoom y-axis to show details better
            y_range = np.ptp(audio)
            y_center = np.median(audio)
            ax.set_ylim(y_center - y_range * 0.6, y_center + y_range * 0.6)
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Error loading\n{row["filename"]}\n{str(e)}',
                   ha='center', va='center', fontsize=10, color='red')
            ax.set_xticks([])
            ax.set_yticks([])
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()


def create_control_plot(metadata_df, data_dir, output_file):
    """
    Create a plot with subplots for all control recordings
    """
    # Filter for controls
    control_df = metadata_df[metadata_df['label'] == 0].copy()
    
    if len(control_df) == 0:
        print("No control recordings found!")
        return
    
    n_recordings = len(control_df)
    
    # Create figure
    n_cols = 3
    n_rows = (n_recordings + n_cols - 1) // n_cols
    
    fig = plt.figure(figsize=(16, 3 * n_rows))
    fig.suptitle(f'Control Recordings (No Touch) (N={n_recordings})', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Calculate statistics for all controls
    all_stds = []
    all_p2ps = []
    
    for idx, (_, row) in enumerate(control_df.iterrows()):
        filepath = os.path.join(data_dir, row['filename'])
        
        try:
            sr, audio = load_and_preprocess(filepath)
            time = np.arange(len(audio)) / sr
            
            # Create subplot
            ax = plt.subplot(n_rows, n_cols, idx + 1)
            
            # Plot signal
            ax.plot(time, audio, linewidth=0.5, alpha=0.8, color='#06A77D')
            
            # Calculate statistics
            std_val = np.std(audio)
            p2p_val = np.ptp(audio)
            all_stds.append(std_val)
            all_p2ps.append(p2p_val)
            
            # Title with stats
            title = f"{row['filename']}\n"
            title += f"P2P: {p2p_val:.5f} | Std: {std_val:.5f}"
            ax.set_title(title, fontsize=9, fontweight='bold')
            
            ax.set_xlabel('Time (seconds)', fontsize=8)
            ax.set_ylabel('Amplitude', fontsize=8)
            ax.grid(True, alpha=0.3)
            
            # Zoom y-axis
            y_range = p2p_val
            y_center = np.median(audio)
            ax.set_ylim(y_center - y_range * 0.6, y_center + y_range * 0.6)
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Error loading\n{row["filename"]}\n{str(e)}',
                   ha='center', va='center', fontsize=10, color='red')
            ax.set_xticks([])
            ax.set_yticks([])
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    print(f"\nControl Statistics:")
    print(f"  Mean Std: {np.mean(all_stds):.6f} (±{np.std(all_stds):.6f})")
    print(f"  Mean P2P: {np.mean(all_p2ps):.6f} (±{np.std(all_p2ps):.6f})")
    plt.close()


def create_touch_comparison_plot(metadata_df, data_dir, output_file):
    """
    Create overlay comparison: touch vs control signals
    """
    touch_df = metadata_df[metadata_df['label'] == 1].copy()
    control_df = metadata_df[metadata_df['label'] == 0].copy()
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Signal Comparison: Touch vs Control', fontsize=16, fontweight='bold')
    
    # Panel 1: Sample touch recordings
    ax = axes[0, 0]
    for idx, (_, row) in enumerate(touch_df.head(5).iterrows()):
        filepath = os.path.join(data_dir, row['filename'])
        try:
            sr, audio = load_and_preprocess(filepath)
            time = np.arange(len(audio)) / sr
            ax.plot(time, audio + idx * 0.002, linewidth=0.5, alpha=0.7, 
                   label=f"{row['filename'][:10]}")
        except:
            pass
    ax.set_title('Sample Touch Recordings (Offset for Visibility)', fontweight='bold')
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Amplitude (with offset)')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    
    # Panel 2: Sample control recordings
    ax = axes[0, 1]
    for idx, (_, row) in enumerate(control_df.head(5).iterrows()):
        filepath = os.path.join(data_dir, row['filename'])
        try:
            sr, audio = load_and_preprocess(filepath)
            time = np.arange(len(audio)) / sr
            ax.plot(time, audio + idx * 0.002, linewidth=0.5, alpha=0.7,
                   label=f"{row['filename'][:12]}")
        except:
            pass
    ax.set_title('Sample Control Recordings (Offset for Visibility)', fontweight='bold')
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Amplitude (with offset)')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    
    # Panel 3: Statistics distribution
    ax = axes[1, 0]
    
    touch_stds = []
    touch_p2ps = []
    control_stds = []
    control_p2ps = []
    
    for _, row in touch_df.iterrows():
        filepath = os.path.join(data_dir, row['filename'])
        try:
            sr, audio = load_and_preprocess(filepath)
            touch_stds.append(np.std(audio))
            touch_p2ps.append(np.ptp(audio))
        except:
            pass
    
    for _, row in control_df.iterrows():
        filepath = os.path.join(data_dir, row['filename'])
        try:
            sr, audio = load_and_preprocess(filepath)
            control_stds.append(np.std(audio))
            control_p2ps.append(np.ptp(audio))
        except:
            pass
    
    positions = [1, 2]
    bp1 = ax.boxplot([control_stds, touch_stds], positions=positions,
                      labels=['Control', 'Touch'],
                      patch_artist=True, widths=0.6)
    bp1['boxes'][0].set_facecolor('#06A77D')
    bp1['boxes'][1].set_facecolor('#2E86AB')
    
    ax.set_title('Standard Deviation: Control vs Touch', fontweight='bold')
    ax.set_ylabel('Standard Deviation')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Panel 4: Peak-to-Peak comparison
    ax = axes[1, 1]
    bp2 = ax.boxplot([control_p2ps, touch_p2ps], positions=positions,
                      labels=['Control', 'Touch'],
                      patch_artist=True, widths=0.6)
    bp2['boxes'][0].set_facecolor('#06A77D')
    bp2['boxes'][1].set_facecolor('#2E86AB')
    
    ax.set_title('Peak-to-Peak: Control vs Touch', fontweight='bold')
    ax.set_ylabel('Peak-to-Peak Amplitude')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    
    # Print statistics
    print(f"\n{'='*60}")
    print("SIGNAL STATISTICS COMPARISON")
    print(f"{'='*60}")
    print(f"\nControl Recordings (N={len(control_stds)}):")
    print(f"  Std Dev: {np.mean(control_stds):.6f} ± {np.std(control_stds):.6f}")
    print(f"  P2P Amp: {np.mean(control_p2ps):.6f} ± {np.std(control_p2ps):.6f}")
    print(f"\nTouch Recordings (N={len(touch_stds)}):")
    print(f"  Std Dev: {np.mean(touch_stds):.6f} ± {np.std(touch_stds):.6f}")
    print(f"  P2P Amp: {np.mean(touch_p2ps):.6f} ± {np.std(touch_p2ps):.6f}")
    print(f"\nRatio (Touch/Control):")
    print(f"  Std Dev: {np.mean(touch_stds)/np.mean(control_stds):.2f}x")
    print(f"  P2P Amp: {np.mean(touch_p2ps)/np.mean(control_p2ps):.2f}x")
    print(f"{'='*60}\n")
    
    plt.close()


def main():
    """Main execution"""
    print("""
╔══════════════════════════════════════════════════════════════════════════╗
║               RECORDING VISUALIZATION TOOL                               ║
║          Inspect all your plant recordings grouped by type               ║
╚══════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Configuration
    DATA_DIR = r"D:\BYB Plant Spikerbox\plant piano\dataset\holy basil"
    METADATA_FILE = r"D:\BYB Plant Spikerbox\plant piano\dataset\holy basil\metadata.csv"
    OUTPUT_DIR = r"output/signal_inspection"
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load metadata
    if not os.path.exists(METADATA_FILE):
        print(f"❌ ERROR: Metadata file not found: {METADATA_FILE}")
        print("Please create metadata.csv with your recording information.")
        return
    
    metadata_df = pd.read_csv(METADATA_FILE)
    print(f"\n✓ Loaded metadata: {len(metadata_df)} recordings")
    print(f"  • Control: {sum(metadata_df['label'] == 0)}")
    print(f"  • Touch: {sum(metadata_df['label'] == 1)}")
    
    # Check for touch groups
    if 'touch_group' in metadata_df.columns:
        groups = metadata_df[metadata_df['label'] == 1]['touch_group'].unique()
        print(f"  • Touch groups: {', '.join([str(g) for g in groups if pd.notna(g)])}")
    
    print(f"\nGenerating visualizations...\n")
    
    # Create control plot
    print("1/4: Creating control recordings plot...")
    create_control_plot(metadata_df, DATA_DIR, 
                       f"{OUTPUT_DIR}/01_control_recordings.png")
    
    # Create touch group plots
    if 'touch_group' in metadata_df.columns:
        touch_groups = ['early', 'mid_early', 'mid_late']
        
        for idx, group in enumerate(touch_groups, start=2):
            group_recordings = metadata_df[metadata_df['touch_group'] == group]
            if len(group_recordings) > 0:
                print(f"{idx}/4: Creating {group} touch group plot...")
                create_touch_group_plot(metadata_df, DATA_DIR, group,
                                       f"{OUTPUT_DIR}/0{idx}_{group}_touch.png")
    
    # Create comparison plot
    print("4/4: Creating touch vs control comparison...")
    create_touch_comparison_plot(metadata_df, DATA_DIR,
                                f"{OUTPUT_DIR}/04_comparison.png")
    
    print(f"\n{'='*70}")
    print("✅ VISUALIZATION COMPLETE!")
    print(f"{'='*70}")
    print(f"\n📁 All plots saved to: {OUTPUT_DIR}/")
    print(f"\nGenerated files:")
    print(f"  • 01_control_recordings.png - All control signals")
    print(f"  • 02_early_touch.png - Early touch group")
    print(f"  • 03_mid_early_touch.png - Mid-early touch group")
    print(f"  • 04_comparison.png - Statistical comparison")
    print(f"\n💡 Open these files to inspect your recordings!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
