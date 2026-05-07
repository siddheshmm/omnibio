#!/usr/bin/env python3
"""
Plant Touch Detection - Model Training Script
Window-based approach with simple features
"""

import os
import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy import signal
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (classification_report, confusion_matrix, 
                             accuracy_score, f1_score, roc_auc_score,
                             roc_curve, precision_recall_curve)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')


class PlantTouchClassifier:
    """Simple window-based touch classifier"""
    
    def __init__(self, window_size=5.0):
        """
        Parameters:
        -----------
        window_size : float
            Window size in seconds (default: 5.0)
        """
        self.window_size = window_size
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names = ['rms', 'peak_to_peak', 'std', 'percentile_90', 'mean_abs']
        
    def load_wav(self, filepath):
        """Load and preprocess WAV file"""
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
    
    def extract_features(self, window):
        """
        Extract 5 simple features from a window
        
        Parameters:
        -----------
        window : np.ndarray
            Signal window
            
        Returns:
        --------
        features : dict
            Dictionary of features
        """
        features = {}
        
        # 1. RMS (Root Mean Square) - signal energy
        features['rms'] = np.sqrt(np.mean(window**2))
        
        # 2. Peak-to-Peak - amplitude range
        features['peak_to_peak'] = np.ptp(window)
        
        # 3. Standard Deviation - variability
        features['std'] = np.std(window)
        
        # 4. 90th Percentile - upper amplitude distribution
        features['percentile_90'] = np.percentile(np.abs(window), 90)
        
        # 5. Mean Absolute Value - average magnitude
        features['mean_abs'] = np.mean(np.abs(window))
        
        return features
    
    def extract_windows_from_recording(self, filepath, touch_time=None):
        """
        Extract windows from a recording
        
        Parameters:
        -----------
        filepath : str
            Path to WAV file
        touch_time : float or None
            Time of touch event (None for control)
            
        Returns:
        --------
        windows : list of dict
            List of windows with features and labels
        """
        sr, audio = self.load_wav(filepath)
        window_samples = int(self.window_size * sr)
        windows = []
        
        if touch_time is not None:
            # Touch recording: extract 3 windows
            
            # Window 1: Touch window (centered on touch)
            touch_idx = int(touch_time * sr)
            start_idx = max(0, touch_idx - window_samples // 2)
            end_idx = min(len(audio), start_idx + window_samples)
            
            if end_idx - start_idx == window_samples:
                touch_window = audio[start_idx:end_idx]
                features = self.extract_features(touch_window)
                features['label'] = 1  # Touch
                features['type'] = 'touch'
                features['filename'] = os.path.basename(filepath)
                windows.append(features)
            
            # Window 2: Baseline (well before touch)
            baseline_time = max(5.0, touch_time - 10.0)
            baseline_idx = int(baseline_time * sr)
            start_idx = max(0, baseline_idx - window_samples // 2)
            end_idx = min(len(audio), start_idx + window_samples)
            
            if end_idx - start_idx == window_samples:
                baseline_window = audio[start_idx:end_idx]
                features = self.extract_features(baseline_window)
                features['label'] = 0  # No touch
                features['type'] = 'baseline'
                features['filename'] = os.path.basename(filepath)
                windows.append(features)
            
        else:
            # Control recording: extract 2-3 random windows
            n_windows = 3
            duration = len(audio) / sr
            
            for i in range(n_windows):
                # Random start time (avoid edges)
                max_start_time = duration - self.window_size - 5
                if max_start_time > 10:
                    start_time = np.random.uniform(5, max_start_time)
                    start_idx = int(start_time * sr)
                    end_idx = start_idx + window_samples
                    
                    if end_idx <= len(audio):
                        control_window = audio[start_idx:end_idx]
                        features = self.extract_features(control_window)
                        features['label'] = 0  # No touch
                        features['type'] = 'control'
                        features['filename'] = os.path.basename(filepath)
                        windows.append(features)
        
        return windows
    
    def prepare_dataset(self, metadata_df, data_dir):
        """
        Prepare dataset from metadata
        
        Parameters:
        -----------
        metadata_df : pd.DataFrame
            Metadata with recording info
        data_dir : str
            Directory containing WAV files
            
        Returns:
        --------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Labels
        info_df : pd.DataFrame
            Window information
        """
        all_windows = []
        
        print("Extracting windows and features...")
        for idx, row in metadata_df.iterrows():
            filepath = os.path.join(data_dir, row['filename'])
            
            if not os.path.exists(filepath):
                print(f"⚠️  Skipping {row['filename']} - file not found")
                continue
            
            touch_time = row['touch_time_sec'] if pd.notna(row['touch_time_sec']) else None
            windows = self.extract_windows_from_recording(filepath, touch_time)
            all_windows.extend(windows)
        
        # Convert to DataFrame
        windows_df = pd.DataFrame(all_windows)
        
        # Separate features and labels
        X = windows_df[self.feature_names].values
        y = windows_df['label'].values
        info_df = windows_df[['filename', 'type', 'label']].copy()
        
        print(f"\n✓ Extracted {len(windows_df)} windows")
        print(f"  • Touch windows: {sum(y == 1)}")
        print(f"  • No-touch windows: {sum(y == 0)}")
        print(f"  • Features: {len(self.feature_names)}")
        
        return X, y, info_df
    
    def train(self, X, y, model_type='random_forest'):
        """
        Train model
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Labels
        model_type : str
            Type of model: 'logistic', 'random_forest', 'gradient_boost'
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Create model
        if model_type == 'logistic':
            self.model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
        elif model_type == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=100, max_depth=10, 
                                               class_weight='balanced', random_state=42, n_jobs=-1)
        elif model_type == 'gradient_boost':
            self.model = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Train
        print(f"\nTraining {model_type.upper()} model...")
        self.model.fit(X_scaled, y)
        
        # Training accuracy
        y_pred = self.model.predict(X_scaled)
        train_acc = accuracy_score(y, y_pred)
        print(f"✓ Training accuracy: {train_acc:.4f}")
        
    def evaluate(self, X, y, info_df=None):
        """
        Evaluate model with cross-validation
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Labels
        info_df : pd.DataFrame
            Window information
            
        Returns:
        --------
        results : dict
            Evaluation results
        """
        X_scaled = self.scaler.transform(X)
        
        # Predictions
        y_pred = self.model.predict(X_scaled)
        y_proba = self.model.predict_proba(X_scaled)[:, 1]
        
        # Metrics
        acc = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        
        try:
            auc = roc_auc_score(y, y_proba)
        except:
            auc = 0.5
        
        print(f"\n{'='*60}")
        print("MODEL EVALUATION")
        print(f"{'='*60}")
        print(f"Accuracy: {acc:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"ROC-AUC:  {auc:.4f}")
        
        print(f"\nClassification Report:")
        print(classification_report(y, y_pred, target_names=['No Touch', 'Touch'], zero_division=0))
        
        cm = confusion_matrix(y, y_pred)
        print(f"\nConfusion Matrix:")
        print(f"                Predicted")
        print(f"               No Touch  Touch")
        print(f"Actual No Touch   {cm[0,0]:4d}    {cm[0,1]:4d}")
        print(f"       Touch      {cm[1,0]:4d}    {cm[1,1]:4d}")
        print(f"{'='*60}\n")
        
        return {
            'accuracy': acc,
            'f1_score': f1,
            'roc_auc': auc,
            'confusion_matrix': cm,
            'y_true': y,
            'y_pred': y_pred,
            'y_proba': y_proba
        }
    
    def plot_evaluation(self, results, output_file):
        """Create evaluation plots"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Confusion Matrix
        cm = results['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
                   xticklabels=['No Touch', 'Touch'],
                   yticklabels=['No Touch', 'Touch'])
        axes[0, 0].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('Actual')
        axes[0, 0].set_xlabel('Predicted')
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(results['y_true'], results['y_proba'])
        axes[0, 1].plot(fpr, tpr, linewidth=2, label=f'ROC (AUC={results["roc_auc"]:.3f})')
        axes[0, 1].plot([0, 1], [0, 1], 'k--', linewidth=1)
        axes[0, 1].set_xlabel('False Positive Rate')
        axes[0, 1].set_ylabel('True Positive Rate')
        axes[0, 1].set_title('ROC Curve', fontsize=14, fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(results['y_true'], results['y_proba'])
        axes[1, 0].plot(recall, precision, linewidth=2)
        axes[1, 0].set_xlabel('Recall')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Feature Importance (if available)
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            axes[1, 1].barh(range(len(importances)), importances[indices])
            axes[1, 1].set_yticks(range(len(importances)))
            axes[1, 1].set_yticklabels([self.feature_names[i] for i in indices])
            axes[1, 1].set_xlabel('Importance')
            axes[1, 1].set_title('Feature Importance', fontsize=14, fontweight='bold')
            axes[1, 1].invert_yaxis()
        elif hasattr(self.model, 'coef_'):
            coef = np.abs(self.model.coef_[0])
            indices = np.argsort(coef)[::-1]
            
            axes[1, 1].barh(range(len(coef)), coef[indices])
            axes[1, 1].set_yticks(range(len(coef)))
            axes[1, 1].set_yticklabels([self.feature_names[i] for i in indices])
            axes[1, 1].set_xlabel('Absolute Coefficient')
            axes[1, 1].set_title('Feature Coefficients', fontsize=14, fontweight='bold')
            axes[1, 1].invert_yaxis()
        else:
            axes[1, 1].text(0.5, 0.5, 'Feature importance\nnot available', 
                          ha='center', va='center', fontsize=12)
            axes[1, 1].set_xticks([])
            axes[1, 1].set_yticks([])
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Evaluation plots saved: {output_file}")
        plt.close()
    
    def save_model(self, filepath):
        """Save trained model"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'window_size': self.window_size
        }
        joblib.dump(model_data, filepath)
        print(f"✓ Model saved: {filepath}")
    
    def load_model(self, filepath):
        """Load trained model"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.window_size = model_data['window_size']
        print(f"✓ Model loaded: {filepath}")


def compare_models(X, y, info_df, output_dir):
    """
    Compare multiple models
    
    Parameters:
    -----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Labels
    info_df : pd.DataFrame
        Window information
    output_dir : str
        Output directory
    """
    models = [
        ('Logistic Regression', 'logistic'),
        ('Random Forest', 'random_forest'),
        ('Gradient Boosting', 'gradient_boost')
    ]
    
    results_summary = []
    
    for name, model_type in models:
        print(f"\n{'#'*60}")
        print(f"Training: {name}")
        print(f"{'#'*60}")
        
        # Create classifier
        classifier = PlantTouchClassifier()
        
        # Train
        classifier.train(X, y, model_type=model_type)
        
        # Evaluate
        results = classifier.evaluate(X, y, info_df)
        
        # Save plots
        plot_file = os.path.join(output_dir, f"{model_type}_evaluation.png")
        classifier.plot_evaluation(results, plot_file)
        
        # Save model
        model_file = os.path.join(output_dir, f"{model_type}_model.pkl")
        classifier.save_model(model_file)
        
        # Store results
        results_summary.append({
            'Model': name,
            'Accuracy': results['accuracy'],
            'F1-Score': results['f1_score'],
            'ROC-AUC': results['roc_auc']
        })
    
    # Print comparison
    print(f"\n{'='*60}")
    print("MODEL COMPARISON SUMMARY")
    print(f"{'='*60}\n")
    
    comparison_df = pd.DataFrame(results_summary)
    print(comparison_df.to_string(index=False))
    
    # Save comparison
    comparison_df.to_csv(os.path.join(output_dir, 'model_comparison.csv'), index=False)
    
    # Find best model
    best_idx = comparison_df['F1-Score'].idxmax()
    best_model = comparison_df.iloc[best_idx]
    
    print(f"\n{'='*60}")
    print(f"🏆 BEST MODEL: {best_model['Model']}")
    print(f"   Accuracy: {best_model['Accuracy']:.4f}")
    print(f"   F1-Score: {best_model['F1-Score']:.4f}")
    print(f"   ROC-AUC:  {best_model['ROC-AUC']:.4f}")
    print(f"{'='*60}\n")
    
    return comparison_df


def main():
    """Main training pipeline"""
    print("""
╔══════════════════════════════════════════════════════════════════════════╗
║               PLANT TOUCH DETECTION - MODEL TRAINING                     ║
╚══════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Configuration
    DATA_DIR = r"D:\BYB Plant Spikerbox\plant piano\dataset\holy basil"
    METADATA_FILE = r"D:\BYB Plant Spikerbox\plant piano\dataset\holy basil\metadata.csv"
    OUTPUT_DIR = r"output/models"
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load metadata
    if not os.path.exists(METADATA_FILE):
        print(f"❌ ERROR: Metadata file not found: {METADATA_FILE}")
        return
    
    metadata_df = pd.read_csv(METADATA_FILE)
    print(f"✓ Loaded metadata: {len(metadata_df)} recordings")
    print(f"  • Control: {sum(metadata_df['label'] == 0)}")
    print(f"  • Touch: {sum(metadata_df['label'] == 1)}\n")
    
    # Prepare dataset
    classifier = PlantTouchClassifier(window_size=5.0)
    X, y, info_df = classifier.prepare_dataset(metadata_df, DATA_DIR)
    
    # Print feature statistics
    print(f"\n{'='*60}")
    print("FEATURE STATISTICS")
    print(f"{'='*60}")
    
    feature_df = pd.DataFrame(X, columns=classifier.feature_names)
    feature_df['label'] = y
    
    print("\nTouch vs No-Touch Feature Comparison:")
    for feature in classifier.feature_names:
        touch_vals = feature_df[feature_df['label'] == 1][feature]
        no_touch_vals = feature_df[feature_df['label'] == 0][feature]
        
        ratio = touch_vals.mean() / (no_touch_vals.mean() + 1e-10)
        print(f"  {feature:15s}: Touch={touch_vals.mean():.6f}, NoTouch={no_touch_vals.mean():.6f}, Ratio={ratio:.2f}x")
    
    print(f"{'='*60}\n")
    
    # Compare models
    comparison_df = compare_models(X, y, info_df, OUTPUT_DIR)
    
    print(f"\n{'='*60}")
    print("🎉 TRAINING COMPLETE!")
    print(f"{'='*60}")
    print(f"\n📁 All outputs saved to: {OUTPUT_DIR}/")
    print(f"\n🚀 Next step: Build the Plant Piano with the best model!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
