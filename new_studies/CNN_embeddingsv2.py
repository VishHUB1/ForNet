import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, \
    recall_score, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP
import librosa
import librosa.display
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.utils import to_categorical
import pickle
import json
from datetime import datetime
import warnings
from collections import Counter
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings('ignore')


class ForNetAudioProcessor:
    """Audio preprocessing and feature extraction class"""

    def __init__(self, sample_rate=44100, n_mfcc=20, n_mels=128, hop_length=2048, n_fft=2048):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.n_fft = n_fft

    def load_audio(self, file_path, duration=None):
        """Load audio file with specified parameters"""
        try:
            audio, sr = librosa.load(file_path, sr=self.sample_rate, duration=duration)
            return audio
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None

    def get_audio_statistics(self, audio):
        """Extract basic audio statistics"""
        return {
            'duration': len(audio) / self.sample_rate,
            'rms_energy': np.sqrt(np.mean(audio ** 2)),
            'zero_crossing_rate': np.mean(librosa.feature.zero_crossing_rate(audio)),
            'spectral_centroid': np.mean(librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)),
            'spectral_rolloff': np.mean(librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate)),
            'spectral_bandwidth': np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=self.sample_rate))
        }

    def extract_mfcc(self, audio):
        """Extract MFCC features as per paper specifications"""
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            hop_length=self.hop_length,
            n_fft=self.n_fft
        )
        # Pad or truncate to ensure consistent shape (20, 65)
        target_width = 65
        if mfcc.shape[1] < target_width:
            mfcc = np.pad(mfcc, ((0, 0), (0, target_width - mfcc.shape[1])), mode='constant')
        else:
            mfcc = mfcc[:, :target_width]

        return mfcc

    def extract_mel_spectrogram(self, audio):
        """Extract Mel spectrogram features"""
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            hop_length=1034,  # As specified in paper
            n_fft=1034
        )
        # Target shape (128, 66)
        target_width = 66
        if mel_spec.shape[1] < target_width:
            mel_spec = np.pad(mel_spec, ((0, 0), (0, target_width - mel_spec.shape[1])), mode='constant')
        else:
            mel_spec = mel_spec[:, :target_width]

        return mel_spec

    def extract_log_mel_spectrogram(self, audio):
        """Extract Log-Mel spectrogram features"""
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            hop_length=2024,  # As specified in paper
            n_fft=2024
        )
        # Convert to log scale with small epsilon
        log_mel_spec = np.log(mel_spec + 1e-8)

        # Target shape (128, 128)
        target_width = 128
        if log_mel_spec.shape[1] < target_width:
            log_mel_spec = np.pad(log_mel_spec, ((0, 0), (0, target_width - log_mel_spec.shape[1])), mode='constant')
        else:
            log_mel_spec = log_mel_spec[:, :target_width]

        return log_mel_spec


class ForNetCNN:
    """Exact ForNet CNN architecture as described in the paper"""

    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.embedding_model = None
        self.intermediate_models = {}

    def build_model(self):
        """Build the exact CNN architecture from the paper using Functional API"""
        inputs = layers.Input(shape=self.input_shape)

        # First Convolutional Layer: 32 filters, 5x5 kernel
        conv1 = layers.Conv2D(32, (5, 5), activation='relu', padding='same', name='conv1')(inputs)
        pool1 = layers.MaxPooling2D((2, 2), name='pool1')(conv1)

        # Second Convolutional Layer: 64 filters, 4x4 kernel
        conv2 = layers.Conv2D(64, (4, 4), activation='relu', padding='same', name='conv2')(pool1)
        pool2 = layers.MaxPooling2D((2, 2), name='pool2')(conv2)

        # Third Convolutional Layer: 128 filters, 3x3 kernel
        conv3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3')(pool2)

        # Global Average Pooling (for embeddings extraction)
        embeddings = layers.GlobalAveragePooling2D(name='embeddings')(conv3)

        # First Dense Layer: 50 neurons
        dense1 = layers.Dense(50, activation='relu', name='dense1')(embeddings)
        dropout1 = layers.Dropout(0.5, name='dropout1')(dense1)

        # Output Layer
        outputs = layers.Dense(self.num_classes, activation='softmax', name='output')(dropout1)

        # Create the main model
        model = models.Model(inputs=inputs, outputs=outputs)

        # Compile with Adam optimizer and categorical crossentropy
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        self.model = model

        # Create intermediate models for layer visualization
        self.embedding_model = models.Model(inputs=inputs, outputs=embeddings)
        self.intermediate_models = {
            'conv1': models.Model(inputs=inputs, outputs=conv1),
            'pool1': models.Model(inputs=inputs, outputs=pool1),
            'conv2': models.Model(inputs=inputs, outputs=conv2),
            'pool2': models.Model(inputs=inputs, outputs=pool2),
            'conv3': models.Model(inputs=inputs, outputs=conv3),
            'dense1': models.Model(inputs=inputs, outputs=dense1)
        }

        return model

    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        """Train the model with early stopping"""
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )

        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping],
            verbose=1
        )

        return history

    def train_single_fold(self, X_train, y_train, epochs=100, batch_size=32):
        """Train the model for a single fold without validation split"""
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0
        )

        return history

    def extract_embeddings(self, X):
        """Extract embeddings from Global Average Pooling layer"""
        if self.embedding_model is None:
            raise ValueError("Embedding model must be created first (train the model)")
        return self.embedding_model.predict(X, verbose=0)

    def get_layer_activations(self, X, layer_name):
        """Get activations from intermediate layers"""
        if layer_name not in self.intermediate_models:
            raise ValueError(f"Layer {layer_name} not found. Available layers: {list(self.intermediate_models.keys())}")
        return self.intermediate_models[layer_name].predict(X, verbose=0)


class DataLoader:
    """Load and organize audio data from directory structure"""

    def __init__(self, source_dir, processor):
        self.source_dir = source_dir
        self.processor = processor
        self.class_names = []
        self.file_paths = []
        self.labels = []
        self.audio_stats = []

    def load_data(self):
        """Load all audio files from subdirectories"""
        print("Loading data from directory structure...")

        # Get class names from subdirectories
        self.class_names = [d for d in os.listdir(self.source_dir)
                            if os.path.isdir(os.path.join(self.source_dir, d))]
        self.class_names.sort()

        print(f"Found classes: {self.class_names}")

        # Load file paths and labels
        for class_idx, class_name in enumerate(self.class_names):
            class_dir = os.path.join(self.source_dir, class_name)
            audio_files = [f for f in os.listdir(class_dir)
                           if f.lower().endswith(('.wav', '.mp3', '.flac', '.ogg'))]

            # Limit to 450 samples for Car_Sound class if needed
            if class_name == "Car_Sound":
                audio_files = audio_files[:450]

            for audio_file in audio_files:
                file_path = os.path.join(class_dir, audio_file)
                self.file_paths.append(file_path)
                self.labels.append(class_idx)

                # Extract audio statistics
                audio = self.processor.load_audio(file_path, duration=5)
                if audio is not None:
                    stats = self.processor.get_audio_statistics(audio)
                    stats['class'] = class_name
                    stats['class_idx'] = class_idx
                    self.audio_stats.append(stats)

            print(f"Class '{class_name}': {len(audio_files)} files")

        print(f"Total files loaded: {len(self.file_paths)}")
        return self.file_paths, self.labels, self.class_names


class ResultsManager:
    """Manage results directory and save outputs"""

    def __init__(self, results_dir):
        self.results_dir = results_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.setup_directories()

    def setup_directories(self):
        """Create organized results directory structure"""
        subdirs = [
            "models",
            "embeddings",
            "metrics",
            "visualizations",
            "feature_representations",
            "logs",
            "dataset_analysis",
            "layer_analysis",
            "interactive_plots",
            "statistical_analysis"
        ]

        for subdir in subdirs:
            os.makedirs(os.path.join(self.results_dir, subdir), exist_ok=True)

    def save_embeddings(self, embeddings, labels, class_names, feature_type):
        """Save embeddings for future ensemble training"""
        embedding_data = {
            'embeddings': embeddings,
            'labels': labels,
            'class_names': class_names,
            'feature_type': feature_type,
            'timestamp': self.timestamp
        }

        filename = f"embeddings_{feature_type}_{self.timestamp}.pkl"
        filepath = os.path.join(self.results_dir, "embeddings", filename)

        with open(filepath, 'wb') as f:
            pickle.dump(embedding_data, f)

        print(f"Embeddings saved to: {filepath}")
        return filepath

    def save_model(self, model, feature_type):
        """Save trained model"""
        filename = f"fornet_model_{feature_type}_{self.timestamp}.h5"
        filepath = os.path.join(self.results_dir, "models", filename)
        model.save(filepath)
        print(f"Model saved to: {filepath}")
        return filepath

    def save_metrics(self, metrics, feature_type):
        """Save evaluation metrics"""
        filename = f"metrics_{feature_type}_{self.timestamp}.json"
        filepath = os.path.join(self.results_dir, "metrics", filename)

        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=4)

        print(f"Metrics saved to: {filepath}")
        return filepath


def perform_kfold_validation(features, labels, class_names, feature_type, results_manager, visualizer, k_folds=5):
    """Perform k-fold cross validation as specified in the paper"""

    print(f"Performing {k_folds}-fold cross-validation for {feature_type}...")

    # Initialize cross-validation
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

    # Storage for results across folds
    fold_accuracies = []
    fold_f1_scores = []
    fold_precisions = []
    fold_recalls = []
    all_embeddings = []
    all_labels = []
    all_predictions = []
    all_true_labels = []
    all_pred_probabilities = []

    num_classes = len(class_names)
    input_shape = features.shape[1:]

    fold_results = {}

    for fold, (train_idx, test_idx) in enumerate(skf.split(features, labels)):
        print(f"\nProcessing Fold {fold + 1}/{k_folds}...")

        # Split data for current fold
        X_train_fold, X_test_fold = features[train_idx], features[test_idx]
        y_train_fold, y_test_fold = labels[train_idx], labels[test_idx]

        # Convert labels to categorical
        y_train_cat = to_categorical(y_train_fold, num_classes)
        y_test_cat = to_categorical(y_test_fold, num_classes)

        # Create and build model for this fold
        fornet_fold = ForNetCNN(input_shape, num_classes)
        model_fold = fornet_fold.build_model()

        # Train model for this fold
        history = fornet_fold.train_single_fold(X_train_fold, y_train_cat, epochs=50, batch_size=32)

        # Evaluate on test set for this fold
        y_pred_proba = model_fold.predict(X_test_fold, verbose=0)
        y_pred_fold = np.argmax(y_pred_proba, axis=1)

        # Calculate metrics for this fold
        fold_accuracy = accuracy_score(y_test_fold, y_pred_fold)
        fold_f1 = f1_score(y_test_fold, y_pred_fold, average='weighted')
        fold_precision = precision_score(y_test_fold, y_pred_fold, average='weighted')
        fold_recall = recall_score(y_test_fold, y_pred_fold, average='weighted')

        # Store results
        fold_accuracies.append(fold_accuracy)
        fold_f1_scores.append(fold_f1)
        fold_precisions.append(fold_precision)
        fold_recalls.append(fold_recall)

        # Extract embeddings for this fold
        fold_embeddings = fornet_fold.extract_embeddings(X_test_fold)
        all_embeddings.extend(fold_embeddings)
        all_labels.extend(y_test_fold)
        all_predictions.extend(y_pred_fold)
        all_true_labels.extend(y_test_fold)
        all_pred_probabilities.extend(y_pred_proba)

        # Store fold-specific results
        fold_results[f'fold_{fold + 1}'] = {
            'accuracy': float(fold_accuracy),
            'f1_score': float(fold_f1),
            'precision': float(fold_precision),
            'recall': float(fold_recall),
            'train_size': len(X_train_fold),
            'test_size': len(X_test_fold)
        }

        print(f"Fold {fold + 1} - Accuracy: {fold_accuracy:.4f}, F1: {fold_f1:.4f}")

    # Calculate overall statistics
    mean_accuracy = np.mean(fold_accuracies)
    std_accuracy = np.std(fold_accuracies)
    mean_f1 = np.mean(fold_f1_scores)
    std_f1 = np.std(fold_f1_scores)
    mean_precision = np.mean(fold_precisions)
    std_precision = np.std(fold_precisions)
    mean_recall = np.mean(fold_recalls)
    std_recall = np.std(fold_recalls)

    print(f"\n{k_folds}-Fold Cross-Validation Results for {feature_type}:")
    print(f"Mean Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
    print(f"Mean F1-Score: {mean_f1:.4f} ± {std_f1:.4f}")
    print(f"Mean Precision: {mean_precision:.4f} ± {std_precision:.4f}")
    print(f"Mean Recall: {mean_recall:.4f} ± {std_recall:.4f}")

    # Generate overall classification report
    class_report = classification_report(all_true_labels, all_predictions,
                                         target_names=class_names, output_dict=True)

    # Prepare comprehensive results
    cv_results = {
        'feature_type': feature_type,
        'k_folds': k_folds,
        'fold_results': fold_results,
        'mean_accuracy': float(mean_accuracy),
        'std_accuracy': float(std_accuracy),
        'mean_f1_score': float(mean_f1),
        'std_f1_score': float(std_f1),
        'mean_precision': float(mean_precision),
        'std_precision': float(std_precision),
        'mean_recall': float(mean_recall),
        'std_recall': float(std_recall),
        'individual_accuracies': [float(acc) for acc in fold_accuracies],
        'individual_f1_scores': [float(f1) for f1 in fold_f1_scores],
        'individual_precisions': [float(p) for p in fold_precisions],
        'individual_recalls': [float(r) for r in fold_recalls],
        'classification_report': class_report
    }

    # Convert embeddings and labels to numpy arrays
    all_embeddings = np.array(all_embeddings)
    all_labels = np.array(all_labels)
    all_pred_probabilities = np.array(all_pred_probabilities)

    # Save embeddings from cross-validation
    results_manager.save_embeddings(all_embeddings, all_labels, class_names, f"{feature_type}_kfold")

    # Save cross-validation metrics
    results_manager.save_metrics(cv_results, f"{feature_type}_kfold")

    # Create enhanced visualizations
    visualizer.plot_kfold_results(fold_accuracies, fold_f1_scores, feature_type, k_folds)
    visualizer.plot_confusion_matrix(all_true_labels, all_predictions, class_names, f"{feature_type}_kfold")
    visualizer.plot_embeddings_visualization(all_embeddings, all_labels, class_names, f"{feature_type}_kfold")
    visualizer.plot_roc_curves(all_true_labels, all_pred_probabilities, class_names, f"{feature_type}_kfold")

    return cv_results, all_embeddings, all_labels, all_pred_probabilities


class EnhancedVisualizer:
    """Create comprehensive visualizations for results and analysis"""

    def __init__(self, results_manager):
        self.results_manager = results_manager

        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")

    def create_dataset_analysis(self, audio_stats, class_names, file_paths, labels):
        """Create comprehensive dataset analysis visualizations"""
        print("Creating dataset analysis visualizations...")

        # Convert to DataFrame for easier manipulation
        df_stats = pd.DataFrame(audio_stats)

        # 1. Class distribution and audio characteristics overview
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Class counts
        class_counts = Counter(labels)
        class_names_ordered = [class_names[i] for i in sorted(class_counts.keys())]
        counts = [class_counts[i] for i in sorted(class_counts.keys())]

        axes[0, 0].bar(class_names_ordered, counts, color=plt.cm.Set3(np.linspace(0, 1, len(class_names))))
        axes[0, 0].set_title('Dataset Class Distribution')
        axes[0, 0].set_ylabel('Number of Samples')
        axes[0, 0].tick_params(axis='x', rotation=45)

        # Duration distribution
        axes[0, 1].hist(df_stats['duration'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 1].set_title('Audio Duration Distribution')
        axes[0, 1].set_xlabel('Duration (seconds)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].axvline(np.mean(df_stats['duration']), color='red', linestyle='--',
                           label=f'Mean: {np.mean(df_stats["duration"]):.2f}s')
        axes[0, 1].legend()

        # RMS Energy by class
        sns.boxplot(data=df_stats, x='class', y='rms_energy', ax=axes[1, 0])
        axes[1, 0].set_title('RMS Energy Distribution by Class')
        axes[1, 0].tick_params(axis='x', rotation=45)

        # Spectral centroid by class
        sns.boxplot(data=df_stats, x='class', y='spectral_centroid', ax=axes[1, 1])
        axes[1, 1].set_title('Spectral Centroid Distribution by Class')
        axes[1, 1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        filepath = os.path.join(self.results_manager.results_dir, "dataset_analysis",
                                f"dataset_overview_{self.results_manager.timestamp}.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Detailed audio characteristics
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))

        audio_features = ['duration', 'rms_energy', 'zero_crossing_rate',
                          'spectral_centroid', 'spectral_rolloff', 'spectral_bandwidth']

        for i, feature in enumerate(audio_features):
            row = i // 3
            col = i % 3

            sns.violinplot(data=df_stats, x='class', y=feature, ax=axes[row, col])
            axes[row, col].set_title(f'{feature.replace("_", " ").title()} by Class')
            axes[row, col].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        filepath = os.path.join(self.results_manager.results_dir, "dataset_analysis",
                                f"audio_characteristics_{self.results_manager.timestamp}.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        # 3. Statistical summary
        stats_summary = df_stats.groupby('class').agg({
            'duration': ['mean', 'std', 'min', 'max'],
            'rms_energy': ['mean', 'std'],
            'spectral_centroid': ['mean', 'std'],
            'spectral_rolloff': ['mean', 'std']
        }).round(4)

        # Save statistical summary
        stats_path = os.path.join(self.results_manager.results_dir, "statistical_analysis",
                                  f"dataset_statistics_{self.results_manager.timestamp}.csv")
        stats_summary.to_csv(stats_path)

        print(f"Dataset analysis saved to dataset_analysis/ folder")
        return df_stats, stats_summary

    def plot_kfold_results(self, fold_accuracies, fold_f1_scores, feature_type, k_folds):
        """Plot k-fold validation results with enhanced visualizations"""
        folds = range(1, k_folds + 1)
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Individual fold performance
        axes[0, 0].plot(folds, fold_accuracies, 'o-', label='Accuracy', linewidth=2, markersize=8)
        axes[0, 0].axhline(y=np.mean(fold_accuracies), color='red', linestyle='--',
                           label=f'Mean: {np.mean(fold_accuracies):.3f}')
        axes[0, 0].fill_between(folds,
                                np.mean(fold_accuracies) - np.std(fold_accuracies),
                                np.mean(fold_accuracies) + np.std(fold_accuracies),
                                alpha=0.2, color='red')
        axes[0, 0].set_title(f'{feature_type} - K-Fold Accuracy')
        axes[0, 0].set_xlabel('Fold')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(folds, fold_f1_scores, 'o-', label='F1-Score', color='green', linewidth=2, markersize=8)
        axes[0, 1].axhline(y=np.mean(fold_f1_scores), color='red', linestyle='--',
                           label=f'Mean: {np.mean(fold_f1_scores):.3f}')
        axes[0, 1].fill_between(folds,
                                np.mean(fold_f1_scores) - np.std(fold_f1_scores),
                                np.mean(fold_f1_scores) + np.std(fold_f1_scores),
                                alpha=0.2, color='red')
        axes[0, 1].set_title(f'{feature_type} - K-Fold F1-Score')
        axes[0, 1].set_xlabel('Fold')
        axes[0, 1].set_ylabel('F1-Score')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Box plots for performance distribution
        axes[1, 0].boxplot([fold_accuracies, fold_f1_scores], labels=['Accuracy', 'F1-Score'])
        axes[1, 0].set_title(f'{feature_type} - Performance Distribution')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].grid(True, alpha=0.3)

        # Performance summary statistics
        axes[1, 1].text(0.1, 0.9, f'Performance Summary:', transform=axes[1, 1].transAxes, fontsize=12, weight='bold')
        axes[1, 1].text(0.1, 0.8, f'Accuracy: {np.mean(fold_accuracies):.3f} ± {np.std(fold_accuracies):.3f}',
                        transform=axes[1, 1].transAxes, fontsize=10)
        axes[1, 1].text(0.1, 0.7, f'F1-Score: {np.mean(fold_f1_scores):.3f} ± {np.std(fold_f1_scores):.3f}',
                        transform=axes[1, 1].transAxes, fontsize=10)
        axes[1, 1].text(0.1, 0.6, f'Min Accuracy: {np.min(fold_accuracies):.3f}', transform=axes[1, 1].transAxes,
                        fontsize=10)
        axes[1, 1].text(0.1, 0.5, f'Max Accuracy: {np.max(fold_accuracies):.3f}', transform=axes[1, 1].transAxes,
                        fontsize=10)
        axes[1, 1].axis('off')

        plt.tight_layout()
        filename = f"kfold_results_{feature_type}_{self.results_manager.timestamp}.png"
        filepath = os.path.join(self.results_manager.results_dir, "visualizations", filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"K-fold results plot saved to: {filepath}")
        return filepath

    def plot_confusion_matrix(self, y_true, y_pred, class_names, feature_type):
        """Plot enhanced confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Raw confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names, ax=axes[0])
        axes[0].set_title(f'Confusion Matrix - {feature_type}')
        axes[0].set_xlabel('Predicted Label')
        axes[0].set_ylabel('True Label')

        # Normalized confusion matrix
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names, ax=axes[1])
        axes[1].set_title(f'Normalized Confusion Matrix - {feature_type}')
        axes[1].set_xlabel('Predicted Label')
        axes[1].set_ylabel('True Label')

        plt.tight_layout()
        filename = f"confusion_matrix_{feature_type}_{self.results_manager.timestamp}.png"
        filepath = os.path.join(self.results_manager.results_dir, "visualizations", filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Confusion matrix saved to: {filepath}")
        return filepath

    def plot_embeddings_visualization(self, embeddings, labels, class_names, feature_type):
        """Create comprehensive embeddings visualization"""
        print(f"Creating embeddings visualization for {feature_type}...")

        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        colors = plt.cm.tab10(np.linspace(0, 1, len(class_names)))

        # 1. t-SNE visualization
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings) - 1))
        embeddings_tsne = tsne.fit_transform(embeddings)

        for i, class_name in enumerate(class_names):
            mask = np.array(labels) == i
            axes[0].scatter(embeddings_tsne[mask, 0], embeddings_tsne[mask, 1],
                            c=[colors[i]], label=class_name, alpha=0.7, s=50)

        axes[0].set_title(f'{feature_type} - t-SNE Embeddings')
        axes[0].set_xlabel('t-SNE Component 1')
        axes[0].set_ylabel('t-SNE Component 2')
        axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0].grid(True, alpha=0.3)

        # 2. PCA visualization
        pca = PCA(n_components=2, random_state=42)
        embeddings_pca = pca.fit_transform(embeddings)

        for i, class_name in enumerate(class_names):
            mask = np.array(labels) == i
            axes[1].scatter(embeddings_pca[mask, 0], embeddings_pca[mask, 1],
                            c=[colors[i]], label=class_name, alpha=0.7, s=50)

        axes[1].set_title(
            f'{feature_type} - PCA Embeddings\n(Explained Var: {pca.explained_variance_ratio_.sum():.3f})')
        axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.3f})')
        axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.3f})')
        axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1].grid(True, alpha=0.3)

        # 3. UMAP visualization (if available)
        try:
            umap_reducer = UMAP(n_components=2, random_state=42)
            embeddings_umap = umap_reducer.fit_transform(embeddings)

            for i, class_name in enumerate(class_names):
                mask = np.array(labels) == i
                axes[2].scatter(embeddings_umap[mask, 0], embeddings_umap[mask, 1],
                                c=[colors[i]], label=class_name, alpha=0.7, s=50)

            axes[2].set_title(f'{feature_type} - UMAP Embeddings')
            axes[2].set_xlabel('UMAP Component 1')
            axes[2].set_ylabel('UMAP Component 2')
            axes[2].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            axes[2].grid(True, alpha=0.3)
        except ImportError:
            axes[2].text(0.5, 0.5, 'UMAP not available\nInstall umap-learn',
                         transform=axes[2].transAxes, ha='center', va='center')
            axes[2].axis('off')

        plt.tight_layout()
        filename = f"embeddings_visualization_{feature_type}_{self.results_manager.timestamp}.png"
        filepath = os.path.join(self.results_manager.results_dir, "visualizations", filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Embeddings visualization saved to: {filepath}")
        return filepath

    def plot_roc_curves(self, y_true, y_pred_proba, class_names, feature_type):
        """Plot ROC curves for multi-class classification"""
        print(f"Creating ROC curves for {feature_type}...")

        # Binarize the output
        y_true_bin = label_binarize(y_true, classes=range(len(class_names)))
        n_classes = len(class_names)

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Calculate ROC curve and AUC for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Plot individual class ROC curves
        colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
        for i, color in enumerate(colors):
            axes[0].plot(fpr[i], tpr[i], color=color, lw=2,
                         label=f'{class_names[i]} (AUC = {roc_auc[i]:.3f})')

        axes[0].plot([0, 1], [0, 1], 'k--', lw=2)
        axes[0].set_xlim([0.0, 1.0])
        axes[0].set_ylim([0.0, 1.05])
        axes[0].set_xlabel('False Positive Rate')
        axes[0].set_ylabel('True Positive Rate')
        axes[0].set_title(f'{feature_type} - ROC Curves by Class')
        axes[0].legend(loc="lower right")
        axes[0].grid(True, alpha=0.3)

        # Compute micro-average ROC curve and AUC
        fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_pred_proba.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # Compute macro-average ROC curve and AUC
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= n_classes
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        # Plot average ROC curves
        axes[1].plot(fpr["micro"], tpr["micro"],
                     label=f'Micro-avg (AUC = {roc_auc["micro"]:.3f})',
                     color='deeppink', linestyle=':', linewidth=4)

        axes[1].plot(fpr["macro"], tpr["macro"],
                     label=f'Macro-avg (AUC = {roc_auc["macro"]:.3f})',
                     color='navy', linestyle=':', linewidth=4)

        axes[1].plot([0, 1], [0, 1], 'k--', lw=2)
        axes[1].set_xlim([0.0, 1.0])
        axes[1].set_ylim([0.0, 1.05])
        axes[1].set_xlabel('False Positive Rate')
        axes[1].set_ylabel('True Positive Rate')
        axes[1].set_title(f'{feature_type} - Average ROC Curves')
        axes[1].legend(loc="lower right")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        filename = f"roc_curves_{feature_type}_{self.results_manager.timestamp}.png"
        filepath = os.path.join(self.results_manager.results_dir, "visualizations", filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"ROC curves saved to: {filepath}")
        return filepath, roc_auc

    def plot_feature_representations(self, features, labels, class_names, feature_type):
        """Plot feature representations with analysis"""
        fig, axes = plt.subplots(2, 4, figsize=(20, 12))
        fig.suptitle(f'{feature_type} Feature Representations Analysis', fontsize=16)

        # Plot samples from each class (first 8 classes or all if less)
        num_classes_to_show = min(8, len(class_names))

        for i in range(num_classes_to_show):
            row = i // 4
            col = i % 4

            # Find samples of this class
            class_indices = np.where(np.array(labels) == i)[0]
            if len(class_indices) > 0:
                # Show average feature representation
                class_features = features[class_indices]
                mean_features = np.mean(class_features, axis=0).squeeze()

                im = axes[row, col].imshow(mean_features, aspect='auto', origin='lower', cmap='viridis')
                axes[row, col].set_title(f'{class_names[i]}\n(Avg of {len(class_indices)} samples)')
                axes[row, col].set_xlabel('Time')
                axes[row, col].set_ylabel('Frequency')
                plt.colorbar(im, ax=axes[row, col])

        # Hide unused subplots
        for i in range(num_classes_to_show, 8):
            row = i // 4
            col = i % 4
            axes[row, col].axis('off')

        plt.tight_layout()
        filename = f"feature_representations_{feature_type}_{self.results_manager.timestamp}.png"
        filepath = os.path.join(self.results_manager.results_dir, "feature_representations", filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Feature representations saved to: {filepath}")
        return filepath


def main(source_directory, results_directory, use_kfold=True):
    """Main execution function with enhanced visualizations"""

    print("=== Enhanced ForNet Audio Classification System ===")
    print(f"Source directory: {source_directory}")
    print(f"Results directory: {results_directory}")
    print(f"Using {'5-fold cross-validation' if use_kfold else 'train-test split'}")

    # Initialize components
    processor = ForNetAudioProcessor()
    data_loader = DataLoader(source_directory, processor)
    results_manager = ResultsManager(results_directory)
    visualizer = EnhancedVisualizer(results_manager)

    # Load data with statistics
    file_paths, labels, class_names = data_loader.load_data()
    num_classes = len(class_names)

    # Create comprehensive dataset analysis
    df_audio_stats, stats_summary = visualizer.create_dataset_analysis(
        data_loader.audio_stats, class_names, file_paths, labels)

    # Feature types to process
    feature_types = ['MFCC', 'Mel', 'LogMel']

    all_results = {}
    all_kfold_results = {}
    all_features = {}
    all_embeddings = {}

    for feature_type in feature_types:
        print(f"\n{'=' * 50}")
        print(f"Processing {feature_type} features")
        print(f"{'=' * 50}")

        # Extract features
        features = []
        valid_labels = []

        print(f"Extracting {feature_type} features...")
        for i, file_path in enumerate(file_paths):
            audio = processor.load_audio(file_path)
            if audio is not None:
                if feature_type == 'MFCC':
                    feature = processor.extract_mfcc(audio)
                    feature = feature.reshape(20, 65, 1)
                elif feature_type == 'Mel':
                    feature = processor.extract_mel_spectrogram(audio)
                    feature = feature.reshape(128, 66, 1)
                else:  # LogMel
                    feature = processor.extract_log_mel_spectrogram(audio)
                    feature = feature.reshape(128, 128, 1)

                features.append(feature)
                valid_labels.append(labels[i])

            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(file_paths)} files")

        features = np.array(features)
        valid_labels = np.array(valid_labels)
        all_features[feature_type] = features

        print(f"Features shape: {features.shape}")
        print(f"Labels shape: {valid_labels.shape}")

        # Create feature representations visualization
        visualizer.plot_feature_representations(features, valid_labels, class_names, feature_type)

        if use_kfold:
            # Perform 5-fold cross validation
            cv_results, cv_embeddings, cv_labels, cv_pred_probs = perform_kfold_validation(
                features, valid_labels, class_names, feature_type,
                results_manager, visualizer, k_folds=5
            )
            all_kfold_results[feature_type] = cv_results
            all_embeddings[feature_type] = cv_embeddings

        # Single train-test split
        print(f"\nPerforming single train-test evaluation for {feature_type}...")

        X_train, X_test, y_train, y_test = train_test_split(
            features, valid_labels, test_size=0.2, random_state=42, stratify=valid_labels
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )

        # Convert labels to categorical
        y_train_cat = to_categorical(y_train, num_classes)
        y_val_cat = to_categorical(y_val, num_classes)
        y_test_cat = to_categorical(y_test, num_classes)

        print(f"Training set: {X_train.shape}")
        print(f"Validation set: {X_val.shape}")
        print(f"Test set: {X_test.shape}")

        # Create and train model
        input_shape = features.shape[1:]
        fornet = ForNetCNN(input_shape, num_classes)
        model = fornet.build_model()

        print(f"Model architecture for {feature_type}:")
        model.summary()

        # Train model
        print(f"Training model for {feature_type}...")
        history = fornet.train(X_train, y_train_cat, X_val, y_val_cat, epochs=100, batch_size=32)

        # Evaluate model
        test_loss, test_accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
        y_pred_proba = model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)

        # Calculate detailed metrics
        f1 = f1_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        class_report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)

        # Store results
        results = {
            'feature_type': feature_type,
            'test_accuracy': float(test_accuracy),
            'test_loss': float(test_loss),
            'f1_score': float(f1),
            'precision': float(precision),
            'recall': float(recall),
            'classification_report': class_report
        }
        all_results[feature_type] = results

        print(f"\n{feature_type} Results:")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")

        # Extract embeddings
        embeddings = fornet.extract_embeddings(X_test)
        all_embeddings[f"{feature_type}_single"] = embeddings

        # Save model and results
        results_manager.save_model(model, feature_type)
        results_manager.save_metrics(results, feature_type)
        results_manager.save_embeddings(embeddings, y_test, class_names, feature_type)

        # Create visualizations
        visualizer.plot_confusion_matrix(y_test, y_pred, class_names, feature_type)
        visualizer.plot_embeddings_visualization(embeddings, y_test, class_names, feature_type)
        visualizer.plot_roc_curves(y_test, y_pred_proba, class_names, feature_type)

    # Summary and comparison
    print(f"\n{'=' * 70}")
    print("SUMMARY - COMPARISON OF FEATURE TYPES")
    print(f"{'=' * 70}")

    print("\nSingle Split Results:")
    print(f"{'Feature Type':<15} {'Accuracy':<10} {'F1-Score':<10} {'Precision':<10} {'Recall':<10}")
    print("-" * 60)
    for feature_type, results in all_results.items():
        print(f"{feature_type:<15} {results['test_accuracy']:<10.4f} "
              f"{results['f1_score']:<10.4f} {results['precision']:<10.4f} {results['recall']:<10.4f}")

    if use_kfold and all_kfold_results:
        print("\nK-Fold Cross-Validation Results:")
        print(f"{'Feature Type':<15} {'Mean Accuracy':<15} {'Std Accuracy':<15} {'Mean F1':<10} {'Std F1':<10}")
        print("-" * 80)
        for feature_type, results in all_kfold_results.items():
            print(f"{feature_type:<15} {results['mean_accuracy']:<15.4f} "
                  f"{results['std_accuracy']:<15.4f} {results['mean_f1_score']:<10.4f} {results['std_f1_score']:<10.4f}")

    # Find best performing feature type
    if all_results:
        best_feature_type = max(all_results.keys(), key=lambda k: all_results[k]['test_accuracy'])
        print(f"\nBest performing feature type: {best_feature_type}")
        print(f"Best accuracy: {all_results[best_feature_type]['test_accuracy']:.4f}")

    # Create summary visualization comparing all feature types
    if len(all_results) > 1:
        feature_names = list(all_results.keys())
        accuracies = [all_results[ft]['test_accuracy'] for ft in feature_names]
        f1_scores = [all_results[ft]['f1_score'] for ft in feature_names]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Accuracy comparison
        bars1 = ax1.bar(feature_names, accuracies, color=['skyblue', 'lightgreen', 'lightcoral'])
        ax1.set_title('Test Accuracy Comparison')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0, 1)
        for i, bar in enumerate(bars1):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                     f'{accuracies[i]:.3f}', ha='center', va='bottom')

        # F1-Score comparison
        bars2 = ax2.bar(feature_names, f1_scores, color=['skyblue', 'lightgreen', 'lightcoral'])
        ax2.set_title('F1-Score Comparison')
        ax2.set_ylabel('F1-Score')
        ax2.set_ylim(0, 1)
        for i, bar in enumerate(bars2):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                     f'{f1_scores[i]:.3f}', ha='center', va='bottom')

        plt.tight_layout()
        summary_path = os.path.join(results_manager.results_dir, "visualizations",
                                    f"feature_comparison_{results_manager.timestamp}.png")
        plt.savefig(summary_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Feature comparison visualization saved to: {summary_path}")

    # Save final summary
    final_summary = {
        'timestamp': results_manager.timestamp,
        'dataset_info': {
            'total_files': len(file_paths),
            'num_classes': num_classes,
            'class_names': class_names
        },
        'single_split_results': all_results,
        'kfold_results': all_kfold_results if use_kfold else {},
        'best_feature_type': best_feature_type if all_results else None
    }

    summary_path = os.path.join(results_manager.results_dir, "metrics",
                                f"final_summary_{results_manager.timestamp}.json")
    with open(summary_path, 'w') as f:
        json.dump(final_summary, f, indent=4)

    print(f"\nFinal summary saved to: {summary_path}")
    print(f"All results saved in: {results_manager.results_dir}")
    print("\n=== Analysis Complete ===")

    return all_results, all_kfold_results, all_embeddings


# Example usage
if __name__ == "__main__":
    # Set your paths here
    SOURCE_DIRECTORY = r"C:\Users\visha\Desktop\forest_research\FSM5"  # Directory with class subdirectories
    RESULTS_DIRECTORY = "C:/Users/visha/Desktop/forest_research/new_studies/final_datasets_results/CNN_embeddings"  # Where to save results

    # Configure TensorFlow to use GPU if available
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("GPU available and configured")
    else:
        print("Running on CPU")

    # Run the complete analysis
    try:
        # Run with both single split and k-fold validation
        results, kfold_results, embeddings = main(
            source_directory=SOURCE_DIRECTORY,
            results_directory=RESULTS_DIRECTORY,
            use_kfold=True  # Set to False to skip k-fold validation
        )

        print("\nExperiment completed successfully!")
        print(f"Results saved in: {RESULTS_DIRECTORY}")

    except Exception as e:
        print(f"Error during execution: {str(e)}")
        import traceback

        traceback.print_exc()

"""
USAGE INSTRUCTIONS:

1. Install required dependencies:
   pip install tensorflow librosa scikit-learn matplotlib seaborn pandas numpy
   pip install umap-learn plotly  # Optional for enhanced visualizations

2. Organize your audio dataset:
   /path/to/dataset/
   ├── class1/
   │   ├── audio1.wav
   │   ├── audio2.wav
   │   └── ...
   ├── class2/
   │   ├── audio1.wav
   │   └── ...
   └── ...

3. Update the SOURCE_DIRECTORY and RESULTS_DIRECTORY paths in the main section

4. Run the script:
   python fornet_complete.py

FEATURES:
- Implements exact ForNet CNN architecture from research paper
- Extracts MFCC, Mel-spectrogram, and Log-Mel spectrogram features
- Performs both single train-test split and 5-fold cross-validation
- Creates comprehensive visualizations including:
  - Dataset analysis and statistics
  - Training history plots
  - Confusion matrices
  - ROC curves
  - Embedding visualizations (t-SNE, PCA, UMAP)
  - Feature representation analysis
- Saves all models, embeddings, and metrics for future use
- Provides detailed performance comparisons between feature types

OUTPUT STRUCTURE:
results/
├── dataset_analysis/      # Dataset statistics and visualizations
├── embeddings/           # Saved embeddings for ensemble methods
├── feature_representations/ # Feature analysis plots
├── layer_analysis/       # CNN layer activation analysis
├── metrics/              # Performance metrics and reports
├── models/               # Trained models (.h5 files)
├── statistical_analysis/ # Statistical comparisons
└── visualizations/       # All plots and charts
"""