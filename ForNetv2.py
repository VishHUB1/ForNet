import os
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import tensorflow as tf
from tensorflow import keras
layers = keras.layers
models = keras.models
callbacks = keras.callbacks
import pickle
import json
from pathlib import Path
import warnings
from scipy.ndimage import zoom

warnings.filterwarnings('ignore')


class CNNEnsembleAudioClassifier:
    def __init__(self, data_dir, results_dir="results", sample_rate=22050, duration=3.0):
        self.data_dir = data_dir
        self.results_dir = results_dir
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_samples = int(sample_rate * duration)

        # Feature extraction parameters
        self.mfcc_params = {'n_mfcc': 20, 'n_fft': 2048, 'hop_length': 512}
        self.mel_params = {'n_mels': 128, 'n_fft': 2048, 'hop_length': 512}

        # Create results directory structure
        self.create_results_structure()

        # Initialize data containers
        self.audio_data = []
        self.labels = []
        self.file_paths = []
        self.label_encoder = LabelEncoder()

        # Models container
        self.baseline_models = {}
        self.baseline_histories = {}
        self.embedding_models = {}
        self.ensemble_models = {}
        self.embeddings = {}
        self.baseline_predictions = {}
        self.ensemble_predictions = {}
        self.baseline_metrics = {}
        self.ensemble_metrics = {}

    def create_results_structure(self):
        """Create organized results directory structure"""
        base_path = Path(self.results_dir)
        directories = [
            'models/baseline', 'models/ensemble', 'models/embedding_extractors',
            'metrics/baseline', 'metrics/ensemble', 'metrics/comparisons',
            'visualizations/features', 'visualizations/training',
            'visualizations/evaluation', 'visualizations/comparisons',
            'visualizations/dataset_analysis', 'data_splits', 'embeddings'
        ]

        for dir_path in directories:
            (base_path / dir_path).mkdir(parents=True, exist_ok=True)

    def load_audio_data(self):
        """Load audio files from the directory structure with 3 classes and subdirectories"""
        print("Loading audio data...")
        print(f"Dataset directory: {self.data_dir}")

        class_count = 0
        for class_name in os.listdir(self.data_dir):
            class_path = os.path.join(self.data_dir, class_name)
            if os.path.isdir(class_path):
                class_count += 1
                print(f"Processing class: {class_name}")

                subdirs_found = 0
                files_found = 0

                for subdir in os.listdir(class_path):
                    subdir_path = os.path.join(class_path, subdir)
                    if os.path.isdir(subdir_path):
                        subdirs_found += 1
                        print(f"  Processing subdirectory: {subdir}")

                        for file_name in os.listdir(subdir_path):
                            if file_name.endswith(('.wav', '.mp3', '.flac', '.m4a')):
                                file_path = os.path.join(subdir_path, file_name)
                                try:
                                    # Load audio file
                                    audio, sr = librosa.load(file_path, sr=self.sample_rate, duration=self.duration)

                                    # Pad or truncate to fixed length
                                    if len(audio) < self.n_samples:
                                        audio = np.pad(audio, (0, self.n_samples - len(audio)), 'constant')
                                    else:
                                        audio = audio[:self.n_samples]

                                    self.audio_data.append(audio)
                                    self.labels.append(class_name)
                                    self.file_paths.append(file_path)
                                    files_found += 1

                                except Exception as e:
                                    print(f"Error loading {file_path}: {e}")

                print(f"  Found {subdirs_found} subdirectories with {files_found} audio files")

        print(f"\nSummary:")
        print(f"Total classes found: {class_count}")
        print(f"Total audio files loaded: {len(self.audio_data)}")

        if not self.audio_data:
            raise ValueError("No audio files found! Please check your directory structure and file paths.")

        self.analyze_dataset()

    def analyze_dataset(self):
        """Analyze and visualize dataset characteristics"""
        print("Analyzing dataset...")

        # Create DataFrame for analysis
        df = pd.DataFrame({
            'label': self.labels,
            'file_path': self.file_paths
        })

        # Class distribution
        class_counts = df['label'].value_counts()
        print("\nClass Distribution:")
        print(class_counts)

        # Visualize class distribution
        plt.figure(figsize=(12, 8))
        bars = plt.bar(class_counts.index, class_counts.values, color=['skyblue', 'lightcoral', 'lightgreen'])
        plt.title('Class Distribution in Dataset', fontsize=16, fontweight='bold')
        plt.xlabel('Classes', fontsize=14)
        plt.ylabel('Number of Samples', fontsize=14)
        plt.xticks(rotation=45)

        # Add value labels on bars
        for bar, count in zip(bars, class_counts.values):
            plt.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.5,
                     f'{count}', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/visualizations/dataset_analysis/class_distribution.png', dpi=300)
        plt.close()

        # Save dataset statistics
        stats = {
            'total_samples': len(self.audio_data),
            'classes': list(class_counts.index),
            'class_counts': class_counts.to_dict(),
            'sample_rate': self.sample_rate,
            'duration': self.duration
        }

        with open(f'{self.results_dir}/data_splits/dataset_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)

    def extract_mfcc(self, audio):
        """Extract MFCC features"""
        mfcc = librosa.feature.mfcc(y=audio, sr=self.sample_rate, **self.mfcc_params)
        return mfcc.T  # Shape: (time, n_mfcc)

    def extract_mel_spectrogram(self, audio):
        """Extract Mel spectrogram features"""
        mel = librosa.feature.melspectrogram(y=audio, sr=self.sample_rate, **self.mel_params)
        return mel.T  # Shape: (time, n_mels)

    def extract_log_mel(self, audio):
        """Extract Log-Mel spectrogram features"""
        mel = librosa.feature.melspectrogram(y=audio, sr=self.sample_rate, **self.mel_params)
        log_mel = librosa.power_to_db(mel, ref=np.max)
        return log_mel.T  # Shape: (time, n_mels)

    def extract_all_features(self):
        """Extract all three types of features"""
        print("Extracting features...")

        features = {
            'mfcc': [],
            'mel': [],
            'log_mel': []
        }

        for i, audio in enumerate(self.audio_data):
            if i % 100 == 0:
                print(f"Processing audio {i + 1}/{len(self.audio_data)}")

            # Extract features
            mfcc = self.extract_mfcc(audio)
            mel = self.extract_mel_spectrogram(audio)
            log_mel = self.extract_log_mel(audio)

            features['mfcc'].append(mfcc)
            features['mel'].append(mel)
            features['log_mel'].append(log_mel)

        # Convert to numpy arrays
        for feature_type in features:
            features[feature_type] = np.array(features[feature_type])
            print(f"{feature_type.upper()} shape: {features[feature_type].shape}")

        self.visualize_features(features)
        return features

    def visualize_features(self, features, n_samples=2):
        """Visualize extracted features with waveforms"""
        print("Creating feature visualizations...")

        # Select random samples from each class
        unique_labels = list(set(self.labels))

        for class_label in unique_labels:
            class_indices = [i for i, label in enumerate(self.labels) if label == class_label]
            sample_indices = np.random.choice(class_indices, min(n_samples, len(class_indices)), replace=False)

            for idx in sample_indices:
                fig, axes = plt.subplots(4, 1, figsize=(15, 12))

                # Original waveform
                axes[0].plot(np.linspace(0, self.duration, len(self.audio_data[idx])), self.audio_data[idx])
                axes[0].set_title(f'Waveform - {class_label}', fontsize=14, fontweight='bold')
                axes[0].set_xlabel('Time (s)')
                axes[0].set_ylabel('Amplitude')
                axes[0].grid(True, alpha=0.3)

                # MFCC
                librosa.display.specshow(features['mfcc'][idx].T, x_axis='time', ax=axes[1])
                axes[1].set_title('MFCC Features', fontsize=14, fontweight='bold')
                axes[1].set_ylabel('MFCC Coefficients')

                # Mel Spectrogram
                librosa.display.specshow(features['mel'][idx].T, x_axis='time', ax=axes[2], y_axis='mel')
                axes[2].set_title('Mel Spectrogram', fontsize=14, fontweight='bold')

                # Log-Mel Spectrogram
                librosa.display.specshow(features['log_mel'][idx].T, x_axis='time', ax=axes[3], y_axis='mel')
                axes[3].set_title('Log-Mel Spectrogram', fontsize=14, fontweight='bold')
                axes[3].set_xlabel('Time (s)')

                plt.tight_layout()
                plt.savefig(f'{self.results_dir}/visualizations/features/{class_label}_sample_{idx}_features.png',
                            dpi=300, bbox_inches='tight')
                plt.close()

    def prepare_cnn_input(self, features, target_shape):
        """Prepare features for CNN input with proper reshaping"""
        resized_features = []

        for feature in features:
            # Resize to target shape
            if feature.shape != target_shape[:2]:
                # Simple interpolation for resizing
                zoom_factors = [target_shape[0] / feature.shape[0], target_shape[1] / feature.shape[1]]
                resized = zoom(feature, zoom_factors, order=1)
            else:
                resized = feature

            # Add channel dimension
            resized_features.append(resized.reshape(*target_shape))

        return np.array(resized_features)

    def create_baseline_cnn(self, input_shape, num_classes):
        """Create the baseline CNN architecture"""
        model = models.Sequential([
            # Layer 1: Conv2D with 32 filters, 5x5 kernel
            layers.Conv2D(32, (5, 5), strides=1, padding='same', input_shape=input_shape),

            # Layer 2: MaxPooling2D
            layers.MaxPooling2D((2, 2), strides=2),

            # Layer 3: ReLU
            layers.ReLU(),

            # Layer 4: Conv2D with 64 filters, 4x4 kernel
            layers.Conv2D(64, (4, 4), strides=1, padding='valid'),

            # Layer 5: MaxPooling2D
            layers.MaxPooling2D((2, 2)),

            # Layer 6: ReLU
            layers.ReLU(),

            # Layer 7: Conv2D with 128 filters, 3x3 kernel
            layers.Conv2D(128, (3, 3), strides=1, padding='valid'),

            # Layer 8: GlobalAveragePooling2D
            layers.GlobalAveragePooling2D(),

            # Layer 9: Dense with 50 units
            layers.Dense(50, activation='relu'),

            # Layer 10: Dropout
            layers.Dropout(0.5),

            # Layer 11: Output softmax
            layers.Dense(num_classes, activation='softmax')
        ])

        return model

    def create_embedding_extractor(self, input_shape):
        """Create CNN model for embedding extraction (without final classification layers)"""
        model = models.Sequential([
            # Layer 1: Conv2D with 32 filters, 5x5 kernel
            layers.Conv2D(32, (5, 5), strides=1, padding='same', input_shape=input_shape),

            # Layer 2: MaxPooling2D
            layers.MaxPooling2D((2, 2), strides=2),

            # Layer 3: ReLU
            layers.ReLU(),

            # Layer 4: Conv2D with 64 filters, 4x4 kernel
            layers.Conv2D(64, (4, 4), strides=1, padding='valid'),

            # Layer 5: MaxPooling2D
            layers.MaxPooling2D((2, 2)),

            # Layer 6: ReLU
            layers.ReLU(),

            # Layer 7: Conv2D with 128 filters, 3x3 kernel
            layers.Conv2D(128, (3, 3), strides=1, padding='valid'),

            # Layer 8: GlobalAveragePooling2D (This is our embedding layer)
            layers.GlobalAveragePooling2D()
        ])

        return model

    def train_baseline_cnn(self, X_train, y_train, X_val, y_val, feature_type, num_classes):
        """Train the baseline CNN model"""
        print(f"Training baseline {feature_type.upper()} CNN model...")

        # Create model
        model = self.create_baseline_cnn(X_train.shape[1:], num_classes)

        # Compile model
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        # Print model summary
        print(f"\nBaseline {feature_type.upper()} CNN Model Architecture:")
        model.summary()

        # Callbacks
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )

        # Train model
        history = model.fit(
            X_train, y_train,
            batch_size=32,
            epochs=100,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping],
            verbose=1
        )

        # Save model
        model.save(f'{self.results_dir}/models/baseline/{feature_type}_baseline_cnn.h5')

        return model, history

    def train_embedding_extractor(self, X_train, y_train, X_val, y_val, feature_type):
        """Train CNN for embedding extraction"""
        print(f"Training {feature_type.upper()} embedding extractor...")

        # Create embedding extractor
        embedding_model = self.create_embedding_extractor(X_train.shape[1:])

        # Create a temporary full model for training
        temp_model = models.Sequential([
            embedding_model,
            layers.Dense(50, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(y_train.shape[1], activation='softmax')
        ])

        # Compile model
        temp_model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        # Callbacks
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )

        # Train model
        temp_model.fit(
            X_train, y_train,
            batch_size=32,
            epochs=100,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping],
            verbose=1
        )

        # Save embedding extractor
        embedding_model.save(f'{self.results_dir}/models/embedding_extractors/{feature_type}_embedding_extractor.h5')

        return embedding_model

    def extract_embeddings(self, embedding_model, X_data, feature_type):
        """Extract embeddings using trained CNN"""
        print(f"Extracting {feature_type.upper()} embeddings...")

        embeddings = embedding_model.predict(X_data, verbose=0)

        # Save embeddings
        np.save(f'{self.results_dir}/embeddings/{feature_type}_embeddings.npy', embeddings)

        return embeddings

    def train_ensemble_models(self, X_embeddings, y_true, feature_type):
        """Train ensemble models on CNN embeddings"""
        print(f"Training ensemble models on {feature_type.upper()} embeddings...")

        # Initialize ensemble models
        models = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'xgboost': xgb.XGBClassifier(random_state=42),
            'svm': SVC(probability=True, random_state=42),
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000)
        }

        trained_models = {}
        for model_name, model in models.items():
            print(f"  Training {model_name}...")
            model.fit(X_embeddings, y_true)
            trained_models[model_name] = model

            # Save model
            with open(f'{self.results_dir}/models/ensemble/{feature_type}_{model_name}.pkl', 'wb') as f:
                pickle.dump(model, f)

        return trained_models

    def plot_training_history(self, history, feature_type, model_type="baseline"):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Plot training & validation accuracy
        ax1.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        ax1.set_title(f'{feature_type.upper()} {model_type.title()} - Model Accuracy', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot training & validation loss
        ax2.plot(history.history['loss'], label='Training Loss', linewidth=2)
        ax2.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
        ax2.set_title(f'{feature_type.upper()} {model_type.title()} - Model Loss', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Loss', fontsize=12)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/visualizations/training/{feature_type}_{model_type}_training_history.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

    def evaluate_baseline_model(self, model, X_test, y_test, feature_type):
        """Evaluate baseline CNN model and generate metrics"""
        print(f"Evaluating baseline {feature_type.upper()} CNN model...")

        # Predictions
        y_pred_prob = model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_prob, axis=1)
        y_true = np.argmax(y_test, axis=1)

        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        class_report = classification_report(y_true, y_pred, target_names=self.label_encoder.classes_, output_dict=True)
        conf_matrix = confusion_matrix(y_true, y_pred)

        # Store results
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'classification_report': class_report,
            'confusion_matrix': conf_matrix.tolist()
        }

        # Save metrics
        with open(f'{self.results_dir}/metrics/baseline/{feature_type}_baseline_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2, default=str)

        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.label_encoder.classes_,
                    yticklabels=self.label_encoder.classes_)
        plt.title(f'Baseline {feature_type.upper()} CNN - Confusion Matrix', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=14)
        plt.xlabel('Predicted Label', fontsize=14)
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/visualizations/evaluation/{feature_type}_baseline_confusion_matrix.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

        return metrics, y_pred_prob

    def evaluate_ensemble_models(self, models, X_test, y_true, feature_type):
        """Evaluate ensemble models and generate metrics"""
        print(f"Evaluating ensemble models on {feature_type.upper()} embeddings...")

        ensemble_metrics = {}

        for model_name, model in models.items():
            print(f"  Evaluating {model_name}...")

            # Predictions
            y_pred = model.predict(X_test)
            y_pred_prob = model.predict_proba(X_test)

            # Calculate metrics
            accuracy = accuracy_score(y_true, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
            class_report = classification_report(y_true, y_pred, target_names=self.label_encoder.classes_,
                                                 output_dict=True)
            conf_matrix = confusion_matrix(y_true, y_pred)

            # Store results
            metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'classification_report': class_report,
                'confusion_matrix': conf_matrix.tolist()
            }

            ensemble_metrics[model_name] = metrics

            # Save metrics
            with open(f'{self.results_dir}/metrics/ensemble/{feature_type}_{model_name}_metrics.json', 'w') as f:
                json.dump(metrics, f, indent=2, default=str)

            # Plot confusion matrix
            plt.figure(figsize=(10, 8))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                        xticklabels=self.label_encoder.classes_,
                        yticklabels=self.label_encoder.classes_)
            plt.title(f'{feature_type.upper()} + {model_name.replace("_", " ").title()} - Confusion Matrix',
                      fontsize=16, fontweight='bold')
            plt.ylabel('True Label', fontsize=14)
            plt.xlabel('Predicted Label', fontsize=14)
            plt.tight_layout()
            plt.savefig(
                f'{self.results_dir}/visualizations/evaluation/{feature_type}_{model_name}_confusion_matrix.png',
                dpi=300, bbox_inches='tight')
            plt.close()

        return ensemble_metrics

    def create_comprehensive_comparisons(self):
        """Create comprehensive comparison visualizations"""
        print("Creating comprehensive comparison visualizations...")

        # 1. Baseline CNN Accuracy Comparison
        baseline_accuracies = {feature_type: self.baseline_metrics[feature_type]['accuracy']
                               for feature_type in self.baseline_metrics}

        plt.figure(figsize=(12, 8))
        feature_types = list(baseline_accuracies.keys())
        accuracy_values = list(baseline_accuracies.values())

        bars = plt.bar(feature_types, accuracy_values, color=['skyblue', 'lightcoral', 'lightgreen'])
        plt.title('Baseline CNN Accuracy Comparison Across Feature Types', fontsize=16, fontweight='bold')
        plt.ylabel('Accuracy', fontsize=14)
        plt.xlabel('Feature Type', fontsize=14)
        plt.ylim(0, 1)

        # Add value labels on bars
        for bar, accuracy in zip(bars, accuracy_values):
            plt.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.01,
                     f'{accuracy:.3f}', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/visualizations/comparisons/baseline_cnn_accuracy_comparison.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Ensemble Models Comparison
        ensemble_comparison_data = []
        for feature_type in self.ensemble_metrics:
            for model_name in self.ensemble_metrics[feature_type]:
                metrics = self.ensemble_metrics[feature_type][model_name]
                ensemble_comparison_data.append({
                    'Feature': feature_type.upper(),
                    'Model': model_name.replace('_', ' ').title(),
                    'Accuracy': metrics['accuracy'],
                    'Precision': metrics['precision'],
                    'Recall': metrics['recall'],
                    'F1-Score': metrics['f1_score']
                })

        df_ensemble = pd.DataFrame(ensemble_comparison_data)

        # Create subplot for ensemble comparison
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

        for idx, metric in enumerate(metrics_to_plot):
            ax = axes[idx // 2, idx % 2]
            sns.barplot(data=df_ensemble, x='Feature', y=metric, hue='Model', ax=ax)
            ax.set_title(f'Ensemble Models {metric} Comparison', fontsize=14, fontweight='bold')
            ax.set_ylim(0, 1)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/visualizations/comparisons/ensemble_models_comparison.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

        # 3. Best Model Comparison (Baseline vs Best Ensemble for each feature)
        best_comparison_data = []

        for feature_type in self.baseline_metrics:
            # Baseline performance
            baseline_acc = self.baseline_metrics[feature_type]['accuracy']
            best_comparison_data.append({
                'Feature': feature_type.upper(),
                'Model': 'Baseline CNN',
                'Accuracy': baseline_acc
            })

            # Best ensemble performance
            if feature_type in self.ensemble_metrics:
                best_ensemble_acc = max([self.ensemble_metrics[feature_type][model]['accuracy']
                                         for model in self.ensemble_metrics[feature_type]])
                best_ensemble_model = max(self.ensemble_metrics[feature_type].items(),
                                          key=lambda x: x[1]['accuracy'])[0]
                best_comparison_data.append({
                    'Feature': feature_type.upper(),
                    'Model': f'Best Ensemble ({best_ensemble_model.replace("_", " ").title()})',
                    'Accuracy': best_ensemble_acc
                })

        df_best = pd.DataFrame(best_comparison_data)

        plt.figure(figsize=(14, 8))
        sns.barplot(data=df_best, x='Feature', y='Accuracy', hue='Model')
        plt.title('Baseline CNN vs Best Ensemble Model Comparison', fontsize=16, fontweight='bold')
        plt.ylabel('Accuracy', fontsize=14)
        plt.xlabel('Feature Type', fontsize=14)
        plt.ylim(0, 1)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # Add value labels
        for i, (feature, group) in enumerate(df_best.groupby('Feature')):
            for j, (_, row) in enumerate(group.iterrows()):
                plt.text(i + (j - 0.5) * 0.4, row['Accuracy'] + 0.01,
                         f'{row["Accuracy"]:.3f}', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/visualizations/comparisons/baseline_vs_best_ensemble.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

        # 4. Per-class performance comparison
        self.create_per_class_comparison()

        # 5. Create summary metrics table
        self.create_summary_table()

    def create_per_class_comparison(self):
        """Create per-class performance comparison"""
        fig, axes = plt.subplots(1, 3, figsize=(24, 8))
        metrics_types = ['precision', 'recall', 'f1-score']

        for idx, metric_type in enumerate(metrics_types):
            comparison_data = []

            # Add baseline CNN data
            for feature_type in self.baseline_metrics:
                for class_name in self.label_encoder.classes_:
                    if class_name in self.baseline_metrics[feature_type]['classification_report']:
                        comparison_data.append({
                            'Feature': feature_type.upper(),
                            'Class': class_name,
                            'Model': 'Baseline CNN',
                            'Score': self.baseline_metrics[feature_type]['classification_report'][class_name][
                                metric_type]
                        })

            # Add best ensemble data
            for feature_type in self.ensemble_metrics:
                # Find best ensemble model for this feature
                best_model = max(self.ensemble_metrics[feature_type].items(),
                                 key=lambda x: x[1]['accuracy'])[0]

                for class_name in self.label_encoder.classes_:
                    if class_name in self.ensemble_metrics[feature_type][best_model]['classification_report']:
                        comparison_data.append({
                            'Feature': feature_type.upper(),
                            'Class': class_name,
                            'Model': f'Best Ensemble ({best_model.replace("_", " ").title()})',
                            'Score':
                                self.ensemble_metrics[feature_type][best_model]['classification_report'][class_name][
                                    metric_type]
                        })

            df_metric = pd.DataFrame(comparison_data)
            sns.barplot(data=df_metric, x='Class', y='Score', hue='Model', ax=axes[idx])
            axes[idx].set_title(f'{metric_type.capitalize()} Comparison by Class', fontsize=14, fontweight='bold')
            axes[idx].set_ylim(0, 1)
            axes[idx].tick_params(axis='x', rotation=45)
            axes[idx].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/visualizations/comparisons/per_class_metrics_comparison.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

    def create_summary_table(self):
        """Create summary metrics table"""
        summary_data = []

        # Baseline CNN results
        for feature_type in self.baseline_metrics:
            metrics = self.baseline_metrics[feature_type]
            summary_data.append({
                'Feature_Type': feature_type.upper(),
                'Model': 'Baseline CNN',
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'F1_Score': f"{metrics['f1_score']:.4f}"
            })

        # Ensemble results
        for feature_type in self.ensemble_metrics:
            for model_name in self.ensemble_metrics[feature_type]:
                metrics = self.ensemble_metrics[feature_type][model_name]
                summary_data.append({
                    'Feature_Type': feature_type.upper(),
                    'Model': f"{model_name.replace('_', ' ').title()}",
                    'Accuracy': f"{metrics['accuracy']:.4f}",
                    'Precision': f"{metrics['precision']:.4f}",
                    'Recall': f"{metrics['recall']:.4f}",
                    'F1_Score': f"{metrics['f1_score']:.4f}"
                })

        df_summary = pd.DataFrame(summary_data)

        # Save summary table
        df_summary.to_csv(f'{self.results_dir}/metrics/comparisons/summary_metrics.csv', index=False)

        # Create visual summary table
        fig, ax = plt.subplots(figsize=(16, 10))
        ax.axis('tight')
        ax.axis('off')

        table = ax.table(cellText=df_summary.values, colLabels=df_summary.columns,
                         cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)

        # Color code the table
        for i in range(len(df_summary.columns)):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')

        # Highlight best performers
        for col_idx, col in enumerate(['Accuracy', 'Precision', 'Recall', 'F1_Score']):
            if col in df_summary.columns:
                max_val = df_summary[col].astype(float).max()
                for row_idx, val in enumerate(df_summary[col]):
                    if float(val) == max_val:
                        table[(row_idx + 1, col_idx)].set_facecolor('#90EE90')

        plt.title('Complete Performance Summary Table', fontsize=16, fontweight='bold', pad=20)
        plt.savefig(f'{self.results_dir}/visualizations/comparisons/summary_table.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

    def visualize_embeddings(self):
        """Visualize embeddings using dimensionality reduction"""
        print("Creating embedding visualizations...")

        from sklearn.manifold import TSNE
        from sklearn.decomposition import PCA

        for feature_type in self.embeddings:
            embeddings = self.embeddings[feature_type]['train']
            labels = self.embeddings[feature_type]['train_labels']

            # PCA visualization
            pca = PCA(n_components=2)
            embeddings_pca = pca.fit_transform(embeddings)

            plt.figure(figsize=(12, 5))

            # PCA plot
            plt.subplot(1, 2, 1)
            scatter = plt.scatter(embeddings_pca[:, 0], embeddings_pca[:, 1],
                                  c=[self.label_encoder.transform([label])[0] for label in labels],
                                  cmap='viridis', alpha=0.7)
            plt.title(f'{feature_type.upper()} Embeddings - PCA', fontsize=14, fontweight='bold')
            plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
            plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
            plt.colorbar(scatter, ticks=range(len(self.label_encoder.classes_)),
                         label='Classes')
            plt.grid(True, alpha=0.3)

            # t-SNE visualization
            tsne = TSNE(n_components=2, random_state=42, perplexity=30)
            embeddings_tsne = tsne.fit_transform(embeddings)

            plt.subplot(1, 2, 2)
            scatter = plt.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1],
                                  c=[self.label_encoder.transform([label])[0] for label in labels],
                                  cmap='viridis', alpha=0.7)
            plt.title(f'{feature_type.upper()} Embeddings - t-SNE', fontsize=14, fontweight='bold')
            plt.xlabel('t-SNE 1')
            plt.ylabel('t-SNE 2')
            plt.colorbar(scatter, ticks=range(len(self.label_encoder.classes_)),
                         label='Classes')
            plt.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(f'{self.results_dir}/visualizations/comparisons/{feature_type}_embeddings_visualization.png',
                        dpi=300, bbox_inches='tight')
            plt.close()

    def generate_final_report(self):
        """Generate comprehensive final report"""
        print("Generating final report...")

        # Find best performing models
        best_baseline = max(self.baseline_metrics.items(), key=lambda x: x[1]['accuracy'])

        best_ensemble_overall = None
        best_ensemble_score = 0

        for feature_type in self.ensemble_metrics:
            for model_name in self.ensemble_metrics[feature_type]:
                score = self.ensemble_metrics[feature_type][model_name]['accuracy']
                if score > best_ensemble_score:
                    best_ensemble_score = score
                    best_ensemble_overall = (feature_type, model_name)

        report = {
            'dataset_info': {
                'total_samples': len(self.audio_data),
                'classes': list(self.label_encoder.classes_),
                'sample_rate': self.sample_rate,
                'duration': self.duration,
                'class_distribution': pd.Series(self.labels).value_counts().to_dict()
            },
            'best_baseline_model': {
                'feature_type': best_baseline[0],
                'accuracy': best_baseline[1]['accuracy'],
                'precision': best_baseline[1]['precision'],
                'recall': best_baseline[1]['recall'],
                'f1_score': best_baseline[1]['f1_score']
            },
            'best_ensemble_model': {
                'feature_type': best_ensemble_overall[0],
                'model_name': best_ensemble_overall[1],
                'accuracy': self.ensemble_metrics[best_ensemble_overall[0]][best_ensemble_overall[1]]['accuracy'],
                'precision': self.ensemble_metrics[best_ensemble_overall[0]][best_ensemble_overall[1]]['precision'],
                'recall': self.ensemble_metrics[best_ensemble_overall[0]][best_ensemble_overall[1]]['recall'],
                'f1_score': self.ensemble_metrics[best_ensemble_overall[0]][best_ensemble_overall[1]]['f1_score']
            },
            'improvement': {
                'accuracy_improvement': best_ensemble_score - best_baseline[1]['accuracy'],
                'relative_improvement': ((best_ensemble_score - best_baseline[1]['accuracy']) / best_baseline[1][
                    'accuracy']) * 100
            }
        }

        # Save report
        with open(f'{self.results_dir}/final_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)

        # Print summary
        print("\n" + "=" * 80)
        print("FINAL PERFORMANCE REPORT")
        print("=" * 80)
        print(
            f"Dataset: {report['dataset_info']['total_samples']} samples across {len(report['dataset_info']['classes'])} classes")
        print(f"Classes: {report['dataset_info']['classes']}")
        print(f"\nBest Baseline CNN:")
        print(f"  Feature Type: {report['best_baseline_model']['feature_type'].upper()}")
        print(f"  Accuracy: {report['best_baseline_model']['accuracy']:.4f}")
        print(f"  Precision: {report['best_baseline_model']['precision']:.4f}")
        print(f"  Recall: {report['best_baseline_model']['recall']:.4f}")
        print(f"  F1-Score: {report['best_baseline_model']['f1_score']:.4f}")

        print(f"\nBest Ensemble Model:")
        print(f"  Feature Type: {report['best_ensemble_model']['feature_type'].upper()}")
        print(f"  Model: {report['best_ensemble_model']['model_name'].replace('_', ' ').title()}")
        print(f"  Accuracy: {report['best_ensemble_model']['accuracy']:.4f}")
        print(f"  Precision: {report['best_ensemble_model']['precision']:.4f}")
        print(f"  Recall: {report['best_ensemble_model']['recall']:.4f}")
        print(f"  F1-Score: {report['best_ensemble_model']['f1_score']:.4f}")

        print(f"\nImprovement over Baseline:")
        print(f"  Accuracy Improvement: +{report['improvement']['accuracy_improvement']:.4f}")
        print(f"  Relative Improvement: +{report['improvement']['relative_improvement']:.2f}%")
        print("=" * 80)

    def run_complete_pipeline(self):
        """Run the complete CNN + Ensemble pipeline"""
        print("Starting Complete CNN + Ensemble Audio Classification Pipeline...")

        # Step 1: Load audio data
        self.load_audio_data()

        # Step 2: Encode labels
        y_encoded = self.label_encoder.fit_transform(self.labels)
        y_categorical = tf.keras.utils.to_categorical(y_encoded)
        num_classes = len(self.label_encoder.classes_)

        print(f"Number of classes: {num_classes}")
        print(f"Classes: {list(self.label_encoder.classes_)}")

        # Step 3: Extract features
        features = self.extract_all_features()

        # Step 4: Prepare data for each feature type
        feature_shapes = {
            'mfcc': (20, 65, 1),
            'mel': (128, 66, 1),
            'log_mel': (128, 128, 1)
        }

        # Step 5: Train baseline CNN and ensemble models for each feature type
        for feature_type in features:
            print(f"\n{'=' * 60}")
            print(f"Processing {feature_type.upper()} features")
            print(f"{'=' * 60}")

            # Prepare CNN input
            X = self.prepare_cnn_input(features[feature_type], feature_shapes[feature_type])

            # Train-test split (70-30)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_categorical, test_size=0.3, random_state=42, stratify=y_encoded
            )

            # Further split training data for validation (80-20 of training data)
            X_train_final, X_val, y_train_final, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42,
                stratify=np.argmax(y_train, axis=1)
            )

            print(f"Training set: {X_train_final.shape}, Validation set: {X_val.shape}, Test set: {X_test.shape}")

            # PHASE 1: Train Baseline CNN
            print(f"\n--- PHASE 1: Baseline CNN Training for {feature_type.upper()} ---")
            baseline_model, baseline_history = self.train_baseline_cnn(
                X_train_final, y_train_final, X_val, y_val, feature_type, num_classes
            )

            # Store baseline model and history
            self.baseline_models[feature_type] = baseline_model
            self.baseline_histories[feature_type] = baseline_history

            # Plot baseline training history
            self.plot_training_history(baseline_history, feature_type, "baseline")

            # Evaluate baseline model
            baseline_metrics, baseline_predictions = self.evaluate_baseline_model(
                baseline_model, X_test, y_test, feature_type
            )
            self.baseline_metrics[feature_type] = baseline_metrics
            self.baseline_predictions[feature_type] = baseline_predictions

            print(f"Baseline {feature_type.upper()} CNN Test Accuracy: {baseline_metrics['accuracy']:.4f}")

            # PHASE 2: Train Embedding Extractor and Ensemble Models
            print(f"\n--- PHASE 2: Embedding Extraction and Ensemble Training for {feature_type.upper()} ---")

            # Train embedding extractor
            embedding_model = self.train_embedding_extractor(
                X_train_final, y_train_final, X_val, y_val, feature_type
            )
            self.embedding_models[feature_type] = embedding_model

            # Extract embeddings
            train_embeddings = self.extract_embeddings(embedding_model, X_train, feature_type + "_train")
            test_embeddings = self.extract_embeddings(embedding_model, X_test, feature_type + "_test")

            # Store embeddings for visualization
            self.embeddings[feature_type] = {
                'train': train_embeddings,
                'test': test_embeddings,
                'train_labels': [self.labels[i] for i in range(len(self.labels)) if i < len(train_embeddings)],
                'test_labels': [self.labels[i] for i in range(len(self.labels)) if i >= len(train_embeddings)]
            }

            # Train ensemble models
            y_train_labels = np.argmax(y_train, axis=1)
            y_test_labels = np.argmax(y_test, axis=1)

            ensemble_models = self.train_ensemble_models(
                train_embeddings, y_train_labels, feature_type
            )

            # Evaluate ensemble models
            ensemble_metrics = self.evaluate_ensemble_models(
                ensemble_models, test_embeddings, y_test_labels, feature_type
            )
            self.ensemble_metrics[feature_type] = ensemble_metrics

            # Print ensemble results
            print(f"\nEnsemble Results for {feature_type.upper()}:")
            for model_name, metrics in ensemble_metrics.items():
                print(f"  {model_name.replace('_', ' ').title()}: {metrics['accuracy']:.4f}")

        # Step 6: Create comprehensive visualizations and comparisons
        print(f"\n{'=' * 60}")
        print("Creating Comprehensive Comparisons and Visualizations")
        print(f"{'=' * 60}")

        self.create_comprehensive_comparisons()
        self.visualize_embeddings()

        # Step 7: Save additional data
        splits_data = {
            'label_encoder_classes': list(self.label_encoder.classes_),
            'feature_shapes': feature_shapes,
            'total_samples': len(self.audio_data)
        }

        with open(f'{self.results_dir}/data_splits/pipeline_info.json', 'w') as f:
            json.dump(splits_data, f, indent=2)

        # Save label encoder
        with open(f'{self.results_dir}/models/label_encoder.pkl', 'wb') as f:
            pickle.dump(self.label_encoder, f)

        # Step 8: Generate final comprehensive report
        self.generate_final_report()

        print(f"\n{'=' * 60}")
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"All results saved in: {self.results_dir}")
        print(f"{'=' * 60}")


def main():
    """Main function to run the complete pipeline"""
    # Configuration - Update these paths according to your setup
    DATA_DIR = r"C:\Users\visha\Desktop\Temp\forest_research\Augmented_Datasetv2"  # Update this path to your dataset directory
    RESULTS_DIR = "C:/Users/visha/Desktop/Temp/forest_research/ForNetv2_results(Augmentedv2)"

    print("Initializing CNN + Ensemble Audio Classification Pipeline...")
    print(f"Data Directory: {DATA_DIR}")
    print(f"Results Directory: {RESULTS_DIR}")

    # Initialize classifier
    classifier = CNNEnsembleAudioClassifier(
        data_dir=DATA_DIR,
        results_dir=RESULTS_DIR,
        sample_rate=22050,
        duration=3.0
    )

    try:
        # Run complete pipeline
        classifier.run_complete_pipeline()

        print("\n" + "=" * 60)
        print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"Check the results directory: {RESULTS_DIR}")
        print("Key outputs:")
        print("  - Baseline CNN models and metrics")
        print("  - CNN embeddings and ensemble models")
        print("  - Comprehensive visualizations and comparisons")
        print("  - Final performance report")
        print("=" * 60)

    except Exception as e:
        print(f"\nERROR: {str(e)}")
        print("Please check your data directory path and file structure.")
        print("Expected structure:")
        print("dataset/")
        print("├── Class1/")
        print("│   ├── subdirectory1/")
        print("│   │   ├── audio1.wav")
        print("│   │   └── ...")
        print("│   └── subdirectory2/...")
        print("├── Class2/")
        print("│   ├── subdirectory1/...")
        print("│   └── ...")
        print("└── Class3/")
        print("    ├── subdirectory1/...")
        print("    └── ...")


if __name__ == "__main__":
    main()