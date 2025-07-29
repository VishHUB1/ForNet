import os
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras
layers = keras.layers
models = keras.models
callbacks = keras.callbacks
import pickle
import json
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')


class ForNetAudioClassifier:
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
        self.models = {}
        self.histories = {}
        self.predictions = {}
        self.metrics = {}

    def create_results_structure(self):
        """Create organized results directory structure"""
        base_path = Path(self.results_dir)
        directories = [
            'models', 'metrics', 'visualizations/features',
            'visualizations/training', 'visualizations/evaluation',
            'visualizations/dataset_analysis', 'data_splits'
        ]

        for dir_path in directories:
            (base_path / dir_path).mkdir(parents=True, exist_ok=True)

    def load_audio_data(self):
        """Load audio files from the directory structure

        Expected structure:
        dataset/
        ├── Animal/
        │   ├── audio1.wav
        │   ├── audio2.wav
        │   └── ...
        ├── Car_Sound/
        │   ├── audio1.wav
        │   ├── audio2.wav
        │   └── ...
        ├── Gunshot/
        │   ├── audio1.wav
        │   └── ...
        ├── Natural_Sound/
        │   ├── audio1.wav
        │   └── ...
        └── Timber_Cutting/
            ├── audio1.wav
            └── ...
        """
        print("Loading audio data...")
        print(f"Dataset directory: {self.data_dir}")

        # The five main classes should be: Animal, Car_Sound, Gunshot, Natural_Sound, Timber_Cutting
        found_classes = []

        for class_name in os.listdir(self.data_dir):
            class_path = os.path.join(self.data_dir, class_name)
            if os.path.isdir(class_path):
                found_classes.append(class_name)
                print(f"Processing class: {class_name}")

                files_found = 0

                # Process all audio files directly in the class directory
                for file_name in os.listdir(class_path):
                    if file_name.endswith(('.wav', '.mp3', '.flac', '.m4a')):
                        file_path = os.path.join(class_path, file_name)
                        try:
                            # Load audio file
                            audio, sr = librosa.load(file_path, sr=self.sample_rate, duration=self.duration)

                            # Pad or truncate to fixed length
                            if len(audio) < self.n_samples:
                                audio = np.pad(audio, (0, self.n_samples - len(audio)), 'constant')
                            else:
                                audio = audio[:self.n_samples]

                            self.audio_data.append(audio)
                            self.labels.append(class_name)  # This assigns the class name directly
                            self.file_paths.append(file_path)
                            files_found += 1

                        except Exception as e:
                            print(f"Error loading {file_path}: {e}")

                print(f"  Found {files_found} audio files")

        print(f"\nSummary:")
        print(f"Found classes: {found_classes}")
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
        plt.figure(figsize=(12, 6))
        class_counts.plot(kind='bar')
        plt.title('Class Distribution in Dataset')
        plt.xlabel('Classes')
        plt.ylabel('Number of Samples')
        plt.xticks(rotation=45)
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

    def visualize_features(self, features, n_samples=3):
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
                axes[0].set_title(f'Waveform - {class_label}')
                axes[0].set_xlabel('Time (s)')
                axes[0].set_ylabel('Amplitude')

                # MFCC
                librosa.display.specshow(features['mfcc'][idx].T, x_axis='time', ax=axes[1])
                axes[1].set_title('MFCC Features')
                axes[1].set_ylabel('MFCC Coefficients')

                # Mel Spectrogram
                librosa.display.specshow(features['mel'][idx].T, x_axis='time', ax=axes[2], y_axis='mel')
                axes[2].set_title('Mel Spectrogram')

                # Log-Mel Spectrogram
                librosa.display.specshow(features['log_mel'][idx].T, x_axis='time', ax=axes[3], y_axis='mel')
                axes[3].set_title('Log-Mel Spectrogram')
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
                from scipy.ndimage import zoom
                zoom_factors = [target_shape[0] / feature.shape[0], target_shape[1] / feature.shape[1]]
                resized = zoom(feature, zoom_factors, order=1)
            else:
                resized = feature

            # Add channel dimension
            resized_features.append(resized.reshape(*target_shape))

        return np.array(resized_features)

    def create_fornet_cnn(self, input_shape, num_classes):
        """Create the ForNet CNN architecture as described"""
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

    def train_model(self, X_train, y_train, X_val, y_val, feature_type, num_classes):
        """Train the ForNet CNN model"""
        print(f"Training {feature_type.upper()} model...")

        # Create model
        model = self.create_fornet_cnn(X_train.shape[1:], num_classes)

        # Compile model
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        # Print model summary
        print(f"\n{feature_type.upper()} Model Architecture:")
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
        model.save(f'{self.results_dir}/models/{feature_type}_model.h5')

        return model, history

    def plot_training_history(self, history, feature_type):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Plot training & validation accuracy
        ax1.plot(history.history['accuracy'], label='Training Accuracy')
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title(f'{feature_type.upper()} - Model Accuracy')
        ax1.set_ylabel('Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.legend()
        ax1.grid(True)

        # Plot training & validation loss
        ax2.plot(history.history['loss'], label='Training Loss')
        ax2.plot(history.history['val_loss'], label='Validation Loss')
        ax2.set_title(f'{feature_type.upper()} - Model Loss')
        ax2.set_ylabel('Loss')
        ax2.set_xlabel('Epoch')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/visualizations/training/{feature_type}_training_history.png', dpi=300,
                    bbox_inches='tight')
        plt.close()

    def evaluate_model(self, model, X_test, y_test, feature_type):
        """Evaluate model and generate metrics"""
        print(f"Evaluating {feature_type.upper()} model...")

        # Predictions
        y_pred_prob = model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_prob, axis=1)
        y_true = np.argmax(y_test, axis=1)

        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        class_report = classification_report(y_true, y_pred, target_names=self.label_encoder.classes_, output_dict=True)
        conf_matrix = confusion_matrix(y_true, y_pred)

        # Store results
        metrics = {
            'accuracy': accuracy,
            'classification_report': class_report,
            'confusion_matrix': conf_matrix.tolist()
        }

        # Save metrics
        with open(f'{self.results_dir}/metrics/{feature_type}_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2, default=str)

        # Plot confusion matrix
        plt.figure(figsize=(12, 10))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.label_encoder.classes_,
                    yticklabels=self.label_encoder.classes_)
        plt.title(f'{feature_type.upper()} - Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/visualizations/evaluation/{feature_type}_confusion_matrix.png', dpi=300,
                    bbox_inches='tight')
        plt.close()

        return metrics, y_pred_prob

    def create_comparison_visualizations(self):
        """Create comparison visualizations between different feature types"""
        print("Creating comparison visualizations...")

        # Accuracy comparison
        accuracies = {feature_type: self.metrics[feature_type]['accuracy']
                      for feature_type in self.metrics}

        plt.figure(figsize=(10, 6))
        feature_types = list(accuracies.keys())
        accuracy_values = list(accuracies.values())

        bars = plt.bar(feature_types, accuracy_values, color=['skyblue', 'lightcoral', 'lightgreen'])
        plt.title('Model Accuracy Comparison Across Feature Types')
        plt.ylabel('Accuracy')
        plt.xlabel('Feature Type')
        plt.ylim(0, 1)

        # Add value labels on bars
        for bar, accuracy in zip(bars, accuracy_values):
            plt.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.01,
                     f'{accuracy:.3f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/visualizations/evaluation/accuracy_comparison.png', dpi=300,
                    bbox_inches='tight')
        plt.close()

        # Per-class performance comparison
        fig, axes = plt.subplots(1, 3, figsize=(20, 8))
        metrics_types = ['precision', 'recall', 'f1-score']

        for idx, metric_type in enumerate(metrics_types):
            data = []
            for feature_type in self.metrics:
                for class_name in self.label_encoder.classes_:
                    if class_name in self.metrics[feature_type]['classification_report']:
                        data.append({
                            'Feature': feature_type.upper(),
                            'Class': class_name,
                            'Score': self.metrics[feature_type]['classification_report'][class_name][metric_type]
                        })

            df_metric = pd.DataFrame(data)
            sns.barplot(data=df_metric, x='Class', y='Score', hue='Feature', ax=axes[idx])
            axes[idx].set_title(f'{metric_type.capitalize()} Comparison')
            axes[idx].set_ylim(0, 1)
            axes[idx].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/visualizations/evaluation/per_class_metrics_comparison.png', dpi=300,
                    bbox_inches='tight')
        plt.close()

    def run_complete_pipeline(self):
        """Run the complete ForNet pipeline"""
        print("Starting ForNet Audio Classification Pipeline...")

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
            'mfcc': (20, 65, 1),  # As specified in the paper
            'mel': (128, 66, 1),  # Mel spectrogram shape
            'log_mel': (128, 128, 1)  # Log-mel spectrogram shape
        }

        # Step 5: Train and evaluate models for each feature type
        for feature_type in features:
            print(f"\n{'=' * 50}")
            print(f"Processing {feature_type.upper()} features")
            print(f"{'=' * 50}")

            # Prepare CNN input
            X = self.prepare_cnn_input(features[feature_type], feature_shapes[feature_type])

            # Train-test split (70-30)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_categorical, test_size=0.3, random_state=42, stratify=y_encoded
            )

            # Further split training data for validation (80-20 of training data)
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42, stratify=np.argmax(y_train, axis=1)
            )

            print(f"Training set: {X_train.shape}, Validation set: {X_val.shape}, Test set: {X_test.shape}")

            # Train model
            model, history = self.train_model(X_train, y_train, X_val, y_val, feature_type, num_classes)

            # Store model and history
            self.models[feature_type] = model
            self.histories[feature_type] = history

            # Plot training history
            self.plot_training_history(history, feature_type)

            # Evaluate model
            metrics, predictions = self.evaluate_model(model, X_test, y_test, feature_type)
            self.metrics[feature_type] = metrics
            self.predictions[feature_type] = predictions

            print(f"{feature_type.upper()} Test Accuracy: {metrics['accuracy']:.4f}")

        # Step 6: Create comparison visualizations
        self.create_comparison_visualizations()

        # Step 7: Save data splits and label encoder
        splits_data = {
            'train_indices': list(range(len(X_train))),
            'test_indices': list(range(len(X_test))),
            'label_encoder_classes': list(self.label_encoder.classes_)
        }

        with open(f'{self.results_dir}/data_splits/train_test_splits.json', 'w') as f:
            json.dump(splits_data, f, indent=2)

        # Save label encoder
        with open(f'{self.results_dir}/models/label_encoder.pkl', 'wb') as f:
            pickle.dump(self.label_encoder, f)

        # Step 8: Generate final summary report
        self.generate_summary_report()

        print(f"\n{'=' * 50}")
        print("Pipeline completed successfully!")
        print(f"All results saved in: {self.results_dir}")
        print(f"{'=' * 50}")

    def generate_summary_report(self):
        """Generate a comprehensive summary report"""
        summary = {
            'dataset_info': {
                'total_samples': len(self.audio_data),
                'classes': list(self.label_encoder.classes_),
                'sample_rate': self.sample_rate,
                'duration': self.duration
            },
            'model_performance': {}
        }

        for feature_type in self.metrics:
            summary['model_performance'][feature_type] = {
                'accuracy': self.metrics[feature_type]['accuracy'],
                'macro_avg': self.metrics[feature_type]['classification_report']['macro avg'],
                'weighted_avg': self.metrics[feature_type]['classification_report']['weighted avg']
            }

        # Save summary
        with open(f'{self.results_dir}/summary_report.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        # Print summary
        print("\n" + "=" * 60)
        print("FINAL SUMMARY REPORT")
        print("=" * 60)
        print(f"Dataset: {summary['dataset_info']['total_samples']} samples")
        print(f"Classes: {summary['dataset_info']['classes']}")
        print("\nModel Performance:")
        for feature_type, performance in summary['model_performance'].items():
            print(f"  {feature_type.upper()}: {performance['accuracy']:.4f} accuracy")

        # Find best performing model
        best_feature = max(summary['model_performance'].items(),
                           key=lambda x: x[1]['accuracy'])
        print(f"\nBest performing feature type: {best_feature[0].upper()} "
              f"with {best_feature[1]['accuracy']:.4f} accuracy")


def main():
    # Configuration
    DATA_DIR = r"C:\Users\visha\Desktop\Temp\forest_research\FSM5"  # Update this path to your dataset directory
    RESULTS_DIR = "C:/Users/visha/Desktop/Temp/forest_research/FSM5_baseline_results"

    # Initialize classifier
    classifier = ForNetAudioClassifier(
        data_dir=DATA_DIR,
        results_dir=RESULTS_DIR,
        sample_rate=22050,
        duration=3.0
    )

    # Run complete pipeline
    classifier.run_complete_pipeline()


if __name__ == "__main__":
    main()