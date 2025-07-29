import os
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras import layers, models, Input, callbacks
import pickle, json
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')


class ForNetTwoStageClassifier:
    def __init__(self, data_dir, results_dir="results", sample_rate=22050, duration=3.0):
        self.data_dir = data_dir
        self.results_dir = results_dir
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_samples = int(sample_rate * duration)
        self.mfcc_params = {'n_mfcc': 20, 'n_fft': 2048, 'hop_length': 512}
        self.mel_params = {'n_mels': 128, 'n_fft': 2048, 'hop_length': 512}
        self.create_results_structure()
        self.audio_data = []
        self.labels = []
        self.file_paths = []
        self.label_encoder = LabelEncoder()
        self.metrics = {}

    def create_results_structure(self):
        base_path = Path(self.results_dir)
        dirs = [
            'models', 'metrics', 'visualizations/features',
            'visualizations/ensemble', 'visualizations/embeddings',
            'visualizations/dataset_analysis', 'data_splits'
        ]
        for d in dirs:
            (base_path / d).mkdir(parents=True, exist_ok=True)

    def load_audio_data(self):
        for class_name in os.listdir(self.data_dir):
            class_path = os.path.join(self.data_dir, class_name)
            if os.path.isdir(class_path):
                for subdir in os.listdir(class_path):
                    subdir_path = os.path.join(class_path, subdir)
                    if os.path.isdir(subdir_path):
                        for file_name in os.listdir(subdir_path):
                            if file_name.lower().endswith(('.wav', '.mp3', '.flac', '.m4a')):
                                file_path = os.path.join(subdir_path, file_name)
                                try:
                                    audio, _ = librosa.load(file_path, sr=self.sample_rate, duration=self.duration)
                                    if len(audio) < self.n_samples:
                                        audio = np.pad(audio, (0, self.n_samples - len(audio)), 'constant')
                                    else:
                                        audio = audio[:self.n_samples]
                                    self.audio_data.append(audio)
                                    self.labels.append(class_name)
                                    self.file_paths.append(file_path)
                                except Exception as e:
                                    print(f"Error loading {file_path}: {e}")
        assert len(self.audio_data) > 0, "No audio files found in dataset."
        self.analyze_dataset()

    def analyze_dataset(self):
        df = pd.DataFrame({'label': self.labels})
        plt.figure(figsize=(10,6))
        df['label'].value_counts().plot(kind='bar')
        plt.title('Class Distribution in Dataset')
        plt.xlabel('Classes')
        plt.ylabel('Samples')
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/visualizations/dataset_analysis/class_dist.png', dpi=300)
        plt.close()

    def extract_mfcc(self, audio):
        mfcc = librosa.feature.mfcc(y=audio, sr=self.sample_rate, **self.mfcc_params)
        return mfcc.T

    def extract_mel_spectrogram(self, audio):
        mel = librosa.feature.melspectrogram(y=audio, sr=self.sample_rate, **self.mel_params)
        return mel.T

    def extract_log_mel(self, audio):
        mel = librosa.feature.melspectrogram(y=audio, sr=self.sample_rate, **self.mel_params)
        log_mel = librosa.power_to_db(mel, ref=np.max)
        return log_mel.T

    def extract_all_features(self):
        features = {'mfcc': [], 'mel': [], 'log_mel': []}
        for audio in self.audio_data:
            features['mfcc'].append(self.extract_mfcc(audio))
            features['mel'].append(self.extract_mel_spectrogram(audio))
            features['log_mel'].append(self.extract_log_mel(audio))
        for k in features:
            features[k] = np.array(features[k])
        return features

    def prepare_cnn_input(self, features, target_shape):
        from scipy.ndimage import zoom
        X = []
        for f in features:
            if f.shape != target_shape[:2]:
                zooms = [target_shape[0] / f.shape[0], target_shape[1] / f.shape[1]]
                f_resized = zoom(f, zooms, order=1)
            else:
                f_resized = f
            X.append(f_resized.reshape(*target_shape))
        return np.array(X)

    def create_embedding_cnn(self, input_shape, num_classes):
        # Use functional API to define input explicitly and build model
        inputs = Input(shape=input_shape)
        x = layers.Conv2D(32, (5,5), padding='same')(inputs)
        x = layers.MaxPooling2D((2,2))(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(64, (4,4))(x)
        x = layers.MaxPooling2D((2,2))(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(128, (3,3))(x)
        gap = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(50, activation='relu')(gap)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        model = models.Model(inputs=inputs, outputs=outputs)
        # Also return gap layer output tensor for embedding extraction
        return model, gap

    def train_and_extract_embeddings(self, X, y_cat, feature_type, num_classes):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_cat, test_size=0.3, random_state=42,
            stratify=np.argmax(y_cat, axis=1)
        )
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42,
            stratify=np.argmax(y_train, axis=1)
        )
        model, gap_layer = self.create_embedding_cnn(X.shape[1:], num_classes)

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        estop = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        model.fit(X_tr, y_tr, batch_size=32, epochs=100, validation_data=(X_val, y_val), callbacks=[estop], verbose=1)

        model_path = f'{self.results_dir}/models/{feature_type}_emb_cnn.keras'
        model.save(model_path)

        # Embedding extractor model outputs gap_layer
        emb_model = models.Model(inputs=model.input, outputs=gap_layer)

        embeddings = emb_model.predict(X, batch_size=32)
        return embeddings

    def run_ensemble_cv(self, embeddings, y, feature_type, label_names):
        results = {}
        X = embeddings.copy()
        le = LabelEncoder().fit(y)
        Y = le.transform(y)
        n_classes = len(le.classes_)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for clf_name, clf in [
            ('RandomForest', RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)),
            ('XGBoost', XGBClassifier(n_estimators=200, use_label_encoder=False, eval_metric='mlogloss', random_state=42))
        ]:
            accs, rocs = [], []
            folds_reports = []
            for fold, (train_idx, test_idx) in enumerate(skf.split(X, Y), 1):
                clf.fit(X[train_idx], Y[train_idx])
                pred = clf.predict(X[test_idx])
                proba = clf.predict_proba(X[test_idx])
                accs.append(accuracy_score(Y[test_idx], pred))
                folds_reports.append(classification_report(Y[test_idx], pred, target_names=label_names, output_dict=True))
                if n_classes > 2:
                    y_bin = tf.keras.utils.to_categorical(Y[test_idx], n_classes)
                    rocs.append(roc_auc_score(y_bin, proba, multi_class='ovr'))
                else:
                    rocs.append(roc_auc_score(Y[test_idx], proba[:,1]))
            metrics_summary = {
                'acc_mean': float(np.mean(accs)),
                'acc_std': float(np.std(accs)),
                'roc_auc': float(np.mean(rocs)),
                'f1_macro': float(np.mean([r['macro avg']['f1-score'] for r in folds_reports])),
                'folds_reports': folds_reports
            }
            results[clf_name] = metrics_summary
            with open(f'{self.results_dir}/metrics/{feature_type}_{clf_name}_folds.json', 'w') as f:
                json.dump(folds_reports, f, indent=2)
        with open(f'{self.results_dir}/metrics/{feature_type}_ensemble_summary.json', 'w') as f:
            json.dump(results, f, indent=2)
        return results

    def plot_ensemble_metrics(self, summary_all, label_names):
        dfs = []
        for ft, v in summary_all.items():
            for clf, s in v.items():
                dfs.append({
                    'Feature': ft.upper(),
                    'Classifier': clf,
                    'Accuracy': s['acc_mean'],
                    'ROC_AUC': s['roc_auc'],
                    'F1_Macro': s['f1_macro']
                })
        df = pd.DataFrame(dfs)

        plt.figure(figsize=(8,6))
        sns.barplot(data=df, x='Feature', y='Accuracy', hue='Classifier')
        plt.title('Cross-Val Accuracy: Features vs Classifiers')
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/visualizations/ensemble/feature_classifier_accuracy.png', dpi=300)
        plt.close()

        plt.figure(figsize=(8,6))
        sns.barplot(data=df, x='Feature', y='ROC_AUC', hue='Classifier')
        plt.title('Cross-Val ROC_AUC: Features vs Classifiers')
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/visualizations/ensemble/feature_classifier_rocauc.png', dpi=300)
        plt.close()

        plt.figure(figsize=(8,6))
        sns.barplot(data=df, x='Feature', y='F1_Macro', hue='Classifier')
        plt.title('Cross-Val F1-Macro: Features vs Classifiers')
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/visualizations/ensemble/feature_classifier_f1.png', dpi=300)
        plt.close()

    def visualize_embedding_effect(self, before, after, labels, feature_type):
        from sklearn.manifold import TSNE
        le = LabelEncoder().fit(labels)
        before_reshaped = before.reshape(before.shape[0], -1)
        dims = min(100, before_reshaped.shape[1])
        before_2d = TSNE(n_components=2, random_state=42).fit_transform(before_reshaped[:, :dims])
        after_2d = TSNE(n_components=2, random_state=42).fit_transform(after)

        plt.figure(figsize=(12,5))
        plt.subplot(1,2,1)
        sns.scatterplot(x=before_2d[:,0], y=before_2d[:,1], hue=le.transform(labels), palette='tab10', legend=False)
        plt.title('Raw Features (t-SNE)')
        plt.subplot(1,2,2)
        sns.scatterplot(x=after_2d[:,0], y=after_2d[:,1], hue=le.transform(labels), palette='tab10', legend='full')
        plt.title('CNN Embeddings (t-SNE)')
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/visualizations/embeddings/{feature_type}_raw_vs_embedd_tsne.png', dpi=300)
        plt.close()

    def run_pipeline(self):
        self.load_audio_data()
        y = np.array(self.labels)
        y_int = self.label_encoder.fit_transform(y)
        num_classes = len(self.label_encoder.classes_)
        y_cat = tf.keras.utils.to_categorical(y_int)

        features_dict = self.extract_all_features()

        feature_shapes = {
            'mfcc': (20, 65, 1),
            'mel': (128, 66, 1),
            'log_mel': (128, 128, 1)
        }

        summary_all = {}
        for feature_type, features in features_dict.items():
            X = self.prepare_cnn_input(features, feature_shapes[feature_type])
            embeddings = self.train_and_extract_embeddings(X, y_cat, feature_type, num_classes)
            self.visualize_embedding_effect(X, embeddings, y, feature_type)
            ens_summary = self.run_ensemble_cv(embeddings, y, feature_type, self.label_encoder.classes_)
            summary_all[feature_type] = ens_summary

        self.plot_ensemble_metrics(summary_all, self.label_encoder.classes_)
        with open(f'{self.results_dir}/summary_two_stage.json', 'w') as f:
            json.dump(summary_all, f, indent=2)
        print("Pipeline complete. Results and visualizations stored under the results directory.")


def main():
    DATA_DIR = r"C:\Users\visha\Desktop\forest_research\FSC22_classes"  # Adjust to your dataset path
    RESULTS_DIR = "C:/Users/visha/Desktop/forest_research/ForNet_Ensemble_results"  # Adjust output path
    classifier = ForNetTwoStageClassifier(DATA_DIR, RESULTS_DIR, sample_rate=22050, duration=3.0)
    classifier.run_pipeline()


if __name__ == "__main__":
    main()
