import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist, squareform
from scipy.stats import kruskal, chi2_contingency
import librosa
import librosa.display
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
import warnings
from collections import Counter
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from umap import UMAP
import matplotlib.gridspec as gridspec

warnings.filterwarnings('ignore')


class ForNetAudioProcessor:
    """Audio preprocessing and feature extraction class - Exact paper implementation"""

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

    def extract_mfcc(self, audio):
        """Extract MFCC features as per paper specifications (20 x 65)"""
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
        """Extract Mel spectrogram features (128 x 66)"""
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
        """Extract Log-Mel spectrogram features (128 x 128)"""
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

    def build_model(self):
        """Build the exact CNN architecture from the paper"""
        inputs = layers.Input(shape=self.input_shape)

        # First Convolutional Layer: 32 filters, 5x5 kernel
        conv1 = layers.Conv2D(32, (5, 5), activation='relu', padding='same', name='conv1')(inputs)
        pool1 = layers.MaxPooling2D((2, 2), name='pool1')(conv1)

        # Second Convolutional Layer: 64 filters, 4x4 kernel
        conv2 = layers.Conv2D(64, (4, 4), activation='relu', padding='same', name='conv2')(pool1)
        pool2 = layers.MaxPooling2D((2, 2), name='pool2')(conv2)

        # Third Convolutional Layer: 128 filters, 3x3 kernel
        conv3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3')(pool2)

        # Global Average Pooling (for embeddings extraction) - 128 dimensions
        embeddings = layers.GlobalAveragePooling2D(name='embeddings')(conv3)

        # First Dense Layer: 50 neurons
        dense1 = layers.Dense(50, activation='relu', name='dense1')(embeddings)
        dropout1 = layers.Dropout(0.5, name='dropout1')(dense1)

        # Output Layer
        outputs = layers.Dense(self.num_classes, activation='softmax', name='output')(dropout1)

        # Create the main model
        model = models.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        self.model = model
        # Create embedding model for feature extraction
        self.embedding_model = models.Model(inputs=inputs, outputs=embeddings)

        return model

    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        """Train the model"""
        from tensorflow.keras import callbacks
        early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping],
            verbose=1
        )
        return history

    def extract_embeddings(self, X):
        """Extract 128-dimensional embeddings from Global Average Pooling layer"""
        if self.embedding_model is None:
            raise ValueError("Model must be built first")
        return self.embedding_model.predict(X, verbose=0)


class FeatureAnalyzer:
    """Comprehensive feature analysis and visualization"""

    def __init__(self, results_dir):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)

        # Set plotting style
        plt.style.use('default')
        sns.set_palette("husl")

    def analyze_raw_features(self, features_dict, labels, class_names):
        """Analyze raw feature representations before CNN"""
        print("Analyzing raw feature representations...")

        results = {}

        for feature_type, features in features_dict.items():
            print(f"Analyzing {feature_type} features...")

            # Flatten features for analysis
            features_flat = features.reshape(features.shape[0], -1)

            # Basic statistics
            feature_stats = {
                'shape': features.shape,
                'flattened_shape': features_flat.shape,
                'mean': np.mean(features_flat),
                'std': np.std(features_flat),
                'min': np.min(features_flat),
                'max': np.max(features_flat),
                'sparsity': np.mean(features_flat == 0),
                'dynamic_range': np.max(features_flat) - np.min(features_flat)
            }

            # Per-class statistics
            class_stats = {}
            for i, class_name in enumerate(class_names):
                class_mask = labels == i
                if np.any(class_mask):
                    class_features = features_flat[class_mask]
                    class_stats[class_name] = {
                        'count': np.sum(class_mask),
                        'mean': np.mean(class_features),
                        'std': np.std(class_features),
                        'sparsity': np.mean(class_features == 0)
                    }

            results[feature_type] = {
                'statistics': feature_stats,
                'class_statistics': class_stats,
                'features_flat': features_flat
            }

        return results

    def analyze_cnn_embeddings(self, embeddings_dict, labels, class_names):
        """Analyze CNN embeddings after feature extraction"""
        print("Analyzing CNN embeddings...")

        results = {}

        for feature_type, embeddings in embeddings_dict.items():
            print(f"Analyzing {feature_type} embeddings...")

            # Basic statistics
            embedding_stats = {
                'shape': embeddings.shape,
                'dimensions': embeddings.shape[1],
                'mean': np.mean(embeddings),
                'std': np.std(embeddings),
                'min': np.min(embeddings),
                'max': np.max(embeddings),
                'sparsity': np.mean(embeddings == 0),
                'dynamic_range': np.max(embeddings) - np.min(embeddings)
            }

            # Per-class statistics
            class_stats = {}
            for i, class_name in enumerate(class_names):
                class_mask = labels == i
                if np.any(class_mask):
                    class_embeddings = embeddings[class_mask]
                    class_stats[class_name] = {
                        'count': np.sum(class_mask),
                        'mean': np.mean(class_embeddings),
                        'std': np.std(class_embeddings),
                        'centroid': np.mean(class_embeddings, axis=0),
                        'intra_class_distance': np.mean(pdist(class_embeddings)) if len(class_embeddings) > 1 else 0
                    }

            # Inter-class distances
            centroids = np.array([stats['centroid'] for stats in class_stats.values()])
            inter_class_distances = pdist(centroids) if len(centroids) > 1 else np.array([0])

            # Clustering metrics
            clustering_metrics = self.compute_clustering_metrics(embeddings, labels)

            results[feature_type] = {
                'statistics': embedding_stats,
                'class_statistics': class_stats,
                'inter_class_distances': {
                    'mean': np.mean(inter_class_distances),
                    'std': np.std(inter_class_distances),
                    'min': np.min(inter_class_distances),
                    'max': np.max(inter_class_distances)
                },
                'clustering_metrics': clustering_metrics
            }

        return results

    def compute_clustering_metrics(self, features, labels):
        """Compute various clustering quality metrics"""
        metrics = {}

        try:
            # Silhouette Score
            if len(np.unique(labels)) > 1:
                metrics['silhouette_score'] = silhouette_score(features, labels)

            # Davies-Bouldin Score
            metrics['davies_bouldin_score'] = davies_bouldin_score(features, labels)

            # Calinski-Harabasz Score
            from sklearn.metrics import calinski_harabasz_score
            metrics['calinski_harabasz_score'] = calinski_harabasz_score(features, labels)

            # K-means clustering evaluation
            kmeans = KMeans(n_clusters=len(np.unique(labels)), random_state=42)
            kmeans_labels = kmeans.fit_predict(features)
            metrics['kmeans_ari'] = adjusted_rand_score(labels, kmeans_labels)

        except Exception as e:
            print(f"Error computing clustering metrics: {e}")
            metrics = {'error': str(e)}

        return metrics

    def create_dimensionality_reduction_plots(self, features_dict, embeddings_dict, labels, class_names):
        """
        Create dimensionality reduction visualizations for raw features & CNN embeddings
        with sharper t-SNE plots and parameter sweep.
        """
        print("Creating dimensionality reduction plots with enhanced t-SNE clarity...")

        # Colors for each class
        colors = plt.cm.tab10(np.linspace(0, 1, len(class_names)))

        # t-SNE parameter variations for sweep
        tsne_param_sets = [
            dict(perplexity=5, learning_rate=100, n_iter=2000, init='pca', random_state=42),
            dict(perplexity=20, learning_rate=200, n_iter=2000, init='pca', random_state=42),
            dict(perplexity=40, learning_rate=300, n_iter=2000, init='pca', random_state=42)
        ]

        # Loop through t-SNE parameter sets
        for param_idx, tsne_params in enumerate(tsne_param_sets):
            fig = plt.figure(figsize=(26, 18), dpi=300)
            gs = gridspec.GridSpec(4, 6, figure=fig)

            row = 0
            # --- Raw Features ---
            for feature_type, features in features_dict.items():
                features_flat = features.reshape(features.shape[0], -1)

                # t-SNE
                ax_tsne = fig.add_subplot(gs[row, 0])
                tsne = TSNE(n_components=2, **tsne_params)
                features_tsne = tsne.fit_transform(features_flat)
                for i, class_name in enumerate(class_names):
                    mask = labels == i
                    ax_tsne.scatter(features_tsne[mask, 0], features_tsne[mask, 1],
                                    c=[colors[i]], label=class_name, alpha=0.8, s=20, edgecolors='none')
                ax_tsne.set_title(f'{feature_type} Raw - t-SNE', fontsize=12)
                ax_tsne.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, markerscale=1)

                # PCA
                ax_pca = fig.add_subplot(gs[row, 1])
                pca = PCA(n_components=2, random_state=42)
                features_pca = pca.fit_transform(features_flat)
                for i, class_name in enumerate(class_names):
                    mask = labels == i
                    ax_pca.scatter(features_pca[mask, 0], features_pca[mask, 1],
                                   c=[colors[i]], label=class_name, alpha=0.8, s=20, edgecolors='none')
                ax_pca.set_title(f'{feature_type} Raw - PCA\n(Var: {pca.explained_variance_ratio_.sum():.3f})', fontsize=12)

                row += 1

            row = 0
            # --- CNN Embeddings ---
            for feature_type, embeddings in embeddings_dict.items():
                # t-SNE
                ax_tsne = fig.add_subplot(gs[row, 2])
                tsne = TSNE(n_components=2, **tsne_params)
                embeddings_tsne = tsne.fit_transform(embeddings)
                for i, class_name in enumerate(class_names):
                    mask = labels == i
                    ax_tsne.scatter(embeddings_tsne[mask, 0], embeddings_tsne[mask, 1],
                                    c=[colors[i]], label=class_name, alpha=0.8, s=20, edgecolors='none')
                ax_tsne.set_title(f'{feature_type} CNN - t-SNE', fontsize=12)

                # PCA
                ax_pca = fig.add_subplot(gs[row, 3])
                pca = PCA(n_components=2, random_state=42)
                embeddings_pca = pca.fit_transform(embeddings)
                for i, class_name in enumerate(class_names):
                    mask = labels == i
                    ax_pca.scatter(embeddings_pca[mask, 0], embeddings_pca[mask, 1],
                                   c=[colors[i]], label=class_name, alpha=0.8, s=20, edgecolors='none')
                ax_pca.set_title(f'{feature_type} CNN - PCA\n(Var: {pca.explained_variance_ratio_.sum():.3f})', fontsize=12)

                # UMAP
                try:
                    ax_umap = fig.add_subplot(gs[row, 4])
                    umap_reducer = UMAP(n_components=2, random_state=42)
                    embeddings_umap = umap_reducer.fit_transform(embeddings)
                    for i, class_name in enumerate(class_names):
                        mask = labels == i
                        ax_umap.scatter(embeddings_umap[mask, 0], embeddings_umap[mask, 1],
                                        c=[colors[i]], label=class_name, alpha=0.8, s=20, edgecolors='none')
                    ax_umap.set_title(f'{feature_type} CNN - UMAP', fontsize=12)
                except:
                    ax_umap = fig.add_subplot(gs[row, 4])
                    ax_umap.text(0.5, 0.5, 'UMAP\nNot Available', ha='center', va='center', fontsize=12)

                # Distance heatmap
                ax_dist = fig.add_subplot(gs[row, 5])
                centroids = []
                for i in range(len(class_names)):
                    class_mask = labels == i
                    if np.any(class_mask):
                        centroids.append(np.mean(embeddings[class_mask], axis=0))
                if len(centroids) > 1:
                    centroids = np.array(centroids)
                    distance_matrix = squareform(pdist(centroids))
                    im = ax_dist.imshow(distance_matrix, cmap='viridis')
                    ax_dist.set_xticks(range(len(class_names)))
                    ax_dist.set_yticks(range(len(class_names)))
                    ax_dist.set_xticklabels(class_names, rotation=45, fontsize=8)
                    ax_dist.set_yticklabels(class_names, fontsize=8)
                    ax_dist.set_title(f'{feature_type} - Inter-class Distances', fontsize=12)
                    plt.colorbar(im, ax=ax_dist)

                row += 1

            plt.tight_layout()
            save_path = os.path.join(self.results_dir, f'feature_analysis_tsneparamset_{param_idx+1}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved: {save_path}")

    def create_feature_comparison_plots(self, raw_features_analysis, cnn_embeddings_analysis, class_names):
        """Create detailed comparison between raw features and CNN embeddings"""
        print("Creating feature comparison plots...")

        # 1. Statistical Comparison
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        feature_types = list(raw_features_analysis.keys())

        # Mean values comparison
        raw_means = [raw_features_analysis[ft]['statistics']['mean'] for ft in feature_types]
        cnn_means = [cnn_embeddings_analysis[ft]['statistics']['mean'] for ft in feature_types]

        x = np.arange(len(feature_types))
        width = 0.35

        axes[0, 0].bar(x - width / 2, raw_means, width, label='Raw Features', alpha=0.7)
        axes[0, 0].bar(x + width / 2, cnn_means, width, label='CNN Embeddings', alpha=0.7)
        axes[0, 0].set_title('Mean Values Comparison')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(feature_types)
        axes[0, 0].legend()
        axes[0, 0].set_ylabel('Mean Value')

        # Standard deviation comparison
        raw_stds = [raw_features_analysis[ft]['statistics']['std'] for ft in feature_types]
        cnn_stds = [cnn_embeddings_analysis[ft]['statistics']['std'] for ft in feature_types]

        axes[0, 1].bar(x - width / 2, raw_stds, width, label='Raw Features', alpha=0.7)
        axes[0, 1].bar(x + width / 2, cnn_stds, width, label='CNN Embeddings', alpha=0.7)
        axes[0, 1].set_title('Standard Deviation Comparison')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(feature_types)
        axes[0, 1].legend()
        axes[0, 1].set_ylabel('Standard Deviation')

        # Dynamic range comparison
        raw_ranges = [raw_features_analysis[ft]['statistics']['dynamic_range'] for ft in feature_types]
        cnn_ranges = [cnn_embeddings_analysis[ft]['statistics']['dynamic_range'] for ft in feature_types]

        axes[0, 2].bar(x - width / 2, raw_ranges, width, label='Raw Features', alpha=0.7)
        axes[0, 2].bar(x + width / 2, cnn_ranges, width, label='CNN Embeddings', alpha=0.7)
        axes[0, 2].set_title('Dynamic Range Comparison')
        axes[0, 2].set_xticks(x)
        axes[0, 2].set_xticklabels(feature_types)
        axes[0, 2].legend()
        axes[0, 2].set_ylabel('Dynamic Range')

        # Clustering metrics comparison
        metrics = ['silhouette_score', 'davies_bouldin_score', 'calinski_harabasz_score']

        for idx, metric in enumerate(metrics):
            if idx < 3:  # Only plot first 3 metrics
                cnn_metric_values = []
                for ft in feature_types:
                    if metric in cnn_embeddings_analysis[ft]['clustering_metrics']:
                        cnn_metric_values.append(cnn_embeddings_analysis[ft]['clustering_metrics'][metric])
                    else:
                        cnn_metric_values.append(0)

                axes[1, idx].bar(feature_types, cnn_metric_values, alpha=0.7, color='orange')
                axes[1, idx].set_title(f'{metric.replace("_", " ").title()}')
                axes[1, idx].set_ylabel('Score')
                axes[1, idx].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'feature_statistics_comparison.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Dimensionality Impact Analysis
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Original dimensions vs CNN dimensions
        original_dims = []
        cnn_dims = []
        for ft in feature_types:
            original_dims.append(np.prod(raw_features_analysis[ft]['statistics']['flattened_shape'][1:]))
            cnn_dims.append(cnn_embeddings_analysis[ft]['statistics']['dimensions'])

        axes[0].bar(feature_types, original_dims, alpha=0.7, label='Original Features')
        axes[0].set_title('Original Feature Dimensions')
        axes[0].set_ylabel('Number of Dimensions')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].set_yscale('log')

        axes[1].bar(feature_types, cnn_dims, alpha=0.7, color='orange', label='CNN Embeddings')
        axes[1].set_title('CNN Embedding Dimensions')
        axes[1].set_ylabel('Number of Dimensions')
        axes[1].tick_params(axis='x', rotation=45)

        # Compression ratio
        compression_ratios = [orig / cnn for orig, cnn in zip(original_dims, cnn_dims)]
        axes[2].bar(feature_types, compression_ratios, alpha=0.7, color='green')
        axes[2].set_title('Dimensionality Compression Ratio')
        axes[2].set_ylabel('Compression Ratio (Original/CNN)')
        axes[2].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'dimensionality_analysis.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

    def create_interactive_plots(self, embeddings_dict, labels, class_names):
        """Create interactive plots using Plotly"""
        print("Creating interactive visualizations...")

        for feature_type, embeddings in embeddings_dict.items():
            # t-SNE
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings) - 1))
            embeddings_tsne = tsne.fit_transform(embeddings)

            # Create DataFrame for Plotly
            df_tsne = pd.DataFrame({
                'x': embeddings_tsne[:, 0],
                'y': embeddings_tsne[:, 1],
                'class': [class_names[label] for label in labels]
            })

            # Interactive t-SNE plot
            fig = px.scatter(df_tsne, x='x', y='y', color='class',
                             title=f'{feature_type} CNN Embeddings - Interactive t-SNE',
                             hover_data=['class'])

            fig.update_traces(marker=dict(size=8, opacity=0.7))
            fig.update_layout(
                xaxis_title='t-SNE Component 1',
                yaxis_title='t-SNE Component 2',
                legend_title='Class'
            )

            # Save as HTML
            fig.write_html(os.path.join(self.results_dir, f'{feature_type}_interactive_tsne.html'))

            # 3D visualization if possible
            if embeddings.shape[1] >= 3:
                tsne_3d = TSNE(n_components=3, random_state=42, perplexity=min(30, len(embeddings) - 1))
                embeddings_tsne_3d = tsne_3d.fit_transform(embeddings)

                df_tsne_3d = pd.DataFrame({
                    'x': embeddings_tsne_3d[:, 0],
                    'y': embeddings_tsne_3d[:, 1],
                    'z': embeddings_tsne_3d[:, 2],
                    'class': [class_names[label] for label in labels]
                })

                fig_3d = px.scatter_3d(df_tsne_3d, x='x', y='y', z='z', color='class',
                                       title=f'{feature_type} CNN Embeddings - Interactive 3D t-SNE',
                                       hover_data=['class'])

                fig_3d.update_traces(marker=dict(size=5, opacity=0.7))
                fig_3d.write_html(os.path.join(self.results_dir, f'{feature_type}_interactive_3d_tsne.html'))

    def generate_comprehensive_report(self, raw_analysis, cnn_analysis, class_names):
        """Generate a comprehensive analysis report"""
        print("Generating comprehensive analysis report...")

        report = []
        report.append("# ForNet Feature Analysis Report\n")
        report.append("## Executive Summary\n")

        feature_types = list(raw_analysis.keys())

        # Summary statistics
        report.append("### Dataset Overview")
        total_samples = sum([raw_analysis[ft]['class_statistics'][cn]['count']
                             for ft in feature_types for cn in class_names
                             if cn in raw_analysis[ft]['class_statistics']]) // len(feature_types)
        report.append(f"- Total samples: {total_samples}")
        report.append(f"- Number of classes: {len(class_names)}")
        report.append(f"- Feature types analyzed: {', '.join(feature_types)}\n")

        # Class distribution
        report.append("### Class Distribution")
        for class_name in class_names:
            count = raw_analysis[feature_types[0]]['class_statistics'].get(class_name, {}).get('count', 0)
            report.append(f"- {class_name}: {count} samples")
        report.append("\n")

        # Feature analysis
        report.append("## Raw Features Analysis\n")
        for feature_type in feature_types:
            stats = raw_analysis[feature_type]['statistics']
            report.append(f"### {feature_type} Features")
            report.append(f"- Original shape: {stats['shape']}")
            report.append(f"- Flattened dimensions: {stats['flattened_shape'][1]}")
            report.append(f"- Mean: {stats['mean']:.6f}")
            report.append(f"- Std: {stats['std']:.6f}")
            report.append(f"- Sparsity: {stats['sparsity']:.3%}")
            report.append(f"- Dynamic range: {stats['dynamic_range']:.6f}\n")

        # CNN embeddings analysis
        report.append("## CNN Embeddings Analysis\n")
        for feature_type in feature_types:
            stats = cnn_analysis[feature_type]['statistics']
            clustering = cnn_analysis[feature_type]['clustering_metrics']
            inter_class = cnn_analysis[feature_type]['inter_class_distances']

            report.append(f"### {feature_type} CNN Embeddings")
            report.append(f"- Embedding dimensions: {stats['dimensions']}")
            report.append(f"- Mean: {stats['mean']:.6f}")
            report.append(f"- Std: {stats['std']:.6f}")
            report.append(f"- Sparsity: {stats['sparsity']:.3%}")
            report.append(f"- Dynamic range: {stats['dynamic_range']:.6f}")

            if 'silhouette_score' in clustering:
                report.append(f"- Silhouette Score: {clustering['silhouette_score']:.4f}")
            if 'davies_bouldin_score' in clustering:
                report.append(f"- Davies-Bouldin Score: {clustering['davies_bouldin_score']:.4f}")
            if 'calinski_harabasz_score' in clustering:
                report.append(f"- Calinski-Harabasz Score: {clustering['calinski_harabasz_score']:.4f}")

            report.append(f"- Mean inter-class distance: {inter_class['mean']:.4f}")
            report.append(f"- Std inter-class distance: {inter_class['std']:.4f}\n")

        # Comparison analysis
        report.append("## Feature Transformation Impact\n")
        for feature_type in feature_types:
            raw_dims = np.prod(raw_analysis[feature_type]['statistics']['flattened_shape'][1:])
            cnn_dims = cnn_analysis[feature_type]['statistics']['dimensions']
            compression = raw_dims / cnn_dims

            report.append(f"### {feature_type} Transformation")
            report.append(f"- Dimensionality reduction: {raw_dims:,} → {cnn_dims} ({compression:.1f}x compression)")

            raw_mean = raw_analysis[feature_type]['statistics']['mean']
            cnn_mean = cnn_analysis[feature_type]['statistics']['mean']
            mean_change = ((cnn_mean - raw_mean) / raw_mean) * 100 if raw_mean != 0 else 0

            raw_std = raw_analysis[feature_type]['statistics']['std']
            cnn_std = cnn_analysis[feature_type]['statistics']['std']
            std_change = ((cnn_std - raw_std) / raw_std) * 100 if raw_std != 0 else 0

            report.append(f"- Mean value change: {mean_change:+.1f}%")
            report.append(f"- Std deviation change: {std_change:+.1f}%\n")

        # Best performing features
        report.append("## Performance Recommendations\n")
        best_silhouette = ""
        best_davies = ""
        best_calinski = ""

        max_silhouette = -1
        min_davies = float('inf')
        max_calinski = -1

        for feature_type in feature_types:
            clustering = cnn_analysis[feature_type]['clustering_metrics']
            if 'silhouette_score' in clustering and clustering['silhouette_score'] > max_silhouette:
                max_silhouette = clustering['silhouette_score']
                best_silhouette = feature_type
            if 'davies_bouldin_score' in clustering and clustering['davies_bouldin_score'] < min_davies:
                min_davies = clustering['davies_bouldin_score']
                best_davies = feature_type
            if 'calinski_harabasz_score' in clustering and clustering['calinski_harabasz_score'] > max_calinski:
                max_calinski = clustering['calinski_harabasz_score']
                best_calinski = feature_type

        if best_silhouette:
            report.append(f"- Best clustering separation (Silhouette): {best_silhouette} ({max_silhouette:.4f})")
        if best_davies:
            report.append(f"- Best cluster compactness (Davies-Bouldin): {best_davies} ({min_davies:.4f})")
        if best_calinski:
            report.append(f"- Best cluster definition (Calinski-Harabasz): {best_calinski} ({max_calinski:.4f})")

        # Save report
        report_text = "\n".join(report)
        with open(os.path.join(self.results_dir, 'analysis_report.md'), 'w', encoding='utf-8') as f:
            f.write(report_text)

        print("Analysis report saved as 'analysis_report.md'")
        return report_text


def load_data_from_directory(source_dir, processor):
    """Load all audio files from directory structure"""
    print("Loading data from directory structure...")

    # Get class names from subdirectories
    class_names = [d for d in os.listdir(source_dir)
                   if os.path.isdir(os.path.join(source_dir, d))]
    class_names.sort()

    print(f"Found classes: {class_names}")

    file_paths = []
    labels = []

    # Load file paths and labels
    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(source_dir, class_name)
        audio_files = [f for f in os.listdir(class_dir)
                       if f.lower().endswith(('.wav', '.mp3', '.flac', '.ogg'))]

        # Limit samples if needed (e.g., for Car_Sound class)
        if class_name == "Car_Sound":
            audio_files = audio_files[:450]

        for audio_file in audio_files:
            file_path = os.path.join(class_dir, audio_file)
            file_paths.append(file_path)
            labels.append(class_idx)

        print(f"Class '{class_name}': {len(audio_files)} files")

    print(f"Total files loaded: {len(file_paths)}")
    return file_paths, np.array(labels), class_names


def extract_all_features(file_paths, labels, processor):
    """Extract all three types of features"""
    print("Extracting all feature types...")

    features_dict = {'MFCC': [], 'Mel': [], 'LogMel': []}
    valid_labels = []

    for i, file_path in enumerate(file_paths):
        audio = processor.load_audio(file_path, duration=5)
        if audio is not None:
            # Extract MFCC
            mfcc = processor.extract_mfcc(audio)
            features_dict['MFCC'].append(mfcc.reshape(20, 65, 1))

            # Extract Mel spectrogram
            mel = processor.extract_mel_spectrogram(audio)
            features_dict['Mel'].append(mel.reshape(128, 66, 1))

            # Extract Log-Mel spectrogram
            log_mel = processor.extract_log_mel_spectrogram(audio)
            features_dict['LogMel'].append(log_mel.reshape(128, 128, 1))

            valid_labels.append(labels[i])

        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(file_paths)} files")

    # Convert to numpy arrays
    for feature_type in features_dict:
        features_dict[feature_type] = np.array(features_dict[feature_type])
        print(f"{feature_type} features shape: {features_dict[feature_type].shape}")

    return features_dict, np.array(valid_labels)


def train_fornet_and_extract_embeddings(features_dict, labels, class_names):
    """Train ForNet CNN for each feature type and extract embeddings"""
    print("Training ForNet CNN and extracting embeddings...")

    embeddings_dict = {}
    models_dict = {}

    num_classes = len(class_names)

    for feature_type, features in features_dict.items():
        print(f"\nTraining ForNet for {feature_type} features...")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42, stratify=labels
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )

        # Convert labels to categorical
        y_train_cat = to_categorical(y_train, num_classes)
        y_val_cat = to_categorical(y_val, num_classes)

        print(f"Training set: {X_train.shape}")
        print(f"Validation set: {X_val.shape}")
        print(f"Test set: {X_test.shape}")

        # Create and train model
        input_shape = features.shape[1:]
        fornet = ForNetCNN(input_shape, num_classes)
        model = fornet.build_model()

        print(f"Training model for {feature_type}...")
        history = fornet.train(X_train, y_train_cat, X_val, y_val_cat, epochs=50, batch_size=32)

        # Extract embeddings from test set
        embeddings = fornet.extract_embeddings(X_test)

        print(f"{feature_type} embeddings shape: {embeddings.shape}")
        print(f"Embeddings statistics - Mean: {np.mean(embeddings):.6f}, Std: {np.std(embeddings):.6f}")

        embeddings_dict[feature_type] = embeddings
        models_dict[feature_type] = fornet

        # Store test labels for this feature type (should be same for all)
        if 'test_labels' not in locals():
            test_labels = y_test

    return embeddings_dict, test_labels, models_dict


def main(source_directory, results_directory):
    """Main execution function focusing on feature analysis"""

    print("=== ForNet Feature Analysis System ===")
    print(f"Source directory: {source_directory}")
    print(f"Results directory: {results_directory}")

    # Configure TensorFlow
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("GPU available and configured")
    else:
        print("Running on CPU")

    # Initialize components
    processor = ForNetAudioProcessor()
    analyzer = FeatureAnalyzer(results_directory)

    # Load data
    file_paths, labels, class_names = load_data_from_directory(source_directory, processor)
    print(f"Loaded {len(file_paths)} files from {len(class_names)} classes")

    # Extract all feature types
    features_dict, valid_labels = extract_all_features(file_paths, labels, processor)

    # Analyze raw features
    print("\n" + "=" * 50)
    print("ANALYZING RAW FEATURES")
    print("=" * 50)
    raw_features_analysis = analyzer.analyze_raw_features(features_dict, valid_labels, class_names)

    # Train CNN and extract embeddings
    print("\n" + "=" * 50)
    print("TRAINING CNN AND EXTRACTING EMBEDDINGS")
    print("=" * 50)
    embeddings_dict, test_labels, models_dict = train_fornet_and_extract_embeddings(
        features_dict, valid_labels, class_names
    )

    # Analyze CNN embeddings
    print("\n" + "=" * 50)
    print("ANALYZING CNN EMBEDDINGS")
    print("=" * 50)
    cnn_embeddings_analysis = analyzer.analyze_cnn_embeddings(embeddings_dict, test_labels, class_names)

    # Create comprehensive visualizations
    print("\n" + "=" * 50)
    print("CREATING VISUALIZATIONS")
    print("=" * 50)

    # Use subset of data for visualization to match embedding dimensions
    features_subset = {}
    for feature_type in features_dict:
        # Get the same test indices used for embeddings
        X_train, X_test, y_train, y_test = train_test_split(
            features_dict[feature_type], valid_labels, test_size=0.2,
            random_state=42, stratify=valid_labels
        )
        features_subset[feature_type] = X_test

    analyzer.create_dimensionality_reduction_plots(features_subset, embeddings_dict, test_labels, class_names)
    analyzer.create_feature_comparison_plots(raw_features_analysis, cnn_embeddings_analysis, class_names)
    analyzer.create_interactive_plots(embeddings_dict, test_labels, class_names)

    # Generate comprehensive report
    report = analyzer.generate_comprehensive_report(raw_features_analysis, cnn_embeddings_analysis, class_names)

    # Print summary
    print("\n" + "=" * 70)
    print("ANALYSIS SUMMARY")
    print("=" * 70)

    print("\nRaw Features Summary:")
    for feature_type in features_dict:
        stats = raw_features_analysis[feature_type]['statistics']
        print(f"{feature_type}:")
        print(f"  - Shape: {stats['shape']}")
        print(f"  - Flattened dimensions: {stats['flattened_shape'][1]}")
        print(f"  - Mean: {stats['mean']:.6f}")
        print(f"  - Std: {stats['std']:.6f}")
        print(f"  - Sparsity: {stats['sparsity']:.3%}")

    print("\nCNN Embeddings Summary:")
    for feature_type in embeddings_dict:
        stats = cnn_embeddings_analysis[feature_type]['statistics']
        clustering = cnn_embeddings_analysis[feature_type]['clustering_metrics']
        print(f"{feature_type}:")
        print(f"  - Embedding dimensions: {stats['dimensions']}")
        print(f"  - Mean: {stats['mean']:.6f}")
        print(f"  - Std: {stats['std']:.6f}")
        print(f"  - Sparsity: {stats['sparsity']:.3%}")
        if 'silhouette_score' in clustering:
            print(f"  - Silhouette Score: {clustering['silhouette_score']:.4f}")

    print("\nDimensionality Reduction:")
    for feature_type in features_dict:
        raw_dims = np.prod(raw_features_analysis[feature_type]['statistics']['flattened_shape'][1:])
        cnn_dims = cnn_embeddings_analysis[feature_type]['statistics']['dimensions']
        compression = raw_dims / cnn_dims
        print(f"{feature_type}: {raw_dims:,} → {cnn_dims} ({compression:.1f}x compression)")

    print(f"\nAll results saved in: {results_directory}")
    print("Generated files:")
    print("- feature_analysis_tsneparamset_1.png (t-SNE with perplexity=5)")
    print("- feature_analysis_tsneparamset_2.png (t-SNE with perplexity=20)")
    print("- feature_analysis_tsneparamset_3.png (t-SNE with perplexity=40)")
    print("- feature_statistics_comparison.png")
    print("- dimensionality_analysis.png")
    print("- *_interactive_tsne.html (Interactive plots)")
    print("- analysis_report.md (Detailed report)")

    return raw_features_analysis, cnn_embeddings_analysis, embeddings_dict


# Example usage
if __name__ == "__main__":
    # Set your paths here
    SOURCE_DIRECTORY = r"C:\Users\visha\Desktop\new_studies\FSM5"  # Directory with class subdirectories
    RESULTS_DIRECTORY = "C:/Users/visha/Desktop/new_studies/results_CNN_embeddings_analysisv2"  # Where to save results

    # Run the analysis
    try:
        raw_analysis, cnn_analysis, embeddings = main(
            source_directory=SOURCE_DIRECTORY,
            results_directory=RESULTS_DIRECTORY
        )

        print("\nFeature analysis completed successfully!")

    except Exception as e:
        print(f"Error during execution: {str(e)}")
        import traceback
        traceback.print_exc()

"""
COMPREHENSIVE FORNET FEATURE ANALYSIS SCRIPT

This script provides extensive analysis of features before and after CNN processing:

FEATURES:
1. Exact ForNet CNN architecture implementation
2. Raw feature extraction and analysis (MFCC, Mel, Log-Mel)
3. CNN embeddings extraction (128-dimensional from GAP layer)
4. Comprehensive statistical analysis
5. Multiple dimensionality reduction visualizations (t-SNE, PCA, UMAP)
6. Enhanced t-SNE with parameter sweep for better cluster definition
7. Clustering quality metrics
8. Interactive visualizations
9. Detailed comparison reports

OUTPUT VISUALIZATIONS:
- feature_analysis_tsneparamset_1.png: t-SNE with perplexity=5, learning_rate=100
- feature_analysis_tsneparamset_2.png: t-SNE with perplexity=20, learning_rate=200
- feature_analysis_tsneparamset_3.png: t-SNE with perplexity=40, learning_rate=300
- feature_statistics_comparison.png: Statistical comparisons
- dimensionality_analysis.png: Dimensionality reduction analysis
- Interactive HTML plots for exploration
- Comprehensive markdown report

ANALYSIS METRICS:
- Silhouette Score (cluster separation)
- Davies-Bouldin Score (cluster compactness)
- Calinski-Harabasz Score (cluster definition)
- Inter-class distances
- Intra-class distances
- Statistical properties (mean, std, sparsity, dynamic range)
- Dimensionality compression ratios

USAGE:
1. Set SOURCE_DIRECTORY to your dataset path
2. Set RESULTS_DIRECTORY for outputs
3. Run the script
4. Check generated visualizations and report
"""