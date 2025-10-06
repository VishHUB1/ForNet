import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
from datetime import datetime
from collections import defaultdict, Counter
import warnings
from scipy import stats
from scipy.stats import wilcoxon, friedmanchisquare
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import itertools

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score,
                             precision_recall_curve, roc_curve, auc)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Ensemble Methods
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# XGBoost
try:
    import xgboost as xgb

    XGBOOST_AVAILABLE = True
except ImportError:
    print("XGBoost not available. Install with: pip install xgboost")
    XGBOOST_AVAILABLE = False

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')


class EmbeddingLoader:
    """Load and manage CNN embeddings from pickle files - K-fold only"""

    def __init__(self, embeddings_directory):
        self.embeddings_dir = embeddings_directory
        self.embeddings_data = {}
        self.feature_types = ['MFCC', 'Mel', 'LogMel']

    def load_embeddings(self):
        """Load k-fold embeddings from pickle files only"""
        print("Loading CNN k-fold embeddings...")

        loaded_data = {}

        for feature_type in self.feature_types:
            print(f"Loading {feature_type} k-fold embeddings...")

            # Look for k-fold embedding files only
            pattern = f"embeddings_{feature_type}_kfold_"

            matching_files = [f for f in os.listdir(self.embeddings_dir)
                              if f.startswith(pattern) and f.endswith('.pkl')]

            if not matching_files:
                print(f"Warning: No {feature_type} k-fold embedding files found with pattern '{pattern}'")
                continue

            # Use the most recent file
            latest_file = sorted(matching_files)[-1]
            filepath = os.path.join(self.embeddings_dir, latest_file)

            try:
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                    loaded_data[feature_type] = data
                    print(f"  ✓ Loaded {feature_type} k-fold: {data['embeddings'].shape} embeddings")
            except Exception as e:
                print(f"  ✗ Error loading {filepath}: {e}")

        if not loaded_data:
            raise ValueError("No k-fold embedding files could be loaded!")

        self.embeddings_data = loaded_data
        return loaded_data

    def get_combined_embeddings(self, feature_types=None):
        """Combine embeddings from multiple feature types"""
        if feature_types is None:
            feature_types = list(self.embeddings_data.keys())

        embeddings_list = []
        labels = None
        class_names = None

        for ft in feature_types:
            if ft in self.embeddings_data:
                embeddings_list.append(self.embeddings_data[ft]['embeddings'])
                if labels is None:
                    labels = self.embeddings_data[ft]['labels']
                    class_names = self.embeddings_data[ft]['class_names']

        if not embeddings_list:
            raise ValueError(f"No embeddings found for feature types: {feature_types}")

        combined_embeddings = np.concatenate(embeddings_list, axis=1)

        return combined_embeddings, labels, class_names

    def get_individual_embeddings(self, feature_type):
        """Get embeddings for a single feature type"""
        if feature_type not in self.embeddings_data:
            raise ValueError(f"Feature type '{feature_type}' not loaded")

        data = self.embeddings_data[feature_type]
        return data['embeddings'], data['labels'], data['class_names']


class EnsembleClassifierManager:
    """Manage different ensemble classifiers"""

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.classifiers = {}
        self.setup_classifiers()

    def setup_classifiers(self):
        """Initialize all ensemble classifiers"""

        # Random Forest
        self.classifiers['RandomForest'] = RandomForestClassifier(
            n_estimators=100, random_state=self.random_state, n_jobs=-1
        )

        # XGBoost
        if XGBOOST_AVAILABLE:
            self.classifiers['XGBoost'] = xgb.XGBClassifier(
                n_estimators=100, random_state=self.random_state, eval_metric='logloss'
            )

        # SVM
        self.classifiers['SVM'] = SVC(probability=True, random_state=self.random_state)

        # Logistic Regression
        self.classifiers['LogisticRegression'] = LogisticRegression(
            random_state=self.random_state, max_iter=1000
        )

        # Gradient Boosting
        self.classifiers['GradientBoosting'] = GradientBoostingClassifier(
            n_estimators=100, random_state=self.random_state
        )

    def get_classifier(self, name):
        """Get a specific classifier"""
        return self.classifiers.get(name)

    def get_all_classifiers(self):
        """Get all classifiers"""
        return self.classifiers


class EnsembleEvaluator:
    """Evaluate ensemble methods with comprehensive metrics"""

    def __init__(self, results_manager):
        self.results_manager = results_manager

    def evaluate_single_split(self, classifier, X_train, X_test, y_train, y_test,
                              class_names, classifier_name, feature_name):
        """Evaluate classifier on single train-test split"""

        print(f"Training {classifier_name} on {feature_name}...")
        classifier.fit(X_train, y_train)

        y_pred = classifier.predict(X_test)
        y_pred_proba = None

        try:
            y_pred_proba = classifier.predict_proba(X_test)
        except:
            pass

        metrics = self.calculate_metrics(y_test, y_pred, y_pred_proba, class_names)

        metrics['classifier_name'] = classifier_name
        metrics['feature_name'] = feature_name
        metrics['train_size'] = len(X_train)
        metrics['test_size'] = len(X_test)

        return metrics, y_pred, y_pred_proba

    def evaluate_kfold_cv(self, classifier, X, y, class_names, classifier_name,
                          feature_name, k_folds=10):
        """Evaluate classifier using k-fold cross validation"""

        print(f"Performing {k_folds}-fold CV for {classifier_name} on {feature_name}...")

        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

        fold_results = {}
        all_y_true = []
        all_y_pred = []
        all_y_proba = []
        fold_accuracies = []
        fold_f1_scores = []
        fold_precisions = []
        fold_recalls = []

        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            print(f"  Processing fold {fold + 1}/{k_folds}...")

            X_train_fold, X_test_fold = X[train_idx], X[test_idx]
            y_train_fold, y_test_fold = y[train_idx], y[test_idx]

            classifier.fit(X_train_fold, y_train_fold)

            y_pred_fold = classifier.predict(X_test_fold)

            try:
                y_proba_fold = classifier.predict_proba(X_test_fold)
                all_y_proba.extend(y_proba_fold)
            except:
                y_proba_fold = None

            fold_acc = accuracy_score(y_test_fold, y_pred_fold)
            fold_f1 = f1_score(y_test_fold, y_pred_fold, average='weighted')
            fold_prec = precision_score(y_test_fold, y_pred_fold, average='weighted')
            fold_rec = recall_score(y_test_fold, y_pred_fold, average='weighted')

            fold_accuracies.append(fold_acc)
            fold_f1_scores.append(fold_f1)
            fold_precisions.append(fold_prec)
            fold_recalls.append(fold_rec)

            all_y_true.extend(y_test_fold)
            all_y_pred.extend(y_pred_fold)

            fold_results[f'fold_{fold + 1}'] = {
                'accuracy': float(fold_acc),
                'f1_score': float(fold_f1),
                'precision': float(fold_prec),
                'recall': float(fold_rec),
                'train_size': len(X_train_fold),
                'test_size': len(X_test_fold)
            }

        all_y_proba = np.array(all_y_proba) if all_y_proba else None
        overall_metrics = self.calculate_metrics(all_y_true, all_y_pred, all_y_proba, class_names)

        cv_results = {
            'classifier_name': classifier_name,
            'feature_name': feature_name,
            'k_folds': k_folds,
            'fold_results': fold_results,
            'mean_accuracy': float(np.mean(fold_accuracies)),
            'std_accuracy': float(np.std(fold_accuracies)),
            'mean_f1_score': float(np.mean(fold_f1_scores)),
            'std_f1_score': float(np.std(fold_f1_scores)),
            'mean_precision': float(np.mean(fold_precisions)),
            'std_precision': float(np.std(fold_precisions)),
            'mean_recall': float(np.mean(fold_recalls)),
            'std_recall': float(np.std(fold_recalls)),
            'individual_accuracies': [float(acc) for acc in fold_accuracies],
            'individual_f1_scores': [float(f1) for f1 in fold_f1_scores],
            'overall_metrics': overall_metrics
        }

        return cv_results, all_y_true, all_y_pred, all_y_proba

    def calculate_metrics(self, y_true, y_pred, y_pred_proba, class_names):
        """Calculate comprehensive evaluation metrics"""

        metrics = {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'f1_score': float(f1_score(y_true, y_pred, average='weighted')),
            'precision': float(precision_score(y_true, y_pred, average='weighted')),
            'recall': float(recall_score(y_true, y_pred, average='weighted')),
            'f1_macro': float(f1_score(y_true, y_pred, average='macro')),
            'f1_micro': float(f1_score(y_true, y_pred, average='micro')),
        }

        class_report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
        metrics['classification_report'] = class_report

        if y_pred_proba is not None:
            try:
                metrics['roc_auc_ovr'] = float(roc_auc_score(y_true, y_pred_proba,
                                                             multi_class='ovr', average='weighted'))
                metrics['roc_auc_ovo'] = float(roc_auc_score(y_true, y_pred_proba,
                                                             multi_class='ovo', average='weighted'))
            except:
                pass

        return metrics


class EnhancedVisualizer:
    """Create comprehensive visualizations for ensemble results"""

    def __init__(self, results_manager):
        self.results_manager = results_manager
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                       '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    def plot_dataset_overview(self, embedding_loader):
        """Create comprehensive dataset overview visualizations"""

        fig = plt.figure(figsize=(20, 16))
        fig.suptitle('Dataset Overview and Analysis', fontsize=20, fontweight='bold')

        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)

        # 1. Feature dimensionality comparison
        ax1 = fig.add_subplot(gs[0, :2])
        feature_dims = []
        feature_names = []

        for feature_type in embedding_loader.feature_types:
            if feature_type in embedding_loader.embeddings_data:
                dims = embedding_loader.embeddings_data[feature_type]['embeddings'].shape[1]
                feature_dims.append(dims)
                feature_names.append(feature_type)

        bars = ax1.bar(feature_names, feature_dims, color=['skyblue', 'lightcoral', 'lightgreen'])
        ax1.set_title('Feature Dimensionality Comparison', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Number of Features')
        ax1.grid(True, alpha=0.3)

        # Add value labels on bars
        for bar, dim in zip(bars, feature_dims):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height + 10,
                     f'{dim}', ha='center', va='bottom', fontweight='bold')

        # 2. Class distribution
        ax2 = fig.add_subplot(gs[0, 2:])
        if embedding_loader.embeddings_data:
            first_feature = list(embedding_loader.embeddings_data.keys())[0]
            labels = embedding_loader.embeddings_data[first_feature]['labels']
            class_names = embedding_loader.embeddings_data[first_feature]['class_names']

            class_counts = Counter(labels)
            counts = [class_counts[i] for i in range(len(class_names))]

            wedges, texts, autotexts = ax2.pie(counts, labels=class_names, autopct='%1.1f%%',
                                               colors=plt.cm.Set3(np.linspace(0, 1, len(class_names))))
            ax2.set_title('Class Distribution', fontsize=14, fontweight='bold')

        # 3. Sample distribution across features
        ax3 = fig.add_subplot(gs[1, :2])
        sample_counts = []
        for feature_type in feature_names:
            if feature_type in embedding_loader.embeddings_data:
                count = len(embedding_loader.embeddings_data[feature_type]['embeddings'])
                sample_counts.append(count)

        bars = ax3.bar(feature_names, sample_counts, color=['orange', 'purple', 'brown'])
        ax3.set_title('Sample Count per Feature Type', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Number of Samples')
        ax3.grid(True, alpha=0.3)

        for bar, count in zip(bars, sample_counts):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width() / 2., height + 50,
                     f'{count}', ha='center', va='bottom', fontweight='bold')

        # 4. Feature statistics heatmap
        ax4 = fig.add_subplot(gs[1, 2:])
        stats_data = []
        stats_labels = []

        for feature_type in feature_names:
            if feature_type in embedding_loader.embeddings_data:
                embeddings = embedding_loader.embeddings_data[feature_type]['embeddings']
                stats_data.append([
                    np.mean(embeddings),
                    np.std(embeddings),
                    np.min(embeddings),
                    np.max(embeddings),
                    np.median(embeddings)
                ])
                stats_labels.append(feature_type)

        if stats_data:
            stats_df = pd.DataFrame(stats_data,
                                    columns=['Mean', 'Std', 'Min', 'Max', 'Median'],
                                    index=stats_labels)

            sns.heatmap(stats_df, annot=True, fmt='.3f', cmap='coolwarm', ax=ax4,
                        cbar_kws={'label': 'Value'})
            ax4.set_title('Feature Statistics Heatmap', fontsize=14, fontweight='bold')

        # 5. Combined feature dimensionality
        ax5 = fig.add_subplot(gs[2, :2])
        if len(feature_names) > 1:
            combined_dims = {}
            for i in range(1, len(feature_names) + 1):
                for combo in itertools.combinations(feature_names, i):
                    total_dim = sum(feature_dims[feature_names.index(f)] for f in combo)
                    combined_dims['+'.join(combo)] = total_dim

            combo_names = list(combined_dims.keys())
            combo_dims = list(combined_dims.values())

            bars = ax5.bar(range(len(combo_names)), combo_dims,
                           color=plt.cm.viridis(np.linspace(0, 1, len(combo_names))))
            ax5.set_title('Combined Feature Dimensionalities', fontsize=14, fontweight='bold')
            ax5.set_ylabel('Total Features')
            ax5.set_xticks(range(len(combo_names)))
            ax5.set_xticklabels(combo_names, rotation=45, ha='right')
            ax5.grid(True, alpha=0.3)

            for bar, dim in zip(bars, combo_dims):
                height = bar.get_height()
                ax5.text(bar.get_x() + bar.get_width() / 2., height + 10,
                         f'{dim}', ha='center', va='bottom', fontsize=10, fontweight='bold')

        # 6. Class balance analysis
        ax6 = fig.add_subplot(gs[2, 2:])
        if embedding_loader.embeddings_data:
            class_balance = counts
            balance_ratio = max(class_balance) / min(class_balance)

            ax6.bar(range(len(class_names)), class_balance,
                    color=plt.cm.RdYlBu(np.linspace(0, 1, len(class_names))))
            ax6.set_title(f'Class Balance Analysis\n(Imbalance Ratio: {balance_ratio:.2f})',
                          fontsize=14, fontweight='bold')
            ax6.set_ylabel('Sample Count')
            ax6.set_xticks(range(len(class_names)))
            ax6.set_xticklabels(class_names, rotation=45, ha='right')
            ax6.grid(True, alpha=0.3)

            # Add horizontal line for mean
            mean_count = np.mean(class_balance)
            ax6.axhline(y=mean_count, color='red', linestyle='--', alpha=0.7,
                        label=f'Mean: {mean_count:.1f}')
            ax6.legend()

        # 7. Memory usage analysis
        ax7 = fig.add_subplot(gs[3, :2])
        memory_usage = []
        for feature_type in feature_names:
            if feature_type in embedding_loader.embeddings_data:
                embeddings = embedding_loader.embeddings_data[feature_type]['embeddings']
                memory_mb = embeddings.nbytes / (1024 * 1024)
                memory_usage.append(memory_mb)

        bars = ax7.bar(feature_names, memory_usage, color=['gold', 'silver', '#cd7f32'])  # bronze hex code
        ax7.set_title('Memory Usage by Feature Type', fontsize=14, fontweight='bold')
        ax7.set_ylabel('Memory Usage (MB)')
        ax7.grid(True, alpha=0.3)

        for bar, mem in zip(bars, memory_usage):
            height = bar.get_height()
            ax7.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                     f'{mem:.1f}MB', ha='center', va='bottom', fontweight='bold')

        # 8. Data quality metrics
        ax8 = fig.add_subplot(gs[3, 2:])
        quality_metrics = []
        quality_labels = []

        for feature_type in feature_names:
            if feature_type in embedding_loader.embeddings_data:
                embeddings = embedding_loader.embeddings_data[feature_type]['embeddings']

                # Calculate quality metrics
                nan_ratio = np.isnan(embeddings).sum() / embeddings.size
                inf_ratio = np.isinf(embeddings).sum() / embeddings.size
                zero_ratio = (embeddings == 0).sum() / embeddings.size
                variance = np.var(embeddings)

                quality_metrics.append([nan_ratio, inf_ratio, zero_ratio, variance])
                quality_labels.append(feature_type)

        if quality_metrics:
            quality_df = pd.DataFrame(quality_metrics,
                                      columns=['NaN Ratio', 'Inf Ratio', 'Zero Ratio', 'Variance'],
                                      index=quality_labels)

            sns.heatmap(quality_df, annot=True, fmt='.4f', cmap='YlOrRd', ax=ax8,
                        cbar_kws={'label': 'Ratio/Variance'})
            ax8.set_title('Data Quality Metrics', fontsize=14, fontweight='bold')

        filename = f"dataset_overview_{self.results_manager.timestamp}.png"
        filepath = os.path.join(self.results_manager.results_dir, "visualizations", filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Dataset overview plot saved: {filepath}")
        return filepath

    def plot_advanced_embeddings_analysis(self, embeddings, labels, class_names, feature_name):
        """Create advanced embedding space analysis with multiple dimensionality reduction techniques"""

        fig = plt.figure(figsize=(24, 16))
        fig.suptitle(f'Advanced Embeddings Analysis: {feature_name}', fontsize=20, fontweight='bold')

        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

        # Subsample for computational efficiency
        n_samples = min(2000, len(embeddings))
        indices = np.random.choice(len(embeddings), n_samples, replace=False)
        embeddings_subset = embeddings[indices]
        labels_subset = labels[indices]

        colors = plt.cm.Set3(np.linspace(0, 1, len(class_names)))

        # 1. t-SNE with different perplexities
        perplexities = [5, 30, 50]
        for i, perp in enumerate(perplexities):
            ax = fig.add_subplot(gs[0, i])

            tsne = TSNE(n_components=2, random_state=42, perplexity=min(perp, n_samples - 1))
            embeddings_2d = tsne.fit_transform(embeddings_subset)

            for j, class_name in enumerate(class_names):
                mask = labels_subset == j
                ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                           c=[colors[j]], label=class_name, alpha=0.7, s=30)

            ax.set_title(f't-SNE (perplexity={perp})', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

        # 2. PCA analysis
        ax_pca = fig.add_subplot(gs[0, 3])
        pca = PCA(n_components=2)
        embeddings_pca = pca.fit_transform(embeddings_subset)

        for j, class_name in enumerate(class_names):
            mask = labels_subset == j
            ax_pca.scatter(embeddings_pca[mask, 0], embeddings_pca[mask, 1],
                           c=[colors[j]], label=class_name, alpha=0.7, s=30)

        ax_pca.set_title(f'PCA\n(Var: {pca.explained_variance_ratio_.sum():.3f})',
                         fontsize=12, fontweight='bold')
        ax_pca.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.3f})')
        ax_pca.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.3f})')
        ax_pca.grid(True, alpha=0.3)

        # 3. MDS
        ax_mds = fig.add_subplot(gs[1, 0])
        mds = MDS(n_components=2, random_state=42, max_iter=300, n_init=1)
        embeddings_mds = mds.fit_transform(embeddings_subset)

        for j, class_name in enumerate(class_names):
            mask = labels_subset == j
            ax_mds.scatter(embeddings_mds[mask, 0], embeddings_mds[mask, 1],
                           c=[colors[j]], label=class_name, alpha=0.7, s=30)

        ax_mds.set_title('MDS', fontsize=12, fontweight='bold')
        ax_mds.grid(True, alpha=0.3)

        # 4. K-means clustering analysis
        ax_kmeans = fig.add_subplot(gs[1, 1])

        # Find optimal number of clusters
        silhouette_scores = []
        k_range = range(2, min(11, len(class_names) + 3))

        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(embeddings_subset)
            score = silhouette_score(embeddings_subset, cluster_labels)
            silhouette_scores.append(score)

        ax_kmeans.plot(k_range, silhouette_scores, 'bo-')
        ax_kmeans.set_title('K-means Clustering\n(Silhouette Score)', fontsize=12, fontweight='bold')
        ax_kmeans.set_xlabel('Number of Clusters')
        ax_kmeans.set_ylabel('Silhouette Score')
        ax_kmeans.grid(True, alpha=0.3)

        # Mark the optimal k
        optimal_k = k_range[np.argmax(silhouette_scores)]
        ax_kmeans.axvline(x=optimal_k, color='red', linestyle='--',
                          label=f'Optimal k={optimal_k}')
        ax_kmeans.legend()

        # 5. Feature correlation heatmap
        ax_corr = fig.add_subplot(gs[1, 2])

        # Sample features for correlation analysis
        n_features_sample = min(50, embeddings_subset.shape[1])
        feature_indices = np.random.choice(embeddings_subset.shape[1], n_features_sample, replace=False)
        embeddings_sample = embeddings_subset[:, feature_indices]

        correlation_matrix = np.corrcoef(embeddings_sample.T)

        sns.heatmap(correlation_matrix, cmap='coolwarm', center=0, ax=ax_corr,
                    cbar_kws={'label': 'Correlation'})
        ax_corr.set_title(f'Feature Correlations\n({n_features_sample} features)',
                          fontsize=12, fontweight='bold')

        # 6. Class separability analysis
        ax_sep = fig.add_subplot(gs[1, 3])

        # Calculate pairwise distances between class centroids
        centroids = []
        for i in range(len(class_names)):
            mask = labels_subset == i
            if np.any(mask):
                centroid = np.mean(embeddings_subset[mask], axis=0)
                centroids.append(centroid)

        if len(centroids) > 1:
            distances = []
            pairs = []
            for i in range(len(centroids)):
                for j in range(i + 1, len(centroids)):
                    dist = np.linalg.norm(centroids[i] - centroids[j])
                    distances.append(dist)
                    pairs.append(f'{class_names[i]}-{class_names[j]}')

            y_pos = np.arange(len(pairs))
            bars = ax_sep.barh(y_pos, distances, color=plt.cm.viridis(np.linspace(0, 1, len(distances))))
            ax_sep.set_yticks(y_pos)
            ax_sep.set_yticklabels(pairs, fontsize=8)
            ax_sep.set_xlabel('Euclidean Distance')
            ax_sep.set_title('Class Centroid Distances', fontsize=12, fontweight='bold')
            ax_sep.grid(True, alpha=0.3)

        # 7. Dimensionality reduction comparison
        ax_dim = fig.add_subplot(gs[2, :2])

        # Calculate explained variance for different number of PCA components
        pca_full = PCA()
        pca_full.fit(embeddings_subset)

        cumsum_var = np.cumsum(pca_full.explained_variance_ratio_)
        n_components = np.arange(1, min(51, len(cumsum_var) + 1))

        ax_dim.plot(n_components, cumsum_var[:len(n_components)], 'bo-', linewidth=2)
        ax_dim.axhline(y=0.95, color='red', linestyle='--', alpha=0.7, label='95% Variance')
        ax_dim.axhline(y=0.90, color='orange', linestyle='--', alpha=0.7, label='90% Variance')
        ax_dim.set_xlabel('Number of Components')
        ax_dim.set_ylabel('Cumulative Explained Variance')
        ax_dim.set_title('PCA Explained Variance', fontsize=12, fontweight='bold')
        ax_dim.grid(True, alpha=0.3)
        ax_dim.legend()

        # 8. Feature importance distribution
        ax_feat = fig.add_subplot(gs[2, 2:])

        # Calculate feature variance as proxy for importance
        feature_vars = np.var(embeddings_subset, axis=0)

        # Create histogram of feature variances
        ax_feat.hist(feature_vars, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax_feat.axvline(np.mean(feature_vars), color='red', linestyle='--',
                        label=f'Mean: {np.mean(feature_vars):.4f}')
        ax_feat.axvline(np.median(feature_vars), color='orange', linestyle='--',
                        label=f'Median: {np.median(feature_vars):.4f}')
        ax_feat.set_xlabel('Feature Variance')
        ax_feat.set_ylabel('Frequency')
        ax_feat.set_title('Feature Importance Distribution', fontsize=12, fontweight='bold')
        ax_feat.grid(True, alpha=0.3)
        ax_feat.legend()

        filename = f"advanced_embeddings_analysis_{feature_name}_{self.results_manager.timestamp}.png"
        filepath = os.path.join(self.results_manager.results_dir, "visualizations", filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Advanced embeddings analysis saved: {filepath}")
        return filepath

    def plot_statistical_significance_analysis(self, all_kfold_results):
        """Perform and visualize statistical significance tests"""

        fig = plt.figure(figsize=(20, 12))
        fig.suptitle('Statistical Significance Analysis', fontsize=18, fontweight='bold')

        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

        # Prepare data for statistical tests
        classifier_scores = defaultdict(list)
        feature_scores = defaultdict(list)
        all_scores = []

        for feature_name, feature_results in all_kfold_results.items():
            for clf_name, results in feature_results.items():
                scores = results['individual_accuracies']
                classifier_scores[clf_name].extend(scores)
                feature_scores[feature_name].extend(scores)
                all_scores.extend([(clf_name, feature_name, score) for score in scores])

        # 1. Classifier comparison boxplot with statistical tests
        ax1 = fig.add_subplot(gs[0, 0])

        clf_names = list(classifier_scores.keys())
        clf_scores = [classifier_scores[clf] for clf in clf_names]

        box_plot = ax1.boxplot(clf_scores, labels=clf_names, patch_artist=True)

        # Color the boxes
        colors = plt.cm.Set3(np.linspace(0, 1, len(clf_names)))
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)

        ax1.set_title('Classifier Performance Distribution', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Accuracy')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)

        # Perform Friedman test if we have enough data
        if len(clf_names) >= 3:
            try:
                stat, p_value = friedmanchisquare(*clf_scores)
                ax1.text(0.02, 0.98, f'Friedman Test\nχ² = {stat:.3f}\np = {p_value:.4f}',
                         transform=ax1.transAxes, verticalalignment='top',
                         bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
            except:
                pass

        # 2. Feature comparison
        ax2 = fig.add_subplot(gs[0, 1])

        feat_names = list(feature_scores.keys())
        feat_scores = [feature_scores[feat] for feat in feat_names]

        box_plot2 = ax2.boxplot(feat_scores, labels=feat_names, patch_artist=True)

        colors2 = plt.cm.Pastel1(np.linspace(0, 1, len(feat_names)))
        for patch, color in zip(box_plot2['boxes'], colors2):
            patch.set_facecolor(color)

        ax2.set_title('Feature Type Performance Distribution', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Accuracy')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)

        # 3. Pairwise comparison heatmap
        ax3 = fig.add_subplot(gs[0, 2])

        # Create pairwise comparison matrix
        comparison_matrix = np.zeros((len(clf_names), len(clf_names)))
        p_value_matrix = np.zeros((len(clf_names), len(clf_names)))

        for i, clf1 in enumerate(clf_names):
            for j, clf2 in enumerate(clf_names):
                if i != j:
                    scores1 = classifier_scores[clf1]
                    scores2 = classifier_scores[clf2]

                    # Ensure equal length for comparison
                    min_len = min(len(scores1), len(scores2))
                    scores1_trim = scores1[:min_len]
                    scores2_trim = scores2[:min_len]

                    try:
                        stat, p_val = wilcoxon(scores1_trim, scores2_trim)
                        comparison_matrix[i, j] = np.mean(scores1) - np.mean(scores2)
                        p_value_matrix[i, j] = p_val
                    except:
                        comparison_matrix[i, j] = 0
                        p_value_matrix[i, j] = 1

        sns.heatmap(comparison_matrix, annot=True, fmt='.3f',
                    xticklabels=clf_names, yticklabels=clf_names,
                    cmap='RdBu_r', center=0, ax=ax3,
                    cbar_kws={'label': 'Mean Difference'})
        ax3.set_title('Pairwise Performance Differences', fontsize=12, fontweight='bold')

        # 4. Performance stability analysis
        ax4 = fig.add_subplot(gs[1, 0])

        stability_data = []
        stability_labels = []

        for feature_name, feature_results in all_kfold_results.items():
            for clf_name, results in feature_results.items():
                std_acc = results['std_accuracy']
                mean_acc = results['mean_accuracy']
                cv = std_acc / mean_acc if mean_acc > 0 else 0  # Coefficient of variation

                stability_data.append(cv)
                stability_labels.append(f"{clf_name}\n({feature_name})")

        bars = ax4.bar(range(len(stability_labels)), stability_data,
                       color=plt.cm.viridis(np.linspace(0, 1, len(stability_labels))))
        ax4.set_title('Performance Stability (Coefficient of Variation)', fontsize=12, fontweight='bold')
        ax4.set_ylabel('CV (std/mean)')
        ax4.set_xticks(range(len(stability_labels)))
        ax4.set_xticklabels(stability_labels, rotation=45, ha='right', fontsize=8)
        ax4.grid(True, alpha=0.3)

        # Add horizontal line for median CV
        median_cv = np.median(stability_data)
        ax4.axhline(y=median_cv, color='red', linestyle='--', alpha=0.7,
                    label=f'Median CV: {median_cv:.3f}')
        ax4.legend()

        # 5. Effect size analysis
        ax5 = fig.add_subplot(gs[1, 1])

        # Calculate Cohen's d for pairwise comparisons
        effect_sizes = []
        comparison_pairs = []

        for i, clf1 in enumerate(clf_names):
            for j, clf2 in enumerate(clf_names[i + 1:], i + 1):
                scores1 = np.array(classifier_scores[clf1])
                scores2 = np.array(classifier_scores[clf2])

                # Calculate Cohen's d
                pooled_std = np.sqrt(((len(scores1) - 1) * np.var(scores1, ddof=1) +
                                      (len(scores2) - 1) * np.var(scores2, ddof=1)) /
                                     (len(scores1) + len(scores2) - 2))

                if pooled_std > 0:
                    cohens_d = (np.mean(scores1) - np.mean(scores2)) / pooled_std
                else:
                    cohens_d = 0

                effect_sizes.append(abs(cohens_d))
                comparison_pairs.append(f"{clf1}\nvs\n{clf2}")

        if effect_sizes:
            bars = ax5.bar(range(len(comparison_pairs)), effect_sizes,
                           color=plt.cm.plasma(np.linspace(0, 1, len(effect_sizes))))
            ax5.set_title("Effect Sizes (|Cohen's d|)", fontsize=12, fontweight='bold')
            ax5.set_ylabel("|Cohen's d|")
            ax5.set_xticks(range(len(comparison_pairs)))
            ax5.set_xticklabels(comparison_pairs, rotation=0, ha='center', fontsize=8)
            ax5.grid(True, alpha=0.3)

            # Add effect size interpretation lines
            ax5.axhline(y=0.2, color='green', linestyle='--', alpha=0.5, label='Small (0.2)')
            ax5.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Medium (0.5)')
            ax5.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='Large (0.8)')
            ax5.legend()

        # 6. Confidence intervals
        ax6 = fig.add_subplot(gs[1, 2])

        means = []
        ci_lower = []
        ci_upper = []
        labels = []

        for feature_name, feature_results in all_kfold_results.items():
            for clf_name, results in feature_results.items():
                mean_acc = results['mean_accuracy']
                std_acc = results['std_accuracy']
                n_folds = results['k_folds']

                # Calculate 95% confidence interval
                se = std_acc / np.sqrt(n_folds)
                ci = 1.96 * se  # 95% CI

                means.append(mean_acc)
                ci_lower.append(mean_acc - ci)
                ci_upper.append(mean_acc + ci)
                labels.append(f"{clf_name}\n({feature_name})")

        x_pos = np.arange(len(labels))
        ax6.errorbar(x_pos, means, yerr=[np.array(means) - np.array(ci_lower),
                                         np.array(ci_upper) - np.array(means)],
                     fmt='o', capsize=5, capthick=2, markersize=8)

        ax6.set_title('95% Confidence Intervals', fontsize=12, fontweight='bold')
        ax6.set_ylabel('Accuracy')
        ax6.set_xticks(x_pos)
        ax6.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax6.grid(True, alpha=0.3)

        filename = f"statistical_significance_analysis_{self.results_manager.timestamp}.png"
        filepath = os.path.join(self.results_manager.results_dir, "visualizations", filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Statistical significance analysis saved: {filepath}")
        return filepath

    def plot_learning_curves_analysis(self, all_results, embedding_loader):
        """Create learning curves and training analysis"""

        fig = plt.figure(figsize=(20, 12))
        fig.suptitle('Learning Curves and Training Analysis', fontsize=18, fontweight='bold')

        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

        # 1. Training set size impact
        ax1 = fig.add_subplot(gs[0, 0])

        # Simulate different training sizes
        train_sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

        # Use first available feature for analysis
        if embedding_loader.embeddings_data:
            first_feature = list(embedding_loader.embeddings_data.keys())[0]
            X, y, class_names = embedding_loader.get_individual_embeddings(first_feature)

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Test with RandomForest as example
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import learning_curve

            rf = RandomForestClassifier(n_estimators=50, random_state=42)

            try:
                train_sizes_abs, train_scores, val_scores = learning_curve(
                    rf, X_scaled, y, train_sizes=train_sizes, cv=5,
                    scoring='accuracy', n_jobs=-1, random_state=42
                )

                train_mean = np.mean(train_scores, axis=1)
                train_std = np.std(train_scores, axis=1)
                val_mean = np.mean(val_scores, axis=1)
                val_std = np.std(val_scores, axis=1)

                ax1.plot(train_sizes_abs, train_mean, 'o-', color='blue', label='Training Score')
                ax1.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std,
                                 alpha=0.1, color='blue')

                ax1.plot(train_sizes_abs, val_mean, 'o-', color='red', label='Validation Score')
                ax1.fill_between(train_sizes_abs, val_mean - val_std, val_mean + val_std,
                                 alpha=0.1, color='red')

                ax1.set_xlabel('Training Set Size')
                ax1.set_ylabel('Accuracy Score')
                ax1.set_title(f'Learning Curve\n({first_feature} - RandomForest)', fontsize=12, fontweight='bold')
                ax1.legend()
                ax1.grid(True, alpha=0.3)

            except Exception as e:
                ax1.text(0.5, 0.5, f'Learning curve analysis failed:\n{str(e)}',
                         transform=ax1.transAxes, ha='center', va='center')

        # 2. Cross-validation fold variance
        ax2 = fig.add_subplot(gs[0, 1])

        fold_variances = []
        model_labels = []

        for feature_name, feature_results in all_results.items():
            for clf_name, results in feature_results.items():
                if 'kfold_results' in results:
                    kfold = results['kfold_results']
                    variance = kfold['std_accuracy'] ** 2
                    fold_variances.append(variance)
                    model_labels.append(f"{clf_name}\n({feature_name})")

        if fold_variances:
            bars = ax2.bar(range(len(model_labels)), fold_variances,
                           color=plt.cm.coolwarm(np.linspace(0, 1, len(fold_variances))))
            ax2.set_title('Cross-Validation Variance', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Variance')
            ax2.set_xticks(range(len(model_labels)))
            ax2.set_xticklabels(model_labels, rotation=45, ha='right', fontsize=8)
            ax2.grid(True, alpha=0.3)

        # 3. Performance vs complexity
        ax3 = fig.add_subplot(gs[0, 2])

        complexity_scores = []
        performance_scores = []
        model_names = []

        # Define complexity scores for different models (relative)
        complexity_map = {
            'LogisticRegression': 1,
            'SVM': 3,
            'RandomForest': 4,
            'GradientBoosting': 5,
            'XGBoost': 5
        }

        for feature_name, feature_results in all_results.items():
            for clf_name, results in feature_results.items():
                if 'kfold_results' in results:
                    complexity = complexity_map.get(clf_name, 3)
                    performance = results['kfold_results']['mean_accuracy']

                    complexity_scores.append(complexity)
                    performance_scores.append(performance)
                    model_names.append(f"{clf_name} ({feature_name})")

        if complexity_scores:
            scatter = ax3.scatter(complexity_scores, performance_scores,
                                  c=range(len(complexity_scores)), cmap='viridis',
                                  s=100, alpha=0.7)

            # Add model labels
            for i, name in enumerate(model_names):
                ax3.annotate(name, (complexity_scores[i], performance_scores[i]),
                             xytext=(5, 5), textcoords='offset points', fontsize=8)

            ax3.set_xlabel('Model Complexity (Relative)')
            ax3.set_ylabel('Performance (Accuracy)')
            ax3.set_title('Performance vs Complexity Trade-off', fontsize=12, fontweight='bold')
            ax3.grid(True, alpha=0.3)

        # 4. Feature importance across models
        ax4 = fig.add_subplot(gs[1, :])

        # Create a comprehensive feature analysis
        feature_performance = defaultdict(list)
        classifier_performance = defaultdict(list)

        for feature_name, feature_results in all_results.items():
            for clf_name, results in feature_results.items():
                if 'kfold_results' in results:
                    performance = results['kfold_results']['mean_accuracy']
                    feature_performance[feature_name].append(performance)
                    classifier_performance[clf_name].append(performance)

        # Create grouped bar chart
        feature_names = list(feature_performance.keys())
        classifier_names = list(classifier_performance.keys())

        x = np.arange(len(feature_names))
        width = 0.15

        for i, clf_name in enumerate(classifier_names):
            clf_scores = []
            for feature_name in feature_names:
                # Find performance for this classifier-feature combination
                if feature_name in all_results and clf_name in all_results[feature_name]:
                    score = all_results[feature_name][clf_name]['kfold_results']['mean_accuracy']
                    clf_scores.append(score)
                else:
                    clf_scores.append(0)

            bars = ax4.bar(x + i * width, clf_scores, width,
                           label=clf_name, color=self.colors[i % len(self.colors)], alpha=0.8)

            # Add value labels on bars
            for bar, score in zip(bars, clf_scores):
                if score > 0:
                    height = bar.get_height()
                    ax4.text(bar.get_x() + bar.get_width() / 2., height + 0.005,
                             f'{score:.3f}', ha='center', va='bottom', fontsize=8)

        ax4.set_xlabel('Feature Types')
        ax4.set_ylabel('Accuracy')
        ax4.set_title('Comprehensive Performance Comparison\n(All Classifier-Feature Combinations)',
                      fontsize=14, fontweight='bold')
        ax4.set_xticks(x + width * (len(classifier_names) - 1) / 2)
        ax4.set_xticklabels(feature_names)
        ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax4.grid(True, alpha=0.3)

        filename = f"learning_curves_analysis_{self.results_manager.timestamp}.png"
        filepath = os.path.join(self.results_manager.results_dir, "visualizations", filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Learning curves analysis saved: {filepath}")
        return filepath

    def plot_traditional_model_comparison(self, all_kfold_results, save_path=None):
        """Create UrbanSound8k style comparison plot"""

        # Prepare data in the format: Model combinations
        model_data = []

        for feature_name, feature_results in all_kfold_results.items():
            for clf_name, results in feature_results.items():
                model_data.append({
                    'Model': f"{feature_name}\n{clf_name}",
                    'Feature': feature_name,
                    'Classifier': clf_name,
                    'Accuracy': results['mean_accuracy'],
                    'F1_Score': results['mean_f1_score'],
                    'Recall': results['mean_recall']
                })

        df = pd.DataFrame(model_data)

        # Create the plot
        fig, ax = plt.subplots(figsize=(16, 10))

        x = np.arange(len(df))
        width = 0.25

        # Plot bars
        bars1 = ax.bar(x - width, df['Accuracy'], width, label='Accuracy',
                       color='#1f77b4', alpha=0.8)
        bars2 = ax.bar(x, df['F1_Score'], width, label='F1-score',
                       color='#ff7f0e', alpha=0.8)
        bars3 = ax.bar(x + width, df['Recall'], width, label='Recall',
                       color='#2ca02c', alpha=0.8)

        # Add trend line for recall (similar to original)
        recall_smooth = np.convolve(df['Recall'], np.ones(3) / 3, mode='same')
        ax.plot(x, recall_smooth, color='gray', linewidth=3, alpha=0.7)

        # Formatting
        ax.set_xlabel('Models', fontsize=14, fontweight='bold')
        ax.set_ylabel('Performance', fontsize=14, fontweight='bold')
        ax.set_title('Traditional Model Performance on CNN Embeddings',
                     fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(df['Model'], rotation=45, ha='right', fontsize=10)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

        plt.tight_layout()

        if save_path is None:
            filename = f"traditional_model_comparison_{self.results_manager.timestamp}.png"
            save_path = os.path.join(self.results_manager.results_dir, "visualizations", filename)

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Traditional model comparison plot saved: {save_path}")
        return save_path

    def plot_kfold_comparison(self, cv_results_dict, feature_name):
        """Plot k-fold results comparison across classifiers"""

        classifiers = list(cv_results_dict.keys())
        accuracies = [cv_results_dict[clf]['mean_accuracy'] for clf in classifiers]
        accuracy_stds = [cv_results_dict[clf]['std_accuracy'] for clf in classifiers]
        f1_scores = [cv_results_dict[clf]['mean_f1_score'] for clf in classifiers]
        f1_stds = [cv_results_dict[clf]['std_f1_score'] for clf in classifiers]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

        x_pos = np.arange(len(classifiers))

        # Accuracy plot
        bars1 = ax1.bar(x_pos, accuracies, yerr=accuracy_stds, capsize=5,
                        alpha=0.8, color='skyblue', edgecolor='navy', linewidth=1.5)
        ax1.set_title(f'{feature_name} - 10-Fold CV Accuracy Comparison',
                      fontsize=14, fontweight='bold')
        ax1.set_xlabel('Ensemble Methods', fontsize=12)
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(classifiers, rotation=45, ha='right')
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3)

        for i, (acc, std) in enumerate(zip(accuracies, accuracy_stds)):
            ax1.text(i, acc + std + 0.02, f'{acc:.3f}±{std:.3f}',
                     ha='center', va='bottom', fontsize=10, fontweight='bold')

        # F1-Score plot
        bars2 = ax2.bar(x_pos, f1_scores, yerr=f1_stds, capsize=5,
                        alpha=0.8, color='lightcoral', edgecolor='darkred', linewidth=1.5)
        ax2.set_title(f'{feature_name} - 10-Fold CV F1-Score Comparison',
                      fontsize=14, fontweight='bold')
        ax2.set_xlabel('Ensemble Methods', fontsize=12)
        ax2.set_ylabel('F1-Score', fontsize=12)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(classifiers, rotation=45, ha='right')
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)

        for i, (f1, std) in enumerate(zip(f1_scores, f1_stds)):
            ax2.text(i, f1 + std + 0.02, f'{f1:.3f}±{std:.3f}',
                     ha='center', va='bottom', fontsize=10, fontweight='bold')

        plt.tight_layout()
        filename = f"kfold_comparison_{feature_name}_{self.results_manager.timestamp}.png"
        filepath = os.path.join(self.results_manager.results_dir, "visualizations", filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"K-fold comparison plot saved: {filepath}")
        return filepath

    def plot_performance_radar_chart(self, all_kfold_results):
        """Create radar chart comparing all models"""

        # Prepare data
        models = []
        for feature_name, feature_results in all_kfold_results.items():
            for clf_name, results in feature_results.items():
                models.append({
                    'name': f"{clf_name}\n({feature_name})",
                    'accuracy': results['mean_accuracy'],
                    'f1_score': results['mean_f1_score'],
                    'precision': results['mean_precision'],
                    'recall': results['mean_recall']
                })

        # Select top 6 models for clarity
        models.sort(key=lambda x: x['accuracy'], reverse=True)
        top_models = models[:6]

        # Setup radar chart
        categories = ['Accuracy', 'F1-Score', 'Precision', 'Recall']
        N = len(categories)

        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Complete the circle

        fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar'))

        for i, model in enumerate(top_models):
            values = [model['accuracy'], model['f1_score'],
                      model['precision'], model['recall']]
            values += values[:1]  # Complete the circle

            ax.plot(angles, values, 'o-', linewidth=2,
                    label=model['name'], color=self.colors[i % len(self.colors)])
            ax.fill(angles, values, alpha=0.1, color=self.colors[i % len(self.colors)])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=12)
        ax.set_ylim(0, 1)
        ax.set_title('Top 6 Models - Performance Radar Chart',
                     fontsize=16, fontweight='bold', pad=30)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax.grid(True)

        filename = f"performance_radar_{self.results_manager.timestamp}.png"
        filepath = os.path.join(self.results_manager.results_dir, "visualizations", filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Performance radar chart saved: {filepath}")
        return filepath

    def plot_confusion_matrix_enhanced(self, y_true, y_pred, class_names,
                                       classifier_name, feature_name):
        """Plot enhanced confusion matrix with percentages"""

        cm = confusion_matrix(y_true, y_pred)
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

        # Raw counts
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names, ax=ax1)
        ax1.set_title(f'Confusion Matrix (Counts)\n{classifier_name} ({feature_name})',
                      fontsize=14, fontweight='bold')
        ax1.set_xlabel('Predicted Label')
        ax1.set_ylabel('True Label')

        # Percentages
        sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Reds',
                    xticklabels=class_names, yticklabels=class_names, ax=ax2,
                    cbar_kws={'label': 'Percentage (%)'})
        ax2.set_title(f'Confusion Matrix (Percentages)\n{classifier_name} ({feature_name})',
                      fontsize=14, fontweight='bold')
        ax2.set_xlabel('Predicted Label')
        ax2.set_ylabel('True Label')

        plt.tight_layout()
        filename = f"confusion_matrix_enhanced_{classifier_name}_{feature_name}_{self.results_manager.timestamp}.png"
        filepath = os.path.join(self.results_manager.results_dir, "visualizations", filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        return filepath

    def plot_feature_importance(self, classifier, feature_name, classifier_name, top_n=20):
        """Plot feature importance for tree-based methods"""

        if not hasattr(classifier, 'feature_importances_'):
            return None

        importances = classifier.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]

        plt.figure(figsize=(14, 10))
        plt.title(f'Top {top_n} Feature Importances\n{classifier_name} ({feature_name})',
                  fontsize=16, fontweight='bold')

        bars = plt.bar(range(top_n), importances[indices],
                       color='lightblue', edgecolor='navy', alpha=0.8)

        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height + 0.001,
                     f'{height:.4f}', ha='center', va='bottom', fontsize=8)

        plt.xlabel('Feature Index', fontsize=12)
        plt.ylabel('Importance', fontsize=12)
        plt.xticks(range(top_n), indices, rotation=45)
        plt.grid(True, alpha=0.3)

        filename = f"feature_importance_{classifier_name}_{feature_name}_{self.results_manager.timestamp}.png"
        filepath = os.path.join(self.results_manager.results_dir, "visualizations", filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        return filepath

    def plot_embeddings_tsne(self, embeddings, labels, class_names, feature_name):
        """Plot t-SNE visualization of embeddings with enhanced styling"""

        print(f"Creating t-SNE visualization for {feature_name}...")

        n_samples = len(embeddings)
        if n_samples > 2000:
            indices = np.random.choice(n_samples, 2000, replace=False)
            embeddings_subset = embeddings[indices]
            labels_subset = labels[indices]
        else:
            embeddings_subset = embeddings
            labels_subset = labels

        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings_subset) - 1))
        embeddings_2d = tsne.fit_transform(embeddings_subset)

        plt.figure(figsize=(14, 10))
        colors = plt.cm.Set3(np.linspace(0, 1, len(class_names)))

        for i, class_name in enumerate(class_names):
            mask = labels_subset == i
            plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                        c=[colors[i]], label=class_name, alpha=0.7, s=60, edgecolors='black')

        plt.title(f'{feature_name} Embeddings - t-SNE Visualization',
                  fontsize=16, fontweight='bold')
        plt.xlabel('t-SNE Component 1', fontsize=12)
        plt.ylabel('t-SNE Component 2', fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        plt.grid(True, alpha=0.3)

        filename = f"embeddings_tsne_{feature_name}_{self.results_manager.timestamp}.png"
        filepath = os.path.join(self.results_manager.results_dir, "visualizations", filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        return filepath

    def plot_overall_heatmap_comparison(self, all_results):
        """Plot comprehensive heatmap comparison"""

        data = []
        for feature_name, feature_results in all_results.items():
            for clf_name, results in feature_results.items():
                if 'kfold_results' in results:
                    kfold = results['kfold_results']
                    data.append({
                        'Feature': feature_name,
                        'Classifier': clf_name,
                        'Accuracy': kfold['mean_accuracy'],
                        'F1_Score': kfold['mean_f1_score'],
                        'Precision': kfold['mean_precision'],
                        'Recall': kfold['mean_recall']
                    })

        df = pd.DataFrame(data)

        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('10-Fold CV Performance Heatmaps\n(All Features vs All Classifiers)',
                     fontsize=18, fontweight='bold')

        metrics = ['Accuracy', 'F1_Score', 'Precision', 'Recall']
        titles = ['Accuracy', 'F1-Score', 'Precision', 'Recall']
        cmaps = ['YlOrRd', 'YlGnBu', 'BuPu', 'Oranges']

        for idx, (metric, title, cmap) in enumerate(zip(metrics, titles, cmaps)):
            ax = axes[idx // 2, idx % 2]
            pivot_data = df.pivot(index='Classifier', columns='Feature', values=metric)

            sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap=cmap, ax=ax,
                        cbar_kws={'label': title})
            ax.set_title(f'{title} Comparison', fontsize=14, fontweight='bold')
            ax.set_xlabel('Feature Type', fontsize=12)
            ax.set_ylabel('Classifier', fontsize=12)

        plt.tight_layout()
        filename = f"overall_heatmap_comparison_{self.results_manager.timestamp}.png"
        filepath = os.path.join(self.results_manager.results_dir, "visualizations", filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        return filepath

    def plot_performance_distribution(self, all_kfold_results):
        """Plot distribution of performance across folds"""

        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        fig.suptitle('Performance Distribution Across K-Folds', fontsize=16, fontweight='bold')

        all_accuracies = []
        all_f1_scores = []
        all_precisions = []
        all_recalls = []
        labels = []

        for feature_name, feature_results in all_kfold_results.items():
            for clf_name, results in feature_results.items():
                model_label = f"{clf_name}_{feature_name}"
                labels.append(model_label)
                all_accuracies.append(results['individual_accuracies'])
                all_f1_scores.append(results['individual_f1_scores'])
                # Add precision and recall if available
                if 'individual_precisions' in results:
                    all_precisions.append(results['individual_precisions'])
                    all_recalls.append(results['individual_recalls'])

        # Box plots
        axes[0, 0].boxplot(all_accuracies, labels=labels)
        axes[0, 0].set_title('Accuracy Distribution', fontsize=12, fontweight='bold')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].boxplot(all_f1_scores, labels=labels)
        axes[0, 1].set_title('F1-Score Distribution', fontsize=12, fontweight='bold')
        axes[0, 1].set_ylabel('F1-Score')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)

        # Violin plots for better distribution visualization
        parts1 = axes[1, 0].violinplot(all_accuracies, positions=range(1, len(all_accuracies) + 1))
        axes[1, 0].set_title('Accuracy Density Distribution', fontsize=12, fontweight='bold')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].set_xticks(range(1, len(labels) + 1))
        axes[1, 0].set_xticklabels(labels, rotation=45, ha='right')
        axes[1, 0].grid(True, alpha=0.3)

        parts2 = axes[1, 1].violinplot(all_f1_scores, positions=range(1, len(all_f1_scores) + 1))
        axes[1, 1].set_title('F1-Score Density Distribution', fontsize=12, fontweight='bold')
        axes[1, 1].set_ylabel('F1-Score')
        axes[1, 1].set_xticks(range(1, len(labels) + 1))
        axes[1, 1].set_xticklabels(labels, rotation=45, ha='right')
        axes[1, 1].grid(True, alpha=0.3)

        # Color the violin plots
        for pc in parts1['bodies']:
            pc.set_facecolor('lightblue')
            pc.set_alpha(0.7)
        for pc in parts2['bodies']:
            pc.set_facecolor('lightcoral')
            pc.set_alpha(0.7)

        plt.tight_layout()
        filename = f"performance_distribution_{self.results_manager.timestamp}.png"
        filepath = os.path.join(self.results_manager.results_dir, "visualizations", filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        return filepath

    def create_comprehensive_report(self, all_results, all_kfold_results, best_results):
        """Create a comprehensive PDF-style report"""

        fig = plt.figure(figsize=(20, 24))

        # Title
        fig.suptitle('Ensemble Learning Performance Report\nCNN Embeddings Classification',
                     fontsize=24, fontweight='bold', y=0.98)

        # Create subplots for different sections
        gs = fig.add_gridspec(6, 4, hspace=0.4, wspace=0.3)

        # 1. Summary statistics table
        ax1 = fig.add_subplot(gs[0, :])
        ax1.axis('tight')
        ax1.axis('off')

        # Prepare summary data
        summary_data = []
        for feature_name, feature_results in all_kfold_results.items():
            for clf_name, results in feature_results.items():
                summary_data.append([
                    f"{clf_name}",
                    f"{feature_name}",
                    f"{results['mean_accuracy']:.4f} ± {results['std_accuracy']:.4f}",
                    f"{results['mean_f1_score']:.4f} ± {results['std_f1_score']:.4f}",
                    f"{results['mean_precision']:.4f} ± {results['std_precision']:.4f}",
                    f"{results['mean_recall']:.4f} ± {results['std_recall']:.4f}"
                ])

        headers = ['Classifier', 'Feature', 'Accuracy', 'F1-Score', 'Precision', 'Recall']
        table = ax1.table(cellText=summary_data, colLabels=headers,
                          cellLoc='center', loc='center',
                          colWidths=[0.15, 0.15, 0.175, 0.175, 0.175, 0.175])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 2)
        ax1.set_title('10-Fold Cross-Validation Results Summary',
                      fontsize=16, fontweight='bold', pad=20)

        # 2. Best performers highlight
        ax2 = fig.add_subplot(gs[1, :2])
        ax2.axis('off')

        best_text = "🏆 BEST PERFORMERS 🏆\n\n"
        if best_results['kfold_cv']:
            best_acc = best_results['kfold_cv']['best_accuracy']
            best_f1 = best_results['kfold_cv']['best_f1']

            best_text += f"🎯 Highest Accuracy:\n"
            best_text += f"   {best_acc['classifier']} on {best_acc['feature']}\n"
            best_text += f"   {best_acc['mean_accuracy']:.4f} ± {best_acc['std_accuracy']:.4f}\n\n"

            best_text += f"🎯 Highest F1-Score:\n"
            best_text += f"   {best_f1['classifier']} on {best_f1['feature']}\n"
            best_text += f"   {best_f1['mean_f1_score']:.4f} ± {best_f1['std_f1_score']:.4f}"

        ax2.text(0.05, 0.95, best_text, transform=ax2.transAxes, fontsize=12,
                 verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5",
                                                    facecolor="lightgreen", alpha=0.8))

        # 3. Feature comparison bar chart
        ax3 = fig.add_subplot(gs[1, 2:])
        feature_means = {}
        for feature_name, feature_results in all_kfold_results.items():
            accs = [results['mean_accuracy'] for results in feature_results.values()]
            feature_means[feature_name] = np.mean(accs)

        features = list(feature_means.keys())
        means = list(feature_means.values())
        bars = ax3.bar(features, means, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
        ax3.set_title('Average Performance by Feature Type', fontweight='bold')
        ax3.set_ylabel('Mean Accuracy')
        ax3.grid(True, alpha=0.3)

        # Add value labels on bars
        for bar, mean in zip(bars, means):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width() / 2., height + 0.005,
                     f'{mean:.3f}', ha='center', va='bottom', fontweight='bold')

        # 4-6. Individual feature comparisons (3 plots)
        feature_names = list(all_kfold_results.keys())[:3]  # Take first 3 features
        positions = [(2, 0, 2), (3, 0, 2), (4, 0, 2)]

        for idx, (feature_name, pos) in enumerate(zip(feature_names, positions)):
            ax = fig.add_subplot(gs[pos[0], pos[1]:pos[1] + pos[2]])

            if feature_name in all_kfold_results:
                feature_results = all_kfold_results[feature_name]
                classifiers = list(feature_results.keys())
                accuracies = [feature_results[clf]['mean_accuracy'] for clf in classifiers]
                stds = [feature_results[clf]['std_accuracy'] for clf in classifiers]

                bars = ax.bar(range(len(classifiers)), accuracies, yerr=stds,
                              capsize=5, alpha=0.8, color=self.colors[:len(classifiers)])
                ax.set_title(f'{feature_name} - Classifier Comparison', fontweight='bold')
                ax.set_xticks(range(len(classifiers)))
                ax.set_xticklabels(classifiers, rotation=45, ha='right')
                ax.set_ylabel('Accuracy')
                ax.grid(True, alpha=0.3)

                # Add value labels
                for i, (acc, std) in enumerate(zip(accuracies, stds)):
                    ax.text(i, acc + std + 0.01, f'{acc:.3f}',
                            ha='center', va='bottom', fontsize=9)

        # 5. Statistical significance test results (if applicable)
        ax4 = fig.add_subplot(gs[5, :])
        ax4.axis('off')

        stats_text = "📊 ANALYSIS SUMMARY\n\n"
        stats_text += f"• Total Models Evaluated: {sum(len(fr) for fr in all_kfold_results.values())}\n"
        stats_text += f"• Feature Types: {len(all_kfold_results)} ({', '.join(all_kfold_results.keys())})\n"
        stats_text += f"• Classifiers: {len(set(clf for fr in all_kfold_results.values() for clf in fr.keys()))}\n"
        stats_text += f"• Cross-Validation: 10-Fold Stratified\n"
        stats_text += f"• Best Overall Accuracy: {max(r['mean_accuracy'] for fr in all_kfold_results.values() for r in fr.values()):.4f}\n"
        stats_text += f"• Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

        ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, fontsize=11,
                 verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5",
                                                    facecolor="lightyellow", alpha=0.8))

        filename = f"comprehensive_report_{self.results_manager.timestamp}.png"
        filepath = os.path.join(self.results_manager.results_dir, "visualizations", filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Comprehensive report saved: {filepath}")
        return filepath


class EnsembleResultsManager:
    """Manage results directory and save ensemble outputs"""

    def __init__(self, results_dir):
        self.results_dir = results_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.setup_directories()

    def setup_directories(self):
        """Create organized results directory structure"""
        subdirs = [
            "models",
            "metrics",
            "visualizations",
            "feature_analysis",
            "kfold_results",
            "comparison_results",
            "logs",
            "reports",
            "statistical_analysis",
            "dataset_analysis"
        ]

        for subdir in subdirs:
            os.makedirs(os.path.join(self.results_dir, subdir), exist_ok=True)

    def save_model(self, model, classifier_name, feature_name):
        """Save trained model"""
        filename = f"ensemble_{classifier_name}_{feature_name}_{self.timestamp}.pkl"
        filepath = os.path.join(self.results_dir, "models", filename)

        with open(filepath, 'wb') as f:
            pickle.dump(model, f)

        print(f"Model saved: {filepath}")
        return filepath

    def save_metrics(self, metrics, classifier_name, feature_name, evaluation_type="single"):
        """Save evaluation metrics"""
        filename = f"metrics_{evaluation_type}_{classifier_name}_{feature_name}_{self.timestamp}.json"
        filepath = os.path.join(self.results_dir, "metrics", filename)

        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=4)

        return filepath

    def save_kfold_results(self, cv_results, classifier_name, feature_name):
        """Save k-fold cross validation results"""
        filename = f"kfold_{classifier_name}_{feature_name}_{self.timestamp}.json"
        filepath = os.path.join(self.results_dir, "kfold_results", filename)

        with open(filepath, 'w') as f:
            json.dump(cv_results, f, indent=4)

        return filepath

    def save_comparison_results(self, comparison_data, comparison_type):
        """Save comparison results"""
        filename = f"comparison_{comparison_type}_{self.timestamp}.json"
        filepath = os.path.join(self.results_dir, "comparison_results", filename)

        with open(filepath, 'w') as f:
            json.dump(comparison_data, f, indent=4)

        return filepath


def main(embeddings_directory, results_directory):
    """Main execution function for ensemble learning on CNN k-fold embeddings"""

    print("=" * 70)
    print("ENSEMBLE LEARNING ON CNN K-FOLD EMBEDDINGS")
    print("=" * 70)
    print(f"Embeddings directory: {embeddings_directory}")
    print(f"Results directory: {results_directory}")
    print("Using 10-fold cross-validation on k-fold embeddings")

    # Initialize components
    embedding_loader = EmbeddingLoader(embeddings_directory)
    classifier_manager = EnsembleClassifierManager()
    results_manager = EnsembleResultsManager(results_directory)
    evaluator = EnsembleEvaluator(results_manager)
    visualizer = EnhancedVisualizer(results_manager)

    # Load k-fold embeddings only
    embeddings_data = embedding_loader.load_embeddings()

    # Create dataset overview first
    print("\n" + "=" * 50)
    print("Creating Dataset Overview...")
    print("=" * 50)
    visualizer.plot_dataset_overview(embedding_loader)

    # Define feature combinations to test
    feature_combinations = {
        'MFCC_only': ['MFCC'],
        'Mel_only': ['Mel'],
        'LogMel_only': ['LogMel'],
        'All_Features': ['MFCC', 'Mel', 'LogMel']
    }

    all_classifiers = classifier_manager.get_all_classifiers()
    print(f"Available classifiers: {list(all_classifiers.keys())}")

    all_results = {}
    all_kfold_results = {}

    # Process each feature combination
    for feature_name, feature_types in feature_combinations.items():
        print(f"\n{'=' * 60}")
        print(f"Processing feature combination: {feature_name}")
        print(f"Features: {feature_types}")
        print(f"{'=' * 60}")

        try:
            X, y, class_names = embedding_loader.get_combined_embeddings(feature_types)
            print(f"Combined embeddings shape: {X.shape}")
            print(f"Number of classes: {len(class_names)}")
            print(f"Classes: {class_names}")

            # Create advanced embeddings analysis
            visualizer.plot_advanced_embeddings_analysis(X, y, class_names, feature_name)

            # Create traditional t-SNE visualization
            visualizer.plot_embeddings_tsne(X, y, class_names, feature_name)

            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            all_results[feature_name] = {}
            all_kfold_results[feature_name] = {}

            # Test each classifier
            for clf_name, classifier in all_classifiers.items():
                print(f"\n--- Processing {clf_name} ---")

                try:
                    # K-fold cross-validation
                    cv_results, y_true_cv, y_pred_cv, y_proba_cv = evaluator.evaluate_kfold_cv(
                        classifier, X_scaled, y, class_names, clf_name, feature_name, k_folds=10
                    )

                    all_kfold_results[feature_name][clf_name] = cv_results
                    results_manager.save_kfold_results(cv_results, clf_name, feature_name)

                    # Create enhanced confusion matrix
                    visualizer.plot_confusion_matrix_enhanced(y_true_cv, y_pred_cv, class_names,
                                                              clf_name, feature_name)

                    # Single train-test split for additional analysis
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_scaled, y, test_size=0.2, random_state=42, stratify=y
                    )

                    metrics, y_pred, y_pred_proba = evaluator.evaluate_single_split(
                        classifier, X_train, X_test, y_train, y_test,
                        class_names, clf_name, feature_name
                    )

                    metrics['kfold_results'] = cv_results
                    all_results[feature_name][clf_name] = metrics

                    results_manager.save_metrics(metrics, clf_name, feature_name, "single_split")
                    results_manager.save_model(classifier, clf_name, feature_name)

                    # Feature importance for tree-based methods
                    visualizer.plot_feature_importance(classifier, feature_name, clf_name)

                    print(
                        f"  ✓ {clf_name} completed - CV Accuracy: {cv_results['mean_accuracy']:.4f}±{cv_results['std_accuracy']:.4f}")

                except Exception as e:
                    print(f"  ✗ Error with {clf_name}: {e}")
                    continue

            # Create feature-specific comparison plots
            if all_kfold_results[feature_name]:
                visualizer.plot_kfold_comparison(all_kfold_results[feature_name], feature_name)

        except Exception as e:
            print(f"Error processing {feature_name}: {e}")
            continue

    print(f"\n{'=' * 60}")
    print("Creating comprehensive visualizations and analysis...")
    print(f"{'=' * 60}")

    if all_results:
        # Find best performers
        best_results = find_best_performers(all_results, all_kfold_results)

        # Create all enhanced visualizations
        print("Creating traditional model comparison plot...")
        visualizer.plot_traditional_model_comparison(all_kfold_results)

        print("Creating overall heatmap comparison...")
        visualizer.plot_overall_heatmap_comparison(all_results)

        print("Creating performance radar chart...")
        visualizer.plot_performance_radar_chart(all_kfold_results)

        print("Creating performance distribution plots...")
        visualizer.plot_performance_distribution(all_kfold_results)

        print("Creating statistical significance analysis...")
        visualizer.plot_statistical_significance_analysis(all_kfold_results)

        print("Creating learning curves analysis...")
        visualizer.plot_learning_curves_analysis(all_results, embedding_loader)

        print("Creating comprehensive report...")
        visualizer.create_comprehensive_report(all_results, all_kfold_results, best_results)

        # Save comprehensive summary
        summary = {
            'timestamp': results_manager.timestamp,
            'embeddings_directory': embeddings_directory,
            'feature_combinations': feature_combinations,
            'classifiers_used': list(all_classifiers.keys()),
            'evaluation_method': '10-fold_cross_validation',
            'best_performers': best_results,
            'all_results': all_results,
            'kfold_results': all_kfold_results
        }

        summary_path = os.path.join(results_manager.results_dir, "ensemble_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=4)

        # Create detailed CSV reports
        create_csv_reports(all_results, all_kfold_results, results_manager.results_dir,
                           results_manager.timestamp)

        # Print final results
        print_final_results(all_results, all_kfold_results, best_results)

        print(f"\n{'=' * 60}")
        print("🎉 ENSEMBLE LEARNING ANALYSIS COMPLETE! 🎉")
        print(f"{'=' * 60}")
        print(f"Results directory: {results_manager.results_dir}")
        print(f"Summary file: {summary_path}")
        print(f"Timestamp: {results_manager.timestamp}")

    return results_manager.results_dir


def find_best_performers(all_results, all_kfold_results=None):
    """Find best performing classifier-feature combinations"""

    best_results = {
        'single_split': {},
        'kfold_cv': {} if all_kfold_results else None
    }

    # Single split best performers
    best_accuracy = 0
    best_f1 = 0

    for feature_name, feature_results in all_results.items():
        for clf_name, results in feature_results.items():
            accuracy = results['accuracy']
            f1_score = results['f1_score']

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_results['single_split']['best_accuracy'] = {
                    'classifier': clf_name,
                    'feature': feature_name,
                    'accuracy': accuracy,
                    'f1_score': f1_score
                }

            if f1_score > best_f1:
                best_f1 = f1_score
                best_results['single_split']['best_f1'] = {
                    'classifier': clf_name,
                    'feature': feature_name,
                    'accuracy': accuracy,
                    'f1_score': f1_score
                }

    # K-fold CV best performers
    if all_kfold_results:
        best_kfold_acc = 0
        best_kfold_f1 = 0

        for feature_name, feature_results in all_kfold_results.items():
            for clf_name, results in feature_results.items():
                mean_acc = results['mean_accuracy']
                mean_f1 = results['mean_f1_score']

                if mean_acc > best_kfold_acc:
                    best_kfold_acc = mean_acc
                    best_results['kfold_cv']['best_accuracy'] = {
                        'classifier': clf_name,
                        'feature': feature_name,
                        'mean_accuracy': mean_acc,
                        'std_accuracy': results['std_accuracy'],
                        'mean_f1_score': mean_f1,
                        'std_f1_score': results['std_f1_score']
                    }

                if mean_f1 > best_kfold_f1:
                    best_kfold_f1 = mean_f1
                    best_results['kfold_cv']['best_f1'] = {
                        'classifier': clf_name,
                        'feature': feature_name,
                        'mean_accuracy': mean_acc,
                        'std_accuracy': results['std_accuracy'],
                        'mean_f1_score': mean_f1,
                        'std_f1_score': results['std_f1_score']
                    }

    return best_results


def create_csv_reports(all_results, all_kfold_results, results_dir, timestamp):
    """Create detailed CSV reports for easy analysis"""

    # Single split results
    single_split_data = []
    for feature_name, feature_results in all_results.items():
        for clf_name, results in feature_results.items():
            single_split_data.append({
                'Timestamp': timestamp,
                'Feature_Combination': feature_name,
                'Classifier': clf_name,
                'Accuracy': results['accuracy'],
                'F1_Score': results['f1_score'],
                'Precision': results['precision'],
                'Recall': results['recall'],
                'F1_Macro': results['f1_macro'],
                'F1_Micro': results['f1_micro'],
                'Train_Size': results['train_size'],
                'Test_Size': results['test_size']
            })

    single_split_df = pd.DataFrame(single_split_data)
    single_split_path = os.path.join(results_dir, f"single_split_results_{timestamp}.csv")
    single_split_df.to_csv(single_split_path, index=False)
    print(f"Single split results CSV saved: {single_split_path}")

    # K-fold CV results
    if all_kfold_results:
        kfold_data = []
        for feature_name, feature_results in all_kfold_results.items():
            for clf_name, results in feature_results.items():
                kfold_data.append({
                    'Timestamp': timestamp,
                    'Feature_Combination': feature_name,
                    'Classifier': clf_name,
                    'Mean_Accuracy': results['mean_accuracy'],
                    'Std_Accuracy': results['std_accuracy'],
                    'Mean_F1_Score': results['mean_f1_score'],
                    'Std_F1_Score': results['std_f1_score'],
                    'Mean_Precision': results['mean_precision'],
                    'Std_Precision': results['std_precision'],
                    'Mean_Recall': results['mean_recall'],
                    'Std_Recall': results['std_recall'],
                    'K_Folds': results['k_folds']
                })

        kfold_df = pd.DataFrame(kfold_data)
        kfold_path = os.path.join(results_dir, f"kfold_cv_results_{timestamp}.csv")
        kfold_df.to_csv(kfold_path, index=False)
        print(f"K-fold CV results CSV saved: {kfold_path}")

        # Top performers summary
        top_performers = kfold_df.nlargest(10, 'Mean_Accuracy')
        top_performers_path = os.path.join(results_dir, f"top_performers_{timestamp}.csv")
        top_performers.to_csv(top_performers_path, index=False)
        print(f"Top performers CSV saved: {top_performers_path}")

        return single_split_df, kfold_df, top_performers

    return single_split_df, None, None


def print_final_results(all_results, all_kfold_results, best_results):
    """Print comprehensive final results"""

    print(f"\n{'=' * 80}")
    print("🎯 FINAL RESULTS SUMMARY 🎯")
    print(f"{'=' * 80}")

    # K-fold CV results table
    if all_kfold_results:
        print(f"\n📊 10-FOLD CROSS-VALIDATION RESULTS:")
        print("-" * 80)
        print(f"{'Feature':<12} {'Classifier':<18} {'Accuracy':<18} {'F1-Score':<18} {'Precision':<18} {'Recall':<18}")
        print("-" * 80)

        for feature_name, feature_results in all_kfold_results.items():
            for clf_name, results in feature_results.items():
                mean_acc = results['mean_accuracy']
                std_acc = results['std_accuracy']
                mean_f1 = results['mean_f1_score']
                std_f1 = results['std_f1_score']
                mean_prec = results['mean_precision']
                std_prec = results['std_precision']
                mean_rec = results['mean_recall']
                std_rec = results['std_recall']

                print(f"{feature_name:<12} {clf_name:<18} "
                      f"{mean_acc:.3f}±{std_acc:.3f}     "
                      f"{mean_f1:.3f}±{std_f1:.3f}     "
                      f"{mean_prec:.3f}±{std_prec:.3f}     "
                      f"{mean_rec:.3f}±{std_rec:.3f}")

    # Best performers section
    print(f"\n{'=' * 50}")
    print("🏆 BEST PERFORMERS 🏆")
    print(f"{'=' * 50}")

    if best_results['kfold_cv']:
        best_acc = best_results['kfold_cv']['best_accuracy']
        best_f1 = best_results['kfold_cv']['best_f1']

        print(f"\n🥇 HIGHEST ACCURACY (10-Fold CV):")
        print(f"   Model: {best_acc['classifier']} on {best_acc['feature']}")
        print(f"   Accuracy: {best_acc['mean_accuracy']:.4f} ± {best_acc['std_accuracy']:.4f}")
        print(f"   F1-Score: {best_acc['mean_f1_score']:.4f} ± {best_acc['std_f1_score']:.4f}")

        print(f"\n🥇 HIGHEST F1-SCORE (10-Fold CV):")
        print(f"   Model: {best_f1['classifier']} on {best_f1['feature']}")
        print(f"   F1-Score: {best_f1['mean_f1_score']:.4f} ± {best_f1['std_f1_score']:.4f}")
        print(f"   Accuracy: {best_f1['mean_accuracy']:.4f} ± {best_f1['std_accuracy']:.4f}")

    # Feature ranking
    if all_kfold_results:
        print(f"\n📈 FEATURE RANKING (by average accuracy):")
        print("-" * 40)
        feature_scores = {}
        for feature_name, feature_results in all_kfold_results.items():
            scores = [results['mean_accuracy'] for results in feature_results.values()]
            feature_scores[feature_name] = np.mean(scores)

        sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
        for rank, (feature, score) in enumerate(sorted_features, 1):
            print(f"   {rank}. {feature:<15}: {score:.4f}")

    # Classifier ranking
    if all_kfold_results:
        print(f"\n🤖 CLASSIFIER RANKING (by average accuracy):")
        print("-" * 40)
        classifier_scores = defaultdict(list)
        for feature_results in all_kfold_results.values():
            for clf_name, results in feature_results.items():
                classifier_scores[clf_name].append(results['mean_accuracy'])

        clf_avg_scores = {clf: np.mean(scores) for clf, scores in classifier_scores.items()}
        sorted_classifiers = sorted(clf_avg_scores.items(), key=lambda x: x[1], reverse=True)

        for rank, (classifier, score) in enumerate(sorted_classifiers, 1):
            print(f"   {rank}. {classifier:<15}: {score:.4f}")

    print(f"\n{'=' * 80}")


if __name__ == "__main__":
    # Configuration
    embeddings_directory = r"C:\Users\visha\Desktop\forest_research\new_studies\final_datasets_results\CNN_embeddings\embeddings"
    results_directory = "C:/Users/visha/Desktop/forest_research/new_studies/final_datasets_results/ensemble"

    # Validation
    if not os.path.exists(embeddings_directory):
        print(f"❌ Error: Embeddings directory '{embeddings_directory}' does not exist!")
        print("Please provide the correct path to your CNN k-fold embeddings.")
        exit(1)

    # Check for k-fold embedding files
    kfold_files = [f for f in os.listdir(embeddings_directory)
                   if 'kfold' in f.lower() and f.endswith('.pkl')]

    if not kfold_files:
        print(f"❌ Error: No k-fold embedding files found in '{embeddings_directory}'!")
        print("Looking for files with 'kfold' in the filename and .pkl extension.")
        print("\nFound files:")
        all_pkl_files = [f for f in os.listdir(embeddings_directory) if f.endswith('.pkl')]
        for f in sorted(all_pkl_files):
            print(f"  - {f}")
        exit(1)

    print(f"✅ Found {len(kfold_files)} k-fold embedding files:")
    for f in sorted(kfold_files):
        file_path = os.path.join(embeddings_directory, f)
        file_size = os.path.getsize(file_path) / 1024  # KB
        print(f"  - {f} ({file_size:.1f} KB)")

    # Create results directory
    if not os.path.exists(results_directory):
        print(f"📁 Creating results directory: {results_directory}")
        os.makedirs(results_directory, exist_ok=True)

    # Start analysis
    print(f"\n🚀 Starting enhanced ensemble learning pipeline...")
    print(f"📅 Analysis started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        results_dir = main(embeddings_directory, results_directory)

        print(f"\n✅ Analysis completed successfully!")
        print(f"📂 All results saved to: {results_dir}")
        print(f"🕒 Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Display final summary
        summary_file = os.path.join(results_dir, "ensemble_summary.json")
        if os.path.exists(summary_file):
            with open(summary_file, 'r') as f:
                summary = json.load(f)

            print(f"\n📋 QUICK SUMMARY:")
            print(f"   Models evaluated: {sum(len(fr) for fr in summary['kfold_results'].values())}")
            print(f"   Feature types: {len(summary['feature_combinations'])}")
            print(
                f"   Best accuracy: {max(r['mean_accuracy'] for fr in summary['kfold_results'].values() for r in fr.values()):.4f}")
            print(f"   Results timestamp: {summary['timestamp']}")

    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback

        traceback.print_exc()
        exit(1)

    print(f"\n🎉 ENSEMBLE LEARNING ANALYSIS COMPLETE! 🎉")
    print(f"Check the results directory for comprehensive reports and visualizations.")