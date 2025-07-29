import os
import numpy as np
import librosa
import soundfile as sf
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift, Gain
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import shutil
from collections import defaultdict


def create_directory_structure(input_dir, output_dir):
    """Create output directory structure mirroring input directory."""
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    for class_name in ['Animals', 'Nature', 'Anomalous']:
        class_input_path = os.path.join(input_dir, class_name)
        class_output_path = os.path.join(output_dir, class_name)
        os.makedirs(class_output_path)

        if os.path.exists(class_input_path):
            for subclass in os.listdir(class_input_path):
                subclass_input_path = os.path.join(class_input_path, subclass)
                subclass_output_path = os.path.join(class_output_path, subclass)
                if os.path.isdir(subclass_input_path):
                    os.makedirs(subclass_output_path)

    os.makedirs(os.path.join(output_dir, 'metrics_and_visualizations'))


def get_class_stats(input_dir):
    """Calculate statistics of the dataset by broad classes and subclasses."""
    class_files = defaultdict(lambda: defaultdict(list))
    detailed_stats = []

    for class_name in ['Animals', 'Nature', 'Anomalous']:
        class_path = os.path.join(input_dir, class_name)

        if not os.path.exists(class_path):
            continue

        for subclass in os.listdir(class_path):
            subclass_path = os.path.join(class_path, subclass)
            if os.path.isdir(subclass_path):
                audio_files = [f for f in os.listdir(subclass_path) if f.endswith(('.wav', '.mp3'))]
                count = len(audio_files)
                detailed_stats.append({'Class': class_name, 'Subclass': subclass, 'Count': count})

                # Store file paths with their subclass info
                for audio_file in audio_files:
                    class_files[class_name][subclass].append(os.path.join(subclass_path, audio_file))

    return pd.DataFrame(detailed_stats), class_files


def get_class_totals(class_files):
    """Get total count for each class."""
    totals = {}
    for class_name, subclasses in class_files.items():
        total = sum(len(files) for files in subclasses.values())
        totals[class_name] = total
    return totals


def plot_class_distribution(stats_df, output_path, title):
    """Plot class distribution."""
    # Aggregate by class
    class_totals = stats_df.groupby('Class')['Count'].sum().reset_index()

    plt.figure(figsize=(10, 6))
    sns.barplot(data=class_totals, x='Class', y='Count')
    plt.title(title)
    plt.ylabel('Number of Audio Files')

    # Add count labels on bars
    for i, v in enumerate(class_totals['Count']):
        plt.text(i, v + 10, str(v), ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f"{title.lower().replace(' ', '_')}.png"))
    plt.close()


def plot_detailed_distribution(stats_df, output_path, title):
    """Plot detailed subclass distribution."""
    plt.figure(figsize=(15, 8))
    sns.barplot(data=stats_df, x='Subclass', y='Count', hue='Class')
    plt.title(title)
    plt.xticks(rotation=45)
    plt.ylabel('Number of Audio Files')
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f"{title.lower().replace(' ', '_')}_detailed.png"))
    plt.close()


def extract_mfcc(audio, sample_rate, n_mfcc=13):
    """Extract MFCC features."""
    return librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)


def plot_mfcc_comparison(original_audio, augmented_audio, sample_rate, output_path, filename):
    """Plot MFCC comparison before and after augmentation."""
    original_mfcc = extract_mfcc(original_audio, sample_rate)
    augmented_mfcc = extract_mfcc(augmented_audio, sample_rate)

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    librosa.display.specshow(original_mfcc, x_axis='time', sr=sample_rate)
    plt.colorbar()
    plt.title('Original MFCC')

    plt.subplot(1, 2, 2)
    librosa.display.specshow(augmented_mfcc, x_axis='time', sr=sample_rate)
    plt.colorbar()
    plt.title('Augmented MFCC')

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f'mfcc_comparison_{filename}.png'))
    plt.close()


def distribute_files_proportionally(class_files, target_count):
    """Distribute target files proportionally across subclasses."""
    distribution = {}

    for class_name, subclasses in class_files.items():
        total_files = sum(len(files) for files in subclasses.values())
        class_distribution = {}

        if total_files == 0:
            continue

        # Calculate proportional distribution
        remaining_target = target_count
        subclass_items = list(subclasses.items())

        for i, (subclass, files) in enumerate(subclass_items):
            if i == len(subclass_items) - 1:  # Last subclass gets remaining
                class_distribution[subclass] = remaining_target
            else:
                proportion = len(files) / total_files
                allocated = int(target_count * proportion)
                class_distribution[subclass] = max(1, allocated)  # At least 1 file per subclass
                remaining_target -= allocated

        distribution[class_name] = class_distribution

    return distribution


def augment_dataset_by_class(input_dir, output_dir, target_per_class=1000):
    """Apply augmentation techniques to balance the dataset by broad classes while maintaining subclass structure."""

    # Define augmentation pipeline
    augment = Compose([
        AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
        TimeStretch(min_rate=0.8, max_rate=1.2, p=0.5),
        PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
        Shift(min_shift=-0.5, max_shift=0.5, p=0.5),
        Gain(min_gain_db=-12, max_gain_db=12, p=0.5),
        # Additional augmentations for variety
        AddGaussianNoise(min_amplitude=0.01, max_amplitude=0.05, p=0.3),
        TimeStretch(min_rate=0.7, max_rate=1.3, p=0.3),
    ])

    # Get original dataset statistics
    stats_before, class_files = get_class_stats(input_dir)
    class_totals = get_class_totals(class_files)

    print("Original Dataset Statistics:")
    for class_name, total in class_totals.items():
        print(f"{class_name}: {total} files")
        subclasses = class_files[class_name]
        for subclass, files in subclasses.items():
            print(f"  └── {subclass}: {len(files)} files")

    # Plot before stats
    plot_class_distribution(stats_before, os.path.join(output_dir, 'metrics_and_visualizations'),
                            'Dataset Distribution Before Augmentation')
    plot_detailed_distribution(stats_before, os.path.join(output_dir, 'metrics_and_visualizations'),
                               'Detailed Dataset Distribution Before Augmentation')

    # Calculate target distribution for each subclass
    target_distribution = distribute_files_proportionally(class_files, target_per_class)

    augmented_stats = []
    mfcc_comparison_count = 0  # Limit MFCC plots

    # Process each class
    for class_name in ['Animals', 'Nature', 'Anomalous']:
        print(f"\nProcessing {class_name} class...")

        if class_name not in class_files:
            print(f"Warning: {class_name} class not found in input directory")
            continue

        subclasses = class_files[class_name]
        class_target_dist = target_distribution[class_name]

        # Process each subclass
        for subclass, original_files in subclasses.items():
            print(f"\n  Processing subclass: {subclass}")

            current_count = len(original_files)
            target_count = class_target_dist[subclass]
            subclass_output_path = os.path.join(output_dir, class_name, subclass)

            print(f"    Original: {current_count}, Target: {target_count}")

            if current_count == 0:
                print(f"    Warning: No files found in {subclass}")
                continue

            # Handle different scenarios
            if target_count <= current_count:
                # Downsample: randomly select target_count files
                selected_files = np.random.choice(original_files, size=target_count, replace=False)
                for file_path in selected_files:
                    filename = os.path.basename(file_path)
                    shutil.copy(file_path, os.path.join(subclass_output_path, filename))
                print(f"    Downsampled to {target_count} files")

            else:
                # Upsample: copy all original files + generate augmented files
                # First copy all original files
                for file_path in original_files:
                    filename = os.path.basename(file_path)
                    shutil.copy(file_path, os.path.join(subclass_output_path, filename))

                # Calculate how many augmentations we need
                augmentations_needed = target_count - current_count
                print(f"    Generating {augmentations_needed} augmented samples...")

                # Generate augmented samples
                for i in range(augmentations_needed):
                    # Randomly select a file to augment
                    source_file = np.random.choice(original_files)

                    try:
                        # Load audio
                        audio, sr = librosa.load(source_file, sr=None)

                        # Apply augmentation
                        augmented_audio = augment(audio, sample_rate=sr)

                        # Create output filename
                        original_filename = os.path.basename(source_file)
                        name, ext = os.path.splitext(original_filename)
                        output_filename = f"aug_{i:04d}_{name}{ext}"

                        # Save augmented audio
                        sf.write(
                            os.path.join(subclass_output_path, output_filename),
                            augmented_audio,
                            sr
                        )

                        # Generate MFCC comparison for first few augmentations (limit to avoid too many plots)
                        if mfcc_comparison_count < 6:  # Only first 6 total across all classes
                            plot_mfcc_comparison(
                                audio,
                                augmented_audio,
                                sr,
                                os.path.join(output_dir, 'metrics_and_visualizations'),
                                f"{class_name}_{subclass}_{i}_{name}"
                            )
                            mfcc_comparison_count += 1

                    except Exception as e:
                        print(f"    Error processing {source_file}: {e}")
                        continue

                    if (i + 1) % 50 == 0:
                        print(f"    Generated {i + 1}/{augmentations_needed} augmentations")

            # Verify final count
            final_count = len([f for f in os.listdir(subclass_output_path) if f.endswith(('.wav', '.mp3'))])
            augmented_stats.append({'Class': class_name, 'Subclass': subclass, 'Count': final_count})
            print(f"    Final count: {final_count}")

    # Generate final statistics
    final_stats_df = pd.DataFrame(augmented_stats)

    # Calculate class totals for final dataset
    final_class_totals = final_stats_df.groupby('Class')['Count'].sum().reset_index()

    # Plot after stats
    plot_class_distribution(final_stats_df, os.path.join(output_dir, 'metrics_and_visualizations'),
                            'Dataset Distribution After Augmentation')
    plot_detailed_distribution(final_stats_df, os.path.join(output_dir, 'metrics_and_visualizations'),
                               'Detailed Dataset Distribution After Augmentation')

    # Save statistics to CSV
    stats_before.to_csv(os.path.join(output_dir, 'metrics_and_visualizations', 'detailed_stats_before.csv'),
                        index=False)
    final_stats_df.to_csv(os.path.join(output_dir, 'metrics_and_visualizations', 'detailed_stats_after.csv'),
                          index=False)

    # Create class-level summary
    class_summary_before = stats_before.groupby('Class')['Count'].sum().reset_index()
    class_summary_before.columns = ['Class', 'Count_Before']

    class_summary_after = final_stats_df.groupby('Class')['Count'].sum().reset_index()
    class_summary_after.columns = ['Class', 'Count_After']

    summary = pd.merge(class_summary_before, class_summary_after, on='Class', how='outer')
    summary = summary.fillna(0)
    summary.to_csv(os.path.join(output_dir, 'metrics_and_visualizations', 'class_summary.csv'), index=False)

    print("\n" + "=" * 50)
    print("FINAL DATASET SUMMARY")
    print("=" * 50)
    print("Class-level totals:")
    for _, row in summary.iterrows():
        print(f"{row['Class']}: {int(row['Count_Before'])} → {int(row['Count_After'])}")

    print(f"\nTotal files: {int(summary['Count_After'].sum())}")
    print(f"Augmented dataset saved to: {output_dir}")

    # Print subclass distribution for verification
    print("\nSubclass distribution in final dataset:")
    for class_name in ['Animals', 'Nature', 'Anomalous']:
        class_data = final_stats_df[final_stats_df['Class'] == class_name]
        if not class_data.empty:
            print(f"\n{class_name}:")
            for _, row in class_data.iterrows():
                print(f"  └── {row['Subclass']}: {row['Count']} files")


if __name__ == "__main__":
    input_directory = r"C:\Users\visha\Desktop\forest_research\FSC22_classes"
    output_directory = r"C:\Users\visha\Desktop\forest_research\Augmented_Datasetv2"

    # Set target number of samples per class (Animals, Nature, Anomalous)
    target_samples_per_class = 1125

    print(f"Starting dataset balancing...")
    print(f"Target samples per class: {target_samples_per_class}")
    print("Maintaining original subclass structure...")

    create_directory_structure(input_directory, output_directory)
    augment_dataset_by_class(input_directory, output_directory, target_per_class=target_samples_per_class)