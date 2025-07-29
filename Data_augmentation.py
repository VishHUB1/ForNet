#input_directory = r"C:\Users\visha\Desktop\forest_research\FSC22_classes"  # Replace with actual path
#output_directory = "C:/Users/visha/Desktop/forest_research/Augmented_Dataset"  # Replace with desired output path
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


def get_dataset_stats(input_dir):
    """Calculate statistics of the dataset."""
    stats = []
    for class_name in ['Animals', 'Nature', 'Anomalous']:
        class_path = os.path.join(input_dir, class_name)
        if not os.path.exists(class_path):
            continue
        for subclass in os.listdir(class_path):
            subclass_path = os.path.join(class_path, subclass)
            if os.path.isdir(subclass_path):
                count = len([f for f in os.listdir(subclass_path) if f.endswith(('.wav', '.mp3'))])
                stats.append({'Class': class_name, 'Subclass': subclass, 'Count': count})
    return pd.DataFrame(stats)


def plot_dataset_stats(stats_df, output_path, title):
    """Plot dataset distribution."""
    plt.figure(figsize=(12, 6))
    sns.barplot(data=stats_df, x='Subclass', y='Count', hue='Class')
    plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f"{title.lower().replace(' ', '_')}.png"))
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


def augment_audio(input_dir, output_dir, target_count=1000):
    """Apply augmentation techniques to balance the dataset."""
    # Define augmentation pipeline (including techniques from the paper and additional ones)
    augment = Compose([
        AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
        TimeStretch(min_rate=0.8, max_rate=1.2, p=0.5),
        PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
        Shift(min_shift=-0.5, max_shift=0.5, p=0.5),
        Gain(min_gain_db=-12, max_gain_db=12, p=0.5),
        # Additional augmentations
        AddGaussianNoise(min_amplitude=0.01, max_amplitude=0.05, p=0.3),  # Stronger noise
        TimeStretch(min_rate=0.7, max_rate=1.3, p=0.3),  # Wider stretch range
    ])

    stats_before = get_dataset_stats(input_dir)
    plot_dataset_stats(stats_before, os.path.join(output_dir, 'metrics_and_visualizations'),
                       'Dataset Distribution Before Augmentation')

    # Copy original files
    for class_name in ['Animals', 'Nature', 'Anomalous']:
        class_input_path = os.path.join(input_dir, class_name)
        class_output_path = os.path.join(output_dir, class_name)
        if not os.path.exists(class_input_path):
            continue

        for subclass in os.listdir(class_input_path):
            subclass_input_path = os.path.join(class_input_path, subclass)
            subclass_output_path = os.path.join(class_output_path, subclass)
            if not os.path.isdir(subclass_input_path):
                continue

            audio_files = [f for f in os.listdir(subclass_input_path) if f.endswith(('.wav', '.mp3'))]
            current_count = len(audio_files)

            # Copy original files
            for audio_file in audio_files:
                shutil.copy(
                    os.path.join(subclass_input_path, audio_file),
                    os.path.join(subclass_output_path, audio_file)
                )

            # Apply augmentations to reach target count
            augmentations_needed = target_count - current_count
            if augmentations_needed > 0:
                for i in range(augmentations_needed):
                    # Randomly select an audio file to augment
                    audio_file = np.random.choice(audio_files)
                    audio_path = os.path.join(subclass_input_path, audio_file)

                    # Load audio
                    audio, sr = librosa.load(audio_path, sr=None)

                    # Apply augmentation
                    augmented_audio = augment(audio, sample_rate=sr)

                    # Save augmented audio
                    output_filename = f"aug_{i}_{audio_file}"
                    sf.write(
                        os.path.join(subclass_output_path, output_filename),
                        augmented_audio,
                        sr
                    )

                    # Generate MFCC comparison for first augmentation of each subclass
                    if i == 0:
                        plot_mfcc_comparison(
                            audio,
                            augmented_audio,
                            sr,
                            os.path.join(output_dir, 'metrics_and_visualizations'),
                            f"{class_name}_{subclass}_{audio_file}"
                        )

    # Generate stats after augmentation
    stats_after = get_dataset_stats(output_dir)
    plot_dataset_stats(stats_after, os.path.join(output_dir, 'metrics_and_visualizations'),
                       'Dataset Distribution After Augmentation')

    # Save stats to CSV
    stats_before.to_csv(os.path.join(output_dir, 'metrics_and_visualizations', 'stats_before.csv'), index=False)
    stats_after.to_csv(os.path.join(output_dir, 'metrics_and_visualizations', 'stats_after.csv'), index=False)

    # Generate summary statistics
    summary = pd.DataFrame({
        'Class': stats_before['Class'],
        'Subclass': stats_before['Subclass'],
        'Count_Before': stats_before['Count'],
        'Count_After': stats_after['Count']
    })
    summary.to_csv(os.path.join(output_dir, 'metrics_and_visualizations', 'summary_stats.csv'), index=False)


if __name__ == "__main__":
    input_directory = r"C:\Users\visha\Desktop\forest_research\FSC22_classes"  # Replace with actual path
    output_directory = "C:/Users/visha/Desktop/forest_research/Augmented_Dataset"  # Replace with desired output path

    create_directory_structure(input_directory, output_directory)
    augment_audio(input_directory, output_directory, target_count=1000)