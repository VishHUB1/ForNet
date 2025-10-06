ForNet
Overview
ForNet is a two-stage CNN and ensemble learning system designed for forest sound classification. It utilizes MFCC (Mel-Frequency Cepstral Coefficients) features extracted from the FSMS dataset, which comprises 2,263 audio clips sourced from ESC-50 and UrbanSound8K datasets. The system is capable of detecting anomalies such as gunshots and wood sawing with an impressive 95% accuracy. This project enhances forest surveillance efforts, particularly in combating illegal logging by providing an effective audio-based monitoring solution.
Project Structure
The repository is organized as follows:

Root Directory: Contains code and scripts for handling the FCS22 dataset. These files focus on processing and analyzing the FCS22 dataset for forest sound classification tasks, including feature extraction and model training tailored to this specific dataset.
new_studies Folder: This subfolder contains advanced experiments and code related to the FSM5 dataset (a variant or subset of FSMS with 5 distinct classes). The focus here is on deeper analysis, including CNN embeddings, ensemble methods, dataset augmentation, class balancing, and visualizations to understand feature representations.

Inside the new_studies Folder
The new_studies folder is dedicated to exploratory and enhanced studies on the FSM5 dataset. It includes scripts for CNN-based embeddings, ensemble learning, dataset analysis, and visualizations. The FSM5 dataset consists of 5 classes, representing different forest sounds and anomalies (e.g., natural sounds, gunshots, wood sawing, etc.). Key aspects covered:

Dataset Analysis: Initial analysis is performed to understand the needs for data augmentation and class balancing. This helps identify imbalances in class distributions, potential noise in audio clips, and opportunities for augmentation techniques (e.g., adding noise, time-stretching) to improve model robustness.
Execution Order:

Run CNN Embeddings Code First: Start with the CNN embeddings script (e.g., cnn_embeddings.py or similar). This code processes the FSM5 dataset to generate embeddings using a convolutional neural network. These embeddings capture high-level feature representations from the MFCC inputs.
Followed by Ensemble Code: After generating embeddings, run the ensemble script (e.g., ensemble.py or similar). This combines multiple models (e.g., via voting or stacking) to classify the sounds, leveraging the embeddings for improved accuracy on anomaly detection.


CNN Embedding Visualizations: To understand feature representations:

Before Embeddings: Visualizations of raw MFCC features (e.g., spectrograms or heatmaps) to show initial audio characteristics.
After Embeddings: t-SNE or PCA-based visualizations of the CNN-generated embeddings to illustrate how the model clusters similar sounds and separates anomalies. This helps in evaluating the effectiveness of the CNN in learning discriminative features.


Different Feature Representations: The project explores various feature types beyond basic MFCCs, such as:

Spectrograms for time-frequency analysis.
Delta and delta-delta MFCCs for capturing dynamic changes in audio.
Embeddings from pre-trained models (if integrated) for transfer learning.
These comparisons highlight how different representations impact classification performance, with MFCCs chosen as the primary due to their efficiency in environmental sound classification.



Installation

Clone the repository:
textgit clone https://github.com/VishHUB1/ForNet.git

Install required dependencies (assuming Python environment):
textpip install -r requirements.txt
