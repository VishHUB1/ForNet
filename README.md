# ForNet

ForNet is a two-stage CNN and ensemble learning system for forest sound classification using MFCC features from the FSMS dataset (2,664 clips from ESC-50 and UrbanSound8K). It detects anomalies like gunshots and wood sawing with 95% accuracy, aiding forest surveillance against illegal logging.

## Project Structure

- **Root**: Handles FCS22 dataset with main scripts for feature extraction and CNN training.
- **new_studies**: Focuses on FSM5 dataset (5 classes). Run `cnn_embeddings.py` first for feature visualizations (before/after embeddings), then `ensemble.py` for classification. Includes dataset analysis for augmentation and class balancing.

## Installation

Clone repo: `git clone https://github.com/VishHUB1/ForNet.git`  
Install deps: `pip install -r requirements.txt`

## Usage

- FCS22: `python main_fcs22.py`
- FSM5: `cd new_studies`, then `python cnn_embeddings.py`, followed by `python ensemble.py`

## Datasets

Download ESC-50/UrbanSound8K/GoogleDatasets; process via scripts.

T
