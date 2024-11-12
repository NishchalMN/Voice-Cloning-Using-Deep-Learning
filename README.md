# Voice Cloning Using Deep Learning

This repository hosts the code and resources for the **Voice Cloning** project, which aims to create a voice cloning model using deep learning techniques. The project focuses on developing a custom Text-to-Speech (TTS) model that can generate natural-sounding speech for a variety of speakers by training on text and corresponding audio datasets.

## Table of Contents
- [Introduction](#introduction)
- [Problem Statement](#problem-statement)
- [Methodology](#methodology)
- [Requirements](#requirements)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Results](#results)
- [Future Work](#future-work)
- [References](#references)

## Introduction
Voice cloning is the replication of a person’s voice using deep neural networks. The goal of this project is to build a customized TTS model capable of generating natural speech, trained on speaker-specific datasets to produce voice outputs similar to the speaker in question.

## Problem Statement
The project involves creating an AI system that can mimic a person's voice by analyzing audio recordings and text transcripts. This model synthesizes speech directly from (text, audio) pairs, achieving a high-quality speech synthesis that captures speaker nuances, intonation, and style.

## Methodology
The project uses a deep learning approach that involves:
1. **Text Encoding**: Processing input text to generate linguistic and phonetic representations.
2. **Feature Extraction from Mel-Spectrogram**: Converting audio into Mel-spectrograms, which serve as intermediate representations in the pipeline.
3. **Audio Generation**: Converting Mel-spectrograms to audio using algorithms like Griffin-Lim or a vocoder for high-quality output.
   
<img width="649" alt="Screenshot 2024-11-12 at 12 47 16 PM" src="https://github.com/user-attachments/assets/0b240f6f-9eb8-4e60-bf08-b644d7ed3d4a">


The model architecture includes convolutional and recurrent neural networks, particularly leveraging:
- A **Text Encoder** for linguistic features
- A **Feature Extractor** for Mel-spectrograms
- **Concatenation and Convolutional Layers** to merge features and produce the final spectrogram

## Requirements
To run this project, the following dependencies are required:
- **Python 3.8+**
- **TensorFlow**
- **Keras**
- **Librosa**
- **NumPy**
- **Matplotlib**

### Datasets
The model is trained on audio datasets with paired text transcripts:
- [LJ Speech Dataset](https://keithito.com/LJ-Speech-Dataset/)
- [LibriSpeech ASR Corpus](https://www.openslr.org/12)
- [Mozilla Common Voice](https://commonvoice.mozilla.org/en/datasets)

## Model Architecture
The model architecture follows a three-stage approach:
1. **Text Encoder**: Encodes input text with 1-D convolutional layers and Bi-directional LSTM to capture linguistic features.
2. **Feature Extraction from Mel-Spectrogram**: Converts audio input to Mel-spectrograms using parameters like sample rate, FFT window, and Mel bands.
3. **Concatenation and Audio Generation**: Combines text and audio features, generating Mel-spectrograms that are then converted to audio using Griffin-Lim or vocoders.

<img width="783" alt="Screenshot 2024-11-12 at 12 46 37 PM" src="https://github.com/user-attachments/assets/065b5cd6-7c51-460e-b806-5c494de8dc8d">

The training uses the Adam optimizer with Mean Squared Error (MSE) and Mean Absolute Error (MAE) as metrics to evaluate model performance.

## Results

The model achieved a Mean Opinion Score (MOS) of 3.2 ± 0.2, demonstrating close-to-human naturalness for single-speaker synthesis. Improvements in model performance can be achieved with increased computing resources and larger datasets.

## Limitations
- Limited speaker generalization due to a small, single-speaker dataset.
- Noise in the output due to phase reconstruction losses.

## Future Work

Future improvements could include:
	•	Training on a larger multi-speaker dataset to enhance generalization.
	•	Employing state-of-the-art vocoders for higher audio quality.
	•	Optimizing the architecture to reduce computational load and improve synthesis speed.

## References
This project builds upon the following works:
1. [WaveNet: A Generative Model for Raw Audio](https://arxiv.org/abs/1609.03499)
2. [Tacotron: Towards End-to-End Speech Synthesis](https://arxiv.org/abs/1703.10135)
3. [Tacotron 2: Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions](https://arxiv.org/abs/1712.05884)
4. [MelNet: A Generative Model for Audio in the Frequency Domain](https://arxiv.org/abs/1906.01083)
  
