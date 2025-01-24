# Voice Cloning Using Deep Learning

This repository hosts the code and resources for the **Voice Cloning** project, which focuses on building a transformer-based voice cloning model leveraging deep learning techniques. The project employs advanced methods for natural language text processing and Mel spectrogram-based audio synthesis, achieving high-quality, speaker-specific voice outputs.

## Table of Contents
- [Introduction](#introduction)
- [Problem Statement](#problem-statement)
- [Methodology](#methodology)
- [Requirements](#requirements)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Results](#results)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [References](#references)

## Introduction
Voice cloning is the process of replicating a speaker’s voice using advanced neural networks. This project aims to build a Text-to-Speech (TTS) model capable of generating realistic speech for Female American English speakers, trained on text and paired audio datasets.

## Problem Statement
The project seeks to create an AI system that synthesizes speech from text, capturing speaker-specific nuances like intonation and style. The model incorporates attention mechanisms to align text and audio, producing realistic yet slightly robotic outputs.

## Methodology
The project integrates state-of-the-art deep learning techniques:
1. **Text Processing**: Utilizes Large Language Models (LLMs) to encode linguistic and phonetic features from text input.
2. **Feature Extraction from Mel-Spectrograms**: Converts audio into Mel spectrograms, serving as intermediate representations.
3. **Audio Synthesis**: Uses attention mechanisms and transformer-based architectures for generating Mel spectrograms, which are then converted to audio via vocoders.

<img width="649" alt="Screenshot 2024-11-12 at 12 47 16 PM" src="https://github.com/user-attachments/assets/0b240f6f-9eb8-4e60-bf08-b644d7ed3d4a">

## Requirements
To replicate this project, the following dependencies are required:
- Python 3.8+
- TensorFlow or PyTorch
- Librosa for audio processing
- NumPy, Matplotlib, and other standard libraries

### Datasets
The model is trained on publicly available datasets:
- [LJ Speech Dataset](https://keithito.com/LJ-Speech-Dataset/)
- [LibriSpeech ASR Corpus](https://www.openslr.org/12)
- [Mozilla Common Voice](https://commonvoice.mozilla.org/en/datasets)

## Model Architecture
The architecture is designed for high-quality voice cloning with these key components:
1. **Text Encoder**: Employs transformers and LLMs for text processing, generating robust linguistic representations.
2. **Attention Mechanisms**: Ensures proper alignment between text and audio features for natural speech synthesis.
3. **Mel Spectrogram Generator**: Synthesizes Mel spectrograms from encoded text features.
4. **Vocoder**: Converts spectrograms to time-domain audio using WaveNet-like or Griffin-Lim algorithms.

## Training
The model was trained on GPU clusters with the following configurations:
- **Optimizer**: Adam with learning rate scheduling.
- **Loss Function**: Mean Squared Error (MSE) for spectrogram prediction.
- **Metrics**: MOS (Mean Opinion Score) to evaluate naturalness.

## Results
The model achieved:
- **MOS**: 3.2 ± 0.2 for Female American English speakers.
- Speech output mimics speaker nuances but retains slight robotic characteristics due to phase reconstruction limitations.

## Limitations
- **Speaker Generalization**: Performance limited by dataset size and diversity.
- **Audio Quality**: Slightly robotic due to phase artifacts in the vocoder.

## Future Work
Planned enhancements include:
1. Training on larger, multi-speaker datasets for better generalization.
2. Exploring cutting-edge vocoders for improved audio quality.
3. Further optimizing text-to-audio pairing for naturalness.

## References
This project is inspired by and extends the following research:
1. [WaveNet: A Generative Model for Raw Audio](https://arxiv.org/abs/1609.03499)
2. [Tacotron: Towards End-to-End Speech Synthesis](https://arxiv.org/abs/1703.10135)
3. [Tacotron 2: Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions](https://arxiv.org/abs/1712.05884)
4. [MelNet: A Generative Model for Audio in the Frequency Domain](https://arxiv.org/abs/1906.01083)
