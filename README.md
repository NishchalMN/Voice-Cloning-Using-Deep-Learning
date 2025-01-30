# Voice Cloning Using Deep Learning

This repository contains the code and resources for a **few-shot voice cloning system**, capable of replicating a speaker’s voice using only a **short audio clip and text**. The system extracts a speaker embedding using **GE2E**, generates **Mel spectrograms with Tacotron 2**, and synthesizes waveforms using a **WaveNet vocoder**, achieving a **MOS of 3.2/5** for Female American speakers.  

---

## Table of Contents  
- [Introduction](#introduction)  
- [Problem Statement](#problem-statement)  
- [Methodology](#methodology)  
- [Requirements](#requirements)
- [Datasets](#datasets)
- [Model Architecture](#model-architecture)
- [Training](#training)  
- [Results](#results)  
- [Limitations](#limitations)  
- [Future Work](#future-work)  
- [References](#references)  

---

## Introduction  
Voice cloning enables AI models to **synthesize speech in a specific person’s voice** using only a short reference clip. This project implements a **few-shot speaker-adaptive text-to-speech (TTS) system** using **deep learning models** to generate high-fidelity speech from new text input.  

---

## Problem Statement  
The system aims to:  
- **Extract a speaker’s voice characteristics** from a short audio clip.  
- **Synthesize new speech** in that speaker’s voice given any text input.  
- **Ensure high fidelity** while optimizing inference time.  

---

## Methodology  

### 1. Speaker Embedding Extraction (GE2E)  
- Uses **Generalized End-to-End Speaker Verification (GE2E)** to generate a **fixed-dimensional speaker embedding** from a short input clip.  
- This embedding captures **pitch, tone, and speaking style**, allowing adaptation to new speakers.  

### 2. Text-to-Mel Spectrogram Generation (Tacotron 2)  
- Converts input text into **Mel spectrograms**, conditioned on the extracted speaker embedding.  
- **Tacotron 2** uses an attention-based seq2seq model to align phonemes with audio frames.  

### 3. Mel Spectrogram to Waveform Synthesis (WaveNet Vocoder)  
- The **WaveNet vocoder** synthesizes time-domain waveforms from the Mel spectrograms.  
- Fine-tuned to **reduce inference latency** while maintaining high-quality speech synthesis.  

---

## Requirements  

### Installing Dependencies  
To install the required dependencies, run the following command:  

```bash
pip install torch torchaudio numpy librosa matplotlib soundfile tqdm tensorflow
```
<img width="649" alt="Screenshot 2024-11-12 at 12 47 16 PM" src="https://github.com/user-attachments/assets/0b240f6f-9eb8-4e60-bf08-b644d7ed3d4a">

## Datasets
The model is trained on publicly available datasets:
- [LJ Speech Dataset](https://keithito.com/LJ-Speech-Dataset/)
- [LibriSpeech ASR Corpus](https://www.openslr.org/12)
- [Mozilla Common Voice](https://commonvoice.mozilla.org/en/datasets)

## Model Architecture
The architecture is designed for high-quality voice cloning with these key components:
1. **Speaker Encoder (GE2E)**: Extracts a fixed-dimensional embedding from a short audio clip.
2. **Text-to-Spectrogram (Tacotron 2)**: Generates a Mel spectrogram from input text, conditioned on the speaker embedding.
3. **Neural Vocoder (WaveNet)**: Converts the Mel spectrogram into a high-fidelity waveform.

- Architecture Diagram
```
[Audio Clip] --> [GE2E Speaker Encoder] --> [Speaker Embedding]
                            |
                            v
                 [Tacotron 2: Text-to-Mel]
                            |
                            v
                 [WaveNet: Mel-to-Waveform]
                            |
                            v
                     [Generated Speech]
```
## Training
The model was trained on GPU clusters with the following configurations:
- **Optimizer**: Adam with learning rate scheduling.
- **Loss Function**:
    Spectrogram prediction → Mean Squared Error (MSE)
	  Speaker embedding training → Contrastive loss
    WaveNet vocoder → L1 loss
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
1.	Experiment with non-autoregressive models (e.g., FastSpeech 2) for faster inference.
2.	Improve WaveNet latency by pruning and quantization.
3.	Train on larger multi-speaker datasets for better speaker generalization.

## References
This project is inspired by and extends the following research:
1. [Generalized End-to-End Speaker Verification (GE2E)](https://arxiv.org/abs/1710.10467)
2. [Tacotron 2: Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions](https://arxiv.org/abs/1712.05884)
3. [MelNet: A Generative Model for Audio in the Frequency Domain](https://arxiv.org/abs/1906.01083)
