<p align="center">
  <b>AI-Powered Emotion Recognition using Webcam and Microphone</b><br>
  Real-time Multimodal Emotion Detection System
</p>

## 📌 Project Overview
This project presents a real-time **AI-powered emotion recognition system** that detects human emotions by analyzing **facial expressions** through a webcam and **speech signals** through a microphone.  
By combining both visual and audio modalities, the system improves prediction reliability compared to single-source emotion recognition systems.

The project demonstrates the practical application of **multimodal artificial intelligence**, real-time processing, and emotion-aware computing.

---

## 🎯 Objectives
- Capture real-time facial expressions using a webcam
- Analyze speech emotions from microphone input
- Process audio and video streams in parallel using multithreading
- Fuse audio and video emotion results to generate a final emotion output
- Display detected emotions in real time

---

## 🧠 System Workflow
1. Webcam captures live video frames
2. Facial emotions are detected using computer vision techniques
3. Microphone records audio samples
4. Audio features are extracted for emotion analysis
5. Audio and video emotions are processed simultaneously
6. Final emotion is determined using fusion logic
7. Emotion result is displayed on screen

---

## 🛠 Technologies Used
- **Python**
- **OpenCV** – Video capture and image processing
- **FER (Facial Emotion Recognition)** – Facial emotion detection
- **PyAudioAnalysis** – Audio feature extraction
- **SoundDevice** – Real-time audio recording
- **NumPy**
- **Multithreading**

---

## ⚙️ Installation
Clone the repository and install the required dependencies:

```bash
git clone https://github.com/MoenishBaalan/AI-POWERED-EMOTION-RECOGNITION-USING-WEBCAM-AND-MICROPHONE.git
cd AI-POWERED-EMOTION-RECOGNITION-USING-WEBCAM-AND-MICROPHONE
pip install opencv-python fer sounddevice pyAudioAnalysis numpy
