<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-blue?logo=python" />
  <img src="https://img.shields.io/badge/Status-Active-success" />
  <img src="https://img.shields.io/badge/Open%20Source-Yes-brightgreen" />
  <img src="https://img.shields.io/github/stars/MoenishBaalan/AI-POWERED-EMOTION-RECOGNITION-USING-WEBCAM-AND-MICROPHONE?style=social" />
</p>
<h1 align="center">AI-Powered Emotion Recognition</h1>
<p align="center">
  Real-time Emotion Detection using Webcam and Microphone <br/>
  <i><b>Multimodal AI | Audio-Video Fusion</i></b>
</p>

---

##  Overview
This project implements a **real-time AI-powered emotion recognition system** that analyzes  
**facial expressions** via webcam and **speech signals** via microphone.

By combining **visual and audio modalities**, the system delivers more reliable emotion predictions  
compared to single-source emotion recognition approaches.

---

##  Objectives
- Capture live facial expressions using a webcam  
- Analyze speech emotions from microphone input  
- Process audio and video streams in parallel  
- Fuse both modalities to determine the final emotion  
- Display emotion results in real time  

---

##  System Workflow
1. Webcam captures live video frames  
2. Facial emotions are detected using computer vision  
3. Microphone records audio samples  
4. Audio features are extracted for emotion analysis  
5. Audio and video are processed simultaneously  
6. Final emotion is derived using fusion logic  
7. Emotion output is displayed on screen  

---

##  Tech Stack
| Category | Tools |
|--------|------|
| Programming | Python |
| Computer Vision | OpenCV |
| Facial Emotion | FER |
| Audio Processing | PyAudioAnalysis, SoundDevice |
| Utilities | NumPy, Multithreading |

---

##  Installation
```bash
pip install opencv-python fer sounddevice pyAudioAnalysis numpy
```

## Execution
```
python main.py
```
## Requirements

* Webcam enabled
* Microphone access granted

##  Output

It's available inside theassets/output.png

The system displays:

* Detected facial emotion with confidence score
* Speech-based emotion analysis
* Final fused emotion in real time

##  Future Enhancements

* Improve speech emotion classification accuracy
* Integrate deep learning models for audio analysis
* Deploy as a web-based application
* Store emotion history for analytics and insights

##  Author

Moenish Baalan

B.Tech – Information Technology



<p align="center"> <i><b> If you find this project interesting, feel free to star the repository ⭐ </i></b></p> 
