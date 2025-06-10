# 🧍‍♂️ Person Re-identification using YOLOv11 + StrongSORT

This project performs **Person Re-Identification** in videos using **YOLOv11** for object detection and **StrongSORT** for multi-object tracking. The system is capable of identifying and tracking players (class 2) across video frames in real-time.

---

## ⚙️ Environment Setup

Create and activate the conda environment:

```bash
conda create -n strongsort python=3.8 -y
conda activate strongsort
pip install torch torchvision torchaudio
pip install opencv-python
pip install scipy
pip install scikit-learn
pip install ultralytics

## 📜 Run the Script

To run the person re-identification script:

```bash
python prereid.py


and also please change the paths of weight in the sccript manually and use best.pt weights
