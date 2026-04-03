# NeuroTwin: EEG Stress Detection Digital Twin

**NeuroTwin** is a real‑time stress detection system that turns EEG signals into an interactive 3D brain model. It classifies four stress levels (natural, low, mid, high) with **99.08% accuracy** and visualises activation of the Prefrontal Cortex (PFC), Amygdala, and Anterior Cingulate Cortex (ACC).

## Features

- Upload any `.txt` EEG trial file (Cognitive Load Dataset format)
- Slide through 2‑second windows to watch stress evolve
- Interactive 3D brain (fsaverage pial surface) – colours change with stress
- Nerve pressure metric + wellness suggestions (deep breathing, break)

## How to Run

1. **Install dependencies**
   ```bash
   pip install streamlit xgboost numpy pandas PyWavelets plotly nilearn
