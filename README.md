# Wildlife Sound Identifier

Machine learning system for automatic wildlife species identification from environmental audio recordings. The system analyzes audio recordings of wildlife, extracts acoustic features, and predicts the most likely species using a trained machine learning model.

---

# Overview

Environmental audio monitoring is widely used in biodiversity research and conservation. This project demonstrates how machine learning can be applied to environmental sound recordings to identify wildlife species automatically.

Users can upload an audio recording and the system will:

1. Process the audio signal
2. Extract acoustic features
3. Generate a spectrogram
4. Predict the wildlife species using a trained classifier

The system is implemented using Python and deployed using Streamlit.

---

# Features

- Wildlife species identification from environmental audio
- Audio preprocessing and feature extraction
- Mel-spectrogram visualization
- Machine learning classification
- Interactive Streamlit interface
- Upload and analyze audio recordings

---

# Live Demo

You can test the application here:

[Wildlife Sound Identifier](https://wildlife-sound-identifier.streamlit.app/)

Upload an audio recording and view the predicted species along with spectrogram visualization.

---


## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/wildlife-sound-identifier.git
cd wildlife-sound-identifier
```

## Install dependencies:
```
pip install -r requirements.txt
```

To run the application:

Start the Streamlit interface:
```
streamlit run streamlit_app.py
```
