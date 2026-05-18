# Agriculture AI - Plant Disease Detection

An AI-powered web application built with Streamlit and TensorFlow that detects 38 different plant conditions (including healthy plants) from leaf images. The application uses a custom-trained DenseNet121 model and provides confidence scores, professional agricultural recommendations, and PDF report generation.

## Features
- **Accurate Disease Detection**: Powered by a custom DenseNet121 neural network.
- **Top 3 Predictions**: Shows the most likely conditions with confidence scores.
- **Farmer Action Plan**: Detailed, actionable advice including recommended pesticides, organic treatments, and best practices based on the detected condition.
- **Downloadable PDF Report**: Export the results in a professional PDF format.

## Local Setup

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd plant-disease-detection
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application:**
   ```bash
   python -m streamlit run app.py
   ```

## Deployment on Streamlit Community Cloud
This repository is pre-configured for Streamlit Community Cloud:
1. Push this repository to GitHub.
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io).
3. Click "New app".
4. Select the repository, branch, and set the main file path to `app.py`.
5. Click "Deploy". The platform will automatically install packages from `requirements.txt`.
