# Malicious Domain Detector - A Chrome Extension

This project is a practical reimplementation of the [MADONNA](https://doi.org/10.1007/978-3-031-56326-3_20) paper — a lightweight system for detecting malicious domains using statistical features only. It consists of:

- A trained neural network using TensorFlow/Keras
- A Flask API server to serve the model
- A Chrome Extension to analyze and warn about the domain of the current website

---

## Project Structure

```
├── app.py                          # Flask API backend
├── extract_and_predict.py          # Feature extraction + prediction logic
├── model_training.py               # Train script
├── prune_and_quantize.py           # Prune and Quantize script
├── model_pruned_quantized.tflite   # Pruned and Quantized model
├── data/                           # Reference datasets
├── extension/
    ├── manifest.json
    ├── popup.html
    ├── script.js
    └── icon.png
├── README.md                       # The file you're reading
├── requirements.txt                # All the requirements
```

---

## Quick Start (Using Hosted API)

The backend is already hosted (on [Render](https://malicious-domain-checker.onrender.com)), so you don’t need to run any server locally.

### 1. Clone the Repo

```bash
git clone https://github.com/Neit272/Malicious-Domain-Checker.git
```

### 2. Load Chrome Extension

1. Open `chrome://extensions/`
2. Enable **Developer Mode** (top right corner)
3. Click **“Load unpacked”**
4. Select the folder `extension/` from this project

Now whenever you visit a website and click on the extension, the extension will:
- Extract the domain
- Send it to the hosted backend
- Receive a probability + label (benign/malicious)
- Display the result in the popup

> Make sure the backend link in `script.js` matches the deployed API (see next section if you want to host locally).

---

## Run Backend Locally (Optional)

If you want to host the model inference API yourself:

### 1. Install Dependencies

Python 3.10 is recommended to customize and run all the code. Create virtual environment, then install the requirements:

```bash
pip install -r requirements.txt
```
> I'm putting the requirements a bit simple for Render to deploy the backend, but if you're planning to rewrite the code or something, you might want to downgrade Tensorflow to 2.13 and install all the module needed when running your code.

### 2. Run the Server

```bash
python app.py
```

It will start at `http://localhost:5000`.

---

## Switching Between Local and Hosted

In `chrome_extension/script.js`, find the fetch call:

```js
fetch("https://malicious-domain-checker.onrender.com/predict", {
```

To run locally, change it to:

```js
fetch("http://localhost:5000/predict", {
```

Save, then reload the extension in `chrome://extensions`.

---

## Model Info

- Trained on 13 statistical features extracted from domain names (e.g., length, number of vowels, MX records, TTL, etc.)
- Simple 1-hidden-layer neural network
- Pruned and quantized using TensorFlow Model Optimization Toolkit (On Python 3.10)
- Exported as `.tflite` for lightweight inference

---

## Notes

This project is for educational/demonstration purposes only, due to the fact that [the original repo](https://github.com/softwaresec-labs/MADONNA) only provided the model and the extension itself so i re-implemented the repo to provide the better approach to someone who wants to learn about this paper better.

---

## Credits

- Paper: [MADONNA: Browser-Based MAlicious Domain Detection Through Optimized Neural Network with Feature Analysis](https://doi.org/10.1007/978-3-031-56326-3_20)
- Re-implemented and extended by @Neit272, @tungnnx (support)

---

_For questions or contributions, please open an issue or pull request on the repository._
