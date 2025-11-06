# 🛰️ Satellite Land Classifier

A web-based AI-powered **satellite land classification** tool that analyzes selected regions on the map and classifies land into categories like **vegetation, water, sand, rocky, urban**, and more — powered by **OpenCV** and **Flask**.

🌍 **Draw a region → Click Classify → Get instant visual and statistical results!**

---

## 🚀 Live Demo

🔗 **[Launch the App on Render](https://your-app-name.onrender.com)**  
*(Replace this link after deployment)*

---

## ✨ Features

- 🗺️ Interactive map with **drawing tools** (powered by Leaflet)
- 🧠 Real-time classification using **OpenCV color analysis**
- 🌳 Detects vegetation, water bodies, sand, rocky, and urban regions
- 📊 Generates **graphs and percentages** for each land type
- 🎨 Visually appealing UI with blue-white theme and animations
- ⚡ Fast and lightweight — built for educational and demo use

---

## 🧩 Tech Stack

| Component | Technology |
|------------|-------------|
| Frontend | HTML, CSS, JavaScript (Leaflet.js, Chart.js) |
| Backend | Python (Flask) |
| Image Processing | OpenCV |
| Hosting | Render.com (Free Flask Hosting) |
| Source Control | Git + GitHub |

---

## 🖥️ Screenshots

| Crop & Classify | Classification Result |
|------------------|------------------------|
| ![Crop Area](https://via.placeholder.com/300x180?text=Crop+Region) | ![Results](https://via.placeholder.com/300x180?text=Results+Pie+Chart) |

*(Replace these images with your own screenshots once deployed)*

---

## ⚙️ Installation (Run Locally)

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/satellite-land-classifier.git

# Go into the project directory
cd satellite-land-classifier

# (Optional) Create a virtual environment
python -m venv venv
venv\Scripts\activate  # on Windows
# source venv/bin/activate  # on macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Run the Flask app
python app.py
