# Crop-Recommendation-System
Machine Learning based Crop Recommendation System using Flask. Predicts the best crop based on soil nutrients and environmental conditions, with visualization plots and user-friendly web interface.


Here is a **professional GitHub description** for your Crop Recommendation System project, including features and step-by-step instructions to run it in VS Code. 

---

# 🌾 Crop Recommendation System using Machine Learning and Flask

## 📌 Project Description

The Crop Recommendation System is a web-based application that uses Machine Learning to recommend the most suitable crop based on soil nutrients and environmental conditions. The system takes input values such as Nitrogen (N), Phosphorus (P), Potassium (K), temperature, humidity, pH, and rainfall, and predicts the best crop to grow.

The application is built using:

* Python
* Flask (Web Framework)
* Scikit-learn (Machine Learning)
* Pandas and NumPy (Data Processing)
* Matplotlib and Seaborn (Data Visualization)
* HTML and CSS (Frontend)

The system also generates crop-wise visualization plots such as scatter plot, bar chart, and box plot.

---

## 🚀 Features

* Predicts the most suitable crop based on soil and weather conditions
* User-friendly web interface
* Machine Learning model trained using Random Forest Classifier
* Displays crop-specific visualization plots
* Clean and responsive frontend design
* Easy to run locally using VS Code

---

## 📂 Project Structure

```
farmer_project/
│
├── app.py
├── main.py
├── crop_model.pkl
├── Crop_recommendation.csv
│
├── static/
│   └── plots/
│
├── templates/
│   └── index.html
│
└── README.md
```

---

## ⚙️ Requirements

Install the following Python libraries:

```
pip install flask pandas numpy scikit-learn matplotlib seaborn joblib
```

---

## ▶️ Steps to Run the Project in VS Code

### Step 1: Open Project Folder in VS Code

1. Open VS Code
2. Click **File → Open Folder**
3. Select your project folder (`farmer_project`)

---

### Step 2: Open Terminal in VS Code

Click:

```
Terminal → New Terminal
```

---

### Step 3: Train the Model and Generate Plots

Run:

```
python main.py
```

You should see:

```
Accuracy on test set: XX%
Model saved to crop_model.pkl
Crop-wise plots saved to static/plots/
```

---

### Step 4: Run the Flask Application

Run:

```
python app.py
```

You will see:

```
Running on http://127.0.0.1:5000
```

---

### Step 5: Open in Browser

Open your browser and go to:

```
http://127.0.0.1:5000
```

---

## 🧪 Example Input Values

| N  | P  | K  | Temperature | Humidity | pH  | Rainfall |
| -- | -- | -- | ----------- | -------- | --- | -------- |
| 90 | 42 | 43 | 20.8        | 82       | 6.5 | 202      |

Output:

```
Recommended Crop: Rice
```

---

## 📊 Machine Learning Model

Algorithm used:

```
Random Forest Classifier
```

Accuracy achieved:

```
~99%
```

---

## 💡 Future Improvements

* Deploy to cloud (Render / Heroku / AWS)
* Add fertilizer recommendation
* Add crop yield prediction
* Add database support

---

## 👨‍💻 Author

Rangith Kumar
Python Developer | Machine Learning Enthusiast



