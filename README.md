📌 Project Overview

This project predicts the next day's internet traffic usage using a Long Short-Term Memory (LSTM) neural network. The model is trained on synthetic internet usage data and deployed using aStreamlit web application.
 Installation & Setup

1️⃣ Clone the Repository

git clone [https://github.com/your-username/internet-traffic-prediction.git](https://github.com/mars2812/Internet-Traffic-Prediction/tree/master)
cd internet-traffic-prediction

2️⃣ Create & Activate Virtual Environment

python -m venv myenv  # Create virtual environment
source myenv/bin/activate  # Mac/Linux
myenv\Scripts\activate  # Windows

3️⃣ Install Dependencies

pip install -r requirements.txt

4️⃣ Run streamlit  App

streamlit  app.py

🌟 Features


✔ LSTM Model for Time-Series Prediction✔ Flask Web Interface for Predictions✔ Real-Time Data Preprocessing✔ Fully Responsive UI (HTML, CSS, JavaScript)✔ Scalable & Easily Deployable

📊 How It Works


User clicks the 'Predict Next Day Usage' button.

APP loads the trained LSTM model.

Preprocessed data is passed to the model for prediction.

The predicted usage is displayed on the result page.

📌 Technologies Used


Python (Flask, NumPy, Pandas, TensorFlow/Keras, Scikit-learn)

HTML, CSS, JavaScript (Frontend UI)

SQLite / CSV (Data Storage)

Jupyter Notebook / Google Colab (Model Training & EDA)

🛠 Troubleshooting

Issue: Model not loading?🔹 Run python in terminal and check if TensorFlow is installed:

import tensorflow as tf
print(tf.__version__)

Issue: Prediction not showing?
🔹 Restart Flask after making changes:

CTRL+C  # Stop the server
python app.py  # Restart Flask

🤝 Contributing

Feel free to fork this repository, create a new branch, and submit a pull request with improvements! 🚀

📜 License

This project is licensed under the MIT License.
