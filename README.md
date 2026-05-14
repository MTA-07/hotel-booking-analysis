🏨 Hotel Booking Cancellation Prediction with Deep Learning
📌 Project Overview

This project analyzes hotel booking data and builds a deep learning model to predict whether a customer will cancel their reservation. The goal is to help hotels understand customer behavior, reduce cancellation risks, and improve revenue management strategies.

The pipeline covers the full data science workflow:

Data preprocessing
Feature engineering
Exploratory Data Analysis (EDA)
Deep learning model training
Prediction system (inference)
🎯 Objective
Analyze hotel booking patterns
Identify factors affecting cancellations
Build a machine learning model to predict cancellation probability
Provide actionable insights for business decision-making
🧠 Technologies Used
Python
Pandas & NumPy (data manipulation)
Matplotlib (visualization)
Scikit-learn (preprocessing & splitting)
TensorFlow / Keras (deep learning model)
Git & GitHub (version control)
📊 Dataset

The dataset contains hotel booking records including:

Booking dates
Customer type
Number of guests
Stay duration
Previous cancellations
Special requests
Pricing information (ADR)
Cancellation status (target variable)
⚙️ Data Preprocessing
Missing values handled:
children → filled with 0
country → filled with "Unknown"
Unnecessary columns removed (agent, company)
New features created:
total_nights
total_guests
is_family
stay_type
📈 Exploratory Data Analysis (EDA)

The following insights were generated:

Monthly booking distribution
Hotel cancellation rates
Top 10 countries by bookings
Customer behavior patterns

All visualizations are saved in the charts/ directory.

🧩 Feature Engineering

New meaningful features were created to improve model performance:

total_nights = weekend + week nights
total_guests = adults + children + babies
is_family = family booking indicator
stay_type = short or long stay classification

These features help the model better understand customer behavior.

🧠 Deep Learning Model

A neural network was built using TensorFlow/Keras:

Model Architecture:
Input Layer (18 features)
Dense(64, ReLU)
Dense(32, ReLU)
Dense(16, ReLU)
Output Layer (Sigmoid)
Compilation:
Loss Function: Binary Crossentropy
Optimizer: Adam
Metric: Accuracy
📊 Model Performance
Training and validation accuracy were monitored over epochs
Loss curves were analyzed to check overfitting
Final evaluation was done on test dataset

Training history visualization is available in:

charts/training_history.png
🔮 Prediction System

The model can predict cancellation probability for a new customer.

Example output:

Input: Customer booking details
Output: Probability of cancellation (%)
Decision: Risky or Safe booking
📁 Project Structure
GHO/
│
├── data/
│   └── hotel_bookings.csv
│
├── charts/
│   ├── monthly_bookings.png
│   ├── hotel_cancel_rate.png
│   └── training_history.png
│
├── outputs/
│   ├── missing_report.csv
│   ├── hotel_summary.csv
│   └── customer_report.csv
│
├── main.py
├── requirements.txt
└── README.md
🚀 Future Improvements
Hyperparameter tuning
Handling class imbalance (SMOTE, class weights)
Adding ROC-AUC evaluation
Building a web app with Streamlit
Deployment as API (FastAPI / Flask)
📌 Conclusion

This project demonstrates an end-to-end machine learning pipeline applied to real-world hotel booking data. It combines data analysis, feature engineering, and deep learning to create a functional predictive system that can assist hotel revenue management.

📚 References
https://www.kaggle.com/datasets/mojtaba142/hotel-booking
https://www.tensorflow.org/
https://scikit-learn.org/
https://pandas.pydata.org/
