#**Breast Cancer Classification 🩺🎗️**

This project builds a machine learning model to classify whether a tumor is malignant or benign based on a dataset with multiple features. It includes data preprocessing, model training, evaluation, and deployment using Flask.

📌 **Problem Statement**
Breast cancer is a leading cause of death among women worldwide. Early detection using machine learning can help improve diagnosis accuracy. However, having 31 features in the dataset could lead to overfitting. To solve this, we performed feature selection and kept only 13 meaningful features.

🛠️ **Technologies & Libraries Used**

pandas – Data manipulation
numpy – Numerical operations
matplotlib & seaborn – Data visualization
scikit-learn – Machine learning models & evaluation
pickle – Model serialization for deployment
Flask – Web app development

📊 **Data Preprocessing & Feature Selection**

Feature Reduction: Dropped highly correlated columns (correlation > 0.8) to prevent overfitting.
Outlier Removal: Used Interquartile Range (IQR) to eliminate extreme values.
Standardization: Scaled features using StandardScaler from scikit-learn.
Data Splitting: Split the dataset into train (80%) and test (20%) sets.

📈 **Machine Learning Models Used**
After preprocessing, we trained multiple classification models:

Support Vector Machine (SVM)
Logistic Regression
Decision Tree Classifier
K-Nearest Neighbors (KNN)
Bernoulli Naive Bayes

Each model was evaluated using:
Accuracy Score
Precision, Recall, F1-score (classification_report from sklearn.metrics)

📂 **Project Structure**
/Breast-Cancer-Classification
│── templates/              # Frontend files
│   ├── index.html          # Web app interface
│── Machine_learning_project.ipynb  # Jupyter Notebook for ML model training
│── app.py                  # Flask web app for deployment
│── finalized_model.pkl      # Serialized trained model
│── scaler.pkl               # Scaler used for preprocessing
│── README.md               # Project documentation

🚀**Deployment with Flask & Pickle**
The trained model was saved using Pickle and integrated into a Flask web app for enhanced user experience. The web app allows users to input tumor characteristics and get predictions instantly.

📢 **How to Run the Project**

1️⃣ Clone the Repository
git clone https://github.com/youseflasheen/Breast-Cancer-Classification.git
cd Breast-Cancer-Classification

2️⃣ **Install Dependencies**

pip install -r requirements.txt

3️⃣ **Run the Flask App**

python app.py
The web app will be available at http://127.0.0.1:5000/.

📄 **License**
This project is open-source and available for use.

