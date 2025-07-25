# Predicting Customer Spending Behaviour in Streaming Services
🎬 Predicting Customer Spending Behaviour in Streaming Services:
This project explores supervised and unsupervised machine learning techniques to analyze and predict customer behavior within a streaming service context. The dataset includes customer demographic and usage information, enabling the development of regression, classification, and clustering models.

📂 Dataset:
Source: Kaggle

Link: [Streaming Service Data](https://www.kaggle.com/datasets/akashanandt/streaming-service-data)

🧠 Project Goals:
The project investigates:

Customer monthly spending prediction

Churn prediction (classification)

Customer segmentation (clustering)

✅ Tasks & Methods:
🔢 (a) Single Feature Regression:
Compared multiple regression models (linear & polynomial) to predict Monthly Spend from individual numeric features.

Identified the strongest predictor and best-fitting model type.

📊 (b) Multivariate Regression:
Used multiple numerical features to improve prediction performance.

Compared results to single-feature models.

🧬 (c) Categorical + Numerical Features:
Trained ensemble models (e.g., Random Forest Regressor) using both feature types.

Evaluated performance gain from including categorical variables.

🤖 (d) Artificial Neural Network (ANN):
Designed and tuned an ANN to predict Monthly Spend.

Compared ANN performance with other supervised models.

🏆 (e) Model Comparison:
Evaluated all models using metrics like RMSE, R², and MAE.

Recommended the best-performing regression model.

🔍 (f) Customer Churn Prediction (Classification):
Built models to predict whether a customer would churn.

Compared models using:

Accuracy

Precision, Recall, F1-Score

AUC-ROC

🧩 (g) k-Means Clustering:
Applied k-Means to identify customer groups.

Determined optimal k using silhouette score and elbow method.

Visualized clusters to analyze meaningful patterns.

🌐 (h) Clustering Comparison:
Compared k-Means with other clustering algorithms (e.g., DBSCAN, Hierarchical).

Evaluated segmentation quality with silhouette score and Davies–Bouldin index.

🛠️ Tools & Libraries:
Python

Pandas, NumPy

Scikit-learn

Matplotlib, Seaborn

TensorFlow / Keras

XGBoost / Random Forest

Yellowbrick (for visualizations)

📌 Results Summary:
The best regression model combined numerical and categorical features using a Random Forest or ANN.

For classification, XGBoost or Random Forest Classifier showed the best churn prediction performance.

For clustering, k-Means with ~3–4 clusters provided interpretable segmentation; however, DBSCAN performed better in capturing irregular cluster shapes.

📈 Visuals & Insights:
Model performances, confusion matrices, cluster visualizations, and feature importance plots are all included in the notebook for clarity and interpretability.

📚 Author:
Adeniran Adewumi – MSc AI for Engineering Student | Passionate about predictive analytics, customer behavior modeling, and AI for real-world applications.