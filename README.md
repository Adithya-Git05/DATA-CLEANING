DIABETES PREDICTION USING REGRESSION MODELS
This project aims to predict the likelihood of diabetes in patients using the Pima Indians Diabetes dataset. We utilize data preprocessing techniques and implement both Linear Regression and Polynomial Regression models to evaluate prediction performance.

DATASET
Source: Pima Indians Diabetes Database - Kaggle

ATTRIBUTES:

Pregnancies
Glucose
BloodPressure
SkinThickness
Insulin
BMI
DiabetesPedigreeFunction
Age

Outcome (0 = No Diabetes, 1 = Diabetes)

OBJECTIVES:

Clean and preprocess the dataset.
Visualize data distributions and correlations.
Build and evaluate regression models.
Predict diabetes based on user input.

DATA PREPROCESSING:
Missing Values: Values such as 0 in columns like Glucose, BloodPressure, SkinThickness, Insulin, and BMI were replaced with NaN and imputed using the median.

Normalization: Applied Min-Max Scaling to normalize all numerical features to a [0, 1] range.

Exploratory Data Analysis
Used histograms to explore feature distributions.

Created a correlation heatmap to identify relationships between features and the target (Outcome).

MODELS USED:
1. Linear Regression
Used LinearRegression() from sklearn.

R² Score: ~0.255

Mean Squared Error: ~0.171

2. Polynomial Regression
Built using PolynomialFeatures with degree = 2.

R² Score: ~0.221

Mean Squared Error: ~0.178

Both models showed limited prediction ability, highlighting that regression is not ideal for binary classification tasks like this. However, this exercise helps understand model fitting and evaluation.

Prediction Example
A sample prediction for a new data point is performed using both models after preprocessing and scaling.

python
Copy
Edit
single_input = np.array([[6, 148, 72, 35, 0, 33.6, 0.627, 50]])
# Apply the same preprocessing (e.g., scaling)
# Predict using trained models

TECHNOLOGIES AND LIBRARIES:

Python 3
Pandas, NumPy
Matplotlib, Seaborn
Scikit-learn

RESULTS AND LIMITATIONS:

Regression models give continuous outputs, which are not optimal for a classification problem like diabetes prediction.
Classification models such as Logistic Regression, Random Forest, or SVM would be more suitable.

FUTURE IMPROVEMENTS:

Switch to classification models.
Use cross-validation for more robust evaluation.
Hyperparameter tuning using GridSearchCV.
Deploy model with a user interface for real-time predictions.
