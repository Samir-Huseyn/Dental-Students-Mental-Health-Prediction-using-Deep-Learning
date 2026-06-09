# Dental-Students-Mental-Health-Prediction-using-Deep-Learning

This repository features two distinct Deep Learning projects implemented using **TensorFlow/Keras** with a focus on data preprocessing, feature engineering, and optimizing Artificial Neural Networks (ANN) for regression tasks.

---

## 📊 Project 1: Dental Students' Mental Health Analysis (`Dent.csv`)
This project predicts the depression levels (**CES-D score**) of dental students based on their demographic, lifestyle, and psychological metrics.

### ⚙️ Architecture & Implementation
- **Data Preprocessing:** Cleaned and structured data by removing irrelevant identifiers (`id`), checked for latent missing values, and performed standardization via `StandardScaler`.
- **Input Dimensions:** 18 distinct features (Age, Sex, Study Hours, Health Status, etc.).
- **Neural Network Architecture:**
  - Input Layer: 18 units, ReLU activation.
  - Hidden Layer 1: 9 units, ReLU activation.
  - Hidden Layer 2: 3 units, ReLU activation.
  - Output Layer: 1 unit with Linear activation (Optimized for continuous regression).
- **Optimization:** Compiled using the `Adam` optimizer and `MSE` loss function.

### 📈 Results
- **Mean Absolute Error (MAE):** **`6.26`**
- *Interpretation:* The model predicts a student's depression score with an average margin of error of just 6.26 points on the CES-D scale, capturing strong psychological correlations.

---

## 🏡 Project 2: Boston Housing Price Prediction (`data.csv`)
An advanced regression model built on the classic Boston Housing dataset to estimate median house values based on socioeconomic and environmental factors.

### ⚙️ Architecture & Implementation
- **Data Preprocessing:** Handled missing values dynamically by filling the critical `RM` (Average Number of Rooms) column with its median. Applied strict `StandardScaler` transformations on train and test folds to prevent data leakage.
- **Neural Network Architecture:**
  - Input Layer: 13 units, ReLU activation.
  - Hidden Layer 1: 24 units, ReLU activation.
  - Hidden Layer 2: 12 units, ReLU activation.
  - Hidden Layer 3: 6 units, ReLU activation.
  - Output Layer: 1 unit with Linear activation.
- **Optimization:** Trained over 25 epochs using `Adam` and `MSE`.

### 📈 Results
- **Mean Absolute Error (MAE):** **`2.96`** ($2,960)
- *Interpretation:* The model achieves a stellar performance, miscalculating the actual median house value by only ~$2,960 on average, proving the effectiveness of the multi-layered deep architecture.

---

## 🛠️ Tech Stack & Libraries
- **Language:** Python 3.13
- **Deep Learning Framework:** TensorFlow 2.x / Keras
- **Data Manipulation:** Pandas, NumPy
- **Machine Learning Utilities:** Scikit-Learn (`train_test_split`, `StandardScaler`)
