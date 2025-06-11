# **Wine Quality Prediction – Model Evaluation Report**  

## **Overview**  
This project evaluates various machine learning algorithms applied to the **Wine Quality Dataset**, which includes both red and white wine samples. The goal is to predict wine quality (on a scale of 0-10) based on chemical properties such as alcohol content, acidity, pH, and sulfur dioxide levels.  

### **Challenges in Wine Quality Prediction**  
- **Imbalanced Class Distribution**: Quality ratings are not evenly distributed.  
- **Non-linear Relationships**: Chemical properties do not have a straightforward correlation with quality.  
- **Feature Overlap**: Different quality levels may have similar chemical compositions.  

---

## **Algorithms Evaluated**  
We tested multiple machine learning models to determine the best performer:  

| **Algorithm**          | **Accuracy** | **Strengths** | **Weaknesses** |
|------------------------|-------------|---------------|----------------|
| Linear Regression      | 53.85%      | Simple, interpretable | Poor for classification |
| Logistic Regression    | 54.15%      | Good for binary classification | Struggles with multi-class |
| Decision Tree          | 59.10%      | Handles non-linear data | Prone to overfitting |
| K-Nearest Neighbors (KNN) | 54.85%  | Simple, intuitive | Sensitive to noise |
| Support Vector Machine (SVM) | 56.00% | Effective in high dimensions | Requires tuning |
| Naive Bayes            | 34.08%      | Fast, efficient | Poor with correlated features |
| **Random Forest**      | **66.69%**  | Robust, handles non-linearity | Computationally heavy |

---

## **Key Findings**  
### **Best Performing Model: Random Forest (66.69% Accuracy)**  
- **Why it works well**:  
  - Handles non-linear relationships effectively.  
  - Reduces overfitting by averaging multiple decision trees.  
  - Provides feature importance rankings.  

### **Worst Performing Model: Naive Bayes (34.08% Accuracy)**  
- **Why it fails**:  
  - Assumes feature independence (unrealistic for wine data).  
  - Struggles with continuous numerical features.  

### **Feature Importance (Random Forest)**  
<img width="471" alt="{75700A7B-DCFE-4B8B-9DEC-9094118CBF0A}" src="https://github.com/user-attachments/assets/09e6b0dc-9c3e-4d94-9387-b40332fd72cb" />
- **Top predictors of wine quality**:  
  1. **Alcohol content**  
  2. **Volatile acidity**  
  3. **Sulphates**  

---

## **Visualizations**  
### **1. Confusion Matrix (Random Forest)**  
<img width="316" alt="{6F50C964-90F7-442D-A79D-BE4A45A0330A}" src="https://github.com/user-attachments/assets/22fe9c29-c6ae-4ed3-b838-bbe8bff3b141" /> 
- Shows correct vs. incorrect predictions across quality classes.  

### **2. Decision Boundary (SVM vs. KNN)**  
<img width="292" alt="{2A3945AE-FEFF-494B-86C2-A45B30AA6D8A}" src="https://github.com/user-attachments/assets/f7bcdd9c-bab8-476f-8401-ec6ffba7ab12" />
- SVM tries to maximize the margin between classes.  
- KNN classifies based on nearest neighbors.  

### **3. Actual vs. Predicted Quality (Linear Regression)**  
<img width="354" alt="{027BC77C-5114-4CB5-AEEB-6D163AC39651}" src="https://github.com/user-attachments/assets/be881f76-5610-4b8f-9882-27968ffd9d8d" />
- Linear Regression struggles with discrete classification.  

---

## **Conclusion & Recommendations**  
✅ **Best Model**: **Random Forest** (Highest accuracy, robust to noise).  
⚠ **Avoid**: Naive Bayes (Poor accuracy due to unrealistic assumptions).  
📊 **Next Steps**:  
- Hyperparameter tuning (GridSearchCV for better performance).  
- Feature engineering (e.g., binning quality into "Low/Medium/High").  
- Try **Gradient Boosting (XGBoost, LightGBM)** for potential improvement.  

---

## **Code Implementation**  
### **Data Preprocessing**  
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load data
red_wine = pd.read_csv("winequality-red.csv", sep=';')
white_wine = pd.read_csv("winequality-white.csv", sep=';')

# Combine datasets
wine_df = pd.concat([red_wine, white_wine], ignore_index=True)

# Train-test split
X = wine_df.drop('quality', axis=1)
y = wine_df['quality']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### **Training Random Forest**  
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)
y_pred = rf.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Random Forest Accuracy: {accuracy * 100:.2f}%")
```

---

## **Author**  
👨‍💻 **Sarvesh Jayant Patil**  
📧 **sarrveshpatil@gmail.com**  
🔗 **[GitHub](https://github.com/sarrveshpatil)** | **[LinkedIn](https://linkedin.com/in/sarrveshpatil)**  

---
**License**: MIT  
**Dataset Source**: [UCI Wine Quality Dataset](https://archive.ics.uci.edu/ml/datasets/wine+quality)  

---

### **🔍 Want to Explore Further?**  
- Check out the full **[Jupyter Notebook](https://github.com/sarrveshpatil/wine-quality-prediction)** for detailed analysis.  
- Try deploying the model using **Flask/Streamlit** for a live demo!  
