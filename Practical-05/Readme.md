# ML Lab 5: K-Nearest Neighbours (KNN) on the Diabetes Dataset

| | |
|---|---|
| **Name** | Prabhkirat Singh Marwah |
| **Class** | B |
| **Batch** | B4 |
| **Roll No.** | 47 |
| **Practical** | 5: KNN Classification |

---

## 1. Aim
To build a K-Nearest Neighbours (KNN) classifier that predicts whether a patient has diabetes, and to find the best value of K and the best hyperparameters using cross-validation and grid search.

## 2. Dataset
- **File:** `diabetes.csv` (Pima Indians Diabetes dataset)
- **Size:** 768 rows × 9 columns
- **Input features (8):** Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age
- **Target:** `Outcome` (0 = No Diabetes, 1 = Diabetes)
- **Missing values / duplicates:** none in the raw file, but Glucose, BloodPressure, SkinThickness, Insulin and BMI contain invalid `0` values.

## 3. Libraries Used
`numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`

## 4. Steps Followed

### 4.1 Exploratory Data Analysis (EDA)
- Viewed the data with `head()`, `tail()`, `shape`, `info()`, `describe()`
- Checked missing values and duplicates
- Plotted the Outcome distribution, an Age boxplot, a correlation heatmap and feature histograms

### 4.2 Data Cleaning
Invalid zeros found in these columns:

| Column | Zero count |
|---|---|
| Glucose | 5 |
| BloodPressure | 35 |
| SkinThickness | 227 |
| Insulin | 374 |
| BMI | 11 |

These zeros were converted to `NaN` and filled with the **median** of each column.

### 4.3 Train-Test Split
- 80% training (614 samples) and 20% testing (154 samples)
- `stratify=y` keeps the class ratio equal (about 65% No Diabetes and 35% Diabetes in both sets)
- `random_state=42` makes the split repeatable

### 4.4 Feature Scaling
`StandardScaler` was fitted on the training data only and then applied to the test data. KNN is distance-based, so all features must be on the same scale.

### 4.5 Model Training
`KNeighborsClassifier(n_neighbors=5)` was trained on the scaled training data.

### 4.6 Evaluation
Predictions, probabilities, accuracy, recall, F1 score and the confusion matrix were calculated on the test set.

### 4.7 Choosing the Best K
1. **Train vs test accuracy** for K = 1 to 30
2. **10-fold Stratified Cross-Validation** for K = 1 to 30
3. **GridSearchCV** (5-fold) over `n_neighbors`, `weights` and `metric`

## 5. Results

### Baseline model (K = 5)
| Metric | Score |
|---|---|
| Accuracy | 0.7532 |
| Recall | 0.6111 |
| F1 Score | 0.6346 |

**Confusion matrix**

| | Predicted No Diabetes | Predicted Diabetes |
|---|---|---|
| **Actual No Diabetes** | 83 | 17 |
| **Actual Diabetes** | 21 | 33 |

### Finding the best K
| Method | Best K | Accuracy |
|---|---|---|
| Test accuracy | 8 | 0.7662 |
| 10-fold Cross-Validation | 24 | 0.7770 |

### GridSearchCV
| Parameter | Best value |
|---|---|
| `n_neighbors` | 19 |
| `weights` | uniform |
| `metric` | euclidean |
| **Best CV accuracy** | **0.7769** |

## 6. Conclusion
- KNN with K = 5 gave about **75% accuracy** on the diabetes data.
- Tuning K with cross-validation and grid search improved accuracy to about **77.7%**, with the best setting at **K = 19, euclidean distance, uniform weights**.
- Recall (61%) is lower than accuracy, so the model misses some real diabetes patients. This matters in medical problems.
- Scaling and replacing the invalid zeros were important steps for good performance.

## 7. How to Run
1. Place `diabetes.csv` in the same folder as the notebook.
2. Install the requirements:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn
   ```
3. Open `ml_lab_5_knn_.ipynb` in Jupyter Notebook or Google Colab and run all cells from top to bottom.

## 8. Files
```
├── ml_lab_5_knn_.ipynb   # main notebook
├── diabetes.csv          # dataset
└── README.md             # this file
```
