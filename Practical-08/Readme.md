# Practical 8: Support Vector Machine (SVM) for Classification

**Aim:** To implement the Support Vector Machine (SVM) algorithm for classification.

| | |
|---|---|
| **Author** | Prabhkirat Singh Marwah |
| **Section** | B |
| **Batch** | B4 |
| **Roll No.** | 47 |

---

## About the Practical

In this practical, an SVM model is used to predict whether a machine will fail or not (`Machine failure` column) using sensor readings like temperature, rotational speed, torque and tool wear. Four SVM kernels are compared, and then hyperparameter tuning is done to find better settings.

## Files

- `ml_lab_8.ipynb`: Jupyter notebook with the full code
- `ml_8.csv`: dataset used (10,000 rows)

## Libraries Used

- pandas, numpy
- matplotlib, seaborn
- scikit-learn

Install them with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Steps Followed

1. Loaded the dataset and checked it with `head()`, `describe()` and `info()`.
2. Checked for null values and duplicate rows.
3. Selected input features (`X`) and the target (`y` = Machine failure).
4. Split the data into 80% training and 20% testing (with `stratify=y`).
5. Scaled the numeric columns with `StandardScaler` and encoded `Type` with `OneHotEncoder`.
6. Trained SVM models with 4 kernels: Linear, RBF, Polynomial and Sigmoid.
7. Evaluated each model using accuracy, precision, recall, F1 score and a confusion matrix.
8. Tuned the hyperparameters (`kernel`, `C`, `gamma`) using `RandomizedSearchCV`.
9. Compared all models in a final table and bar graph.

## Results (Test Set)

| Kernel | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| Linear | 0.966 | 0.00 | 0.00 | 0.00 |
| RBF | ~0.97 | 0.88 | 0.21 | 0.33 |
| Polynomial (degree 3) | 0.9745 | 0.87 | 0.29 | 0.44 |
| Sigmoid | 0.945 | 0.15 | 0.13 | 0.14 |

The results of the tuned model are shown in the notebook.

## Observations

- Accuracy is high for all models, but the data is imbalanced (very few failure cases), so accuracy alone is not a good measure.
- The Linear SVM predicted "No Failure" for every row, so its precision, recall and F1 are 0.
- The Polynomial kernel gave the best F1 score among the four basic kernels.
- Hyperparameter tuning (scoring on F1) is used to improve the failure detection.

## Conclusion

SVM can be used for classification, and the choice of kernel changes the result a lot. For an imbalanced dataset like this one, precision, recall and F1 score should be checked along with accuracy.

## How to Run

1. Put `ml_8.csv` in the same folder as the notebook.
2. Open `ml_lab_8.ipynb` in Jupyter Notebook or VS Code.
3. Click **Kernel → Restart & Run All**.
