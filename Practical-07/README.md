# ML Lab 7: Naive Bayes Classification

Classifying raisin types using the **Gaussian Naive Bayes** algorithm in Python (scikit-learn).

---

## Author

| Field   | Details               |
|---------|-----------------------|
| Name    | Prabhkirat Singh Marwah |
| Section | B                     |
| Batch   | B4                    |
| Roll No | 47                    |

---

## Objective

To build a Naive Bayes classifier that predicts the class of a raisin from its physical measurements, and to evaluate the model using standard classification metrics.

---

## Dataset

- **File:** `Raisin_Dataset.csv`
- **Target column:** `Class` (the raisin type)
- **Features:** numerical measurements of each raisin, such as area, axis lengths, eccentricity, convex area, extent and perimeter.

---

## Technologies Used

- Python 3
- Pandas and NumPy (data handling)
- Matplotlib and Seaborn (visualization)
- Scikit-learn (model building and evaluation)
- Jupyter Notebook

---

## Project Workflow

1. **Import libraries** – load all required Python packages.
2. **Load the dataset** – read the CSV file and preview it with `head()` and `tail()`.
3. **Encode labels** – convert class names to numbers (0 and 1) using `LabelEncoder`.
4. **Select features** – pick the numerical columns as input (`X`) and the encoded class as target (`y`).
5. **Split the data** – 80% training and 20% testing, using `stratify=y` and `random_state=42`.
6. **Train the model** – fit a `GaussianNB` classifier on the training data.
7. **Predict** – generate predictions for the test set.
8. **Compare results** – show actual vs predicted labels, converted back to class names.
9. **Evaluate** – calculate Accuracy, Precision, Recall and F1 Score.
10. **Report and visualize** – print the classification report and confusion matrix, and plot the confusion matrix as a heatmap.
11. **Check fit** – compare training accuracy with testing accuracy.

---

## Evaluation Metrics

| Metric    | Meaning                                                    |
|-----------|------------------------------------------------------------|
| Accuracy  | Fraction of all predictions that were correct              |
| Precision | When the model predicts a class, how often it is right     |
| Recall    | Out of all real members of a class, how many were found    |
| F1 Score  | Balanced score combining precision and recall              |

Precision, Recall and F1 are calculated with `average='weighted'`.

---

## Results

| Metric            | Value |
|-------------------|-------|
| Accuracy          | _fill from notebook output_ |
| Precision         | _fill from notebook output_ |
| Recall            | _fill from notebook output_ |
| F1 Score          | _fill from notebook output_ |
| Training Accuracy | _fill from notebook output_ |
| Testing Accuracy  | _fill from notebook output_ |

---

## How to Run

1. Clone or download this repository.
2. Install the required libraries:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn jupyter
   ```
3. Place `Raisin_Dataset.csv` in the same folder as the notebook.
4. Start Jupyter and open the notebook:
   ```bash
   jupyter notebook ML_LAB_7.ipynb
   ```
5. Run all cells from top to bottom.

---

## Project Structure

```
.
├── ML_LAB_7.ipynb        # Main notebook
├── Raisin_Dataset.csv    # Dataset
└── README.md             # Project documentation
```

---

## Conclusion

The Gaussian Naive Bayes model was trained and tested on the raisin dataset. Its performance was measured with accuracy, precision, recall, F1 score and a confusion matrix, showing how well the model separates the two raisin classes.

---

*Submitted by Prabhkirat Singh Marwah, Section B, Batch B4, Roll No 47.*
