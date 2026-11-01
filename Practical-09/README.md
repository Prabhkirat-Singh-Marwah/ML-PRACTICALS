# ML Lab 9: K-Means and Hierarchical Clustering

**Practical:** ML Lab 9
**Aim:** To implement K-Means Clustering and Hierarchical Clustering algorithms.

**Author:** Prabhkirat Singh Marwah
**Section:** B | **Batch:** B4 | **Roll No:** 47

---

## Dataset

**File:** `WHOLESALE.csv` (Wholesale Customers dataset)

- 440 customers, 8 columns, no missing values, no duplicates
- **Channel:** 1 = Horeca (hotels/restaurants/cafés), 2 = Retail
- **Region:** 1 = Lisbon, 2 = Oporto, 3 = Other
- **Spending columns:** Fresh, Milk, Grocery, Frozen, Detergents_Paper, Delicassen

---

## Libraries Used

`numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `scipy`

---

## Steps Followed

1. **Data Understanding:** Checked shape, data types, missing values and duplicates.
2. **Univariate Analysis:** Histograms with KDE, boxplots, and IQR-based outlier detection.
3. **Bivariate Analysis:** Spending compared across Channel and Region, plus scatter plots.
4. **Multivariate Analysis:** Correlation heatmap, pair plot, and spending groups (Low/Medium/High).
5. **Preprocessing:** Dropped Channel and Region, then scaled the 6 spending columns using `StandardScaler`.
6. **Finding K:** Used the Elbow Method and Silhouette Score.
7. **K-Means Clustering:** Applied with the best K.
8. **Hierarchical Clustering:** Built a dendrogram (Ward linkage) and applied Agglomerative Clustering.
9. **Visualization:** Used PCA to reduce to 2D and plot the clusters.
10. **Comparison and Profiling:** Compared both algorithms and studied the average spending of each cluster.

---

## Key Results

| Observation | Result |
|---|---|
| Most outliers | Frozen (43 outliers, about 9.8%) |
| Strongest correlation | Grocery and Detergents_Paper (0.92) |
| Best K (Silhouette) | 3 (score 0.5483) |
| PCA (2 components) | Explains 72.46% of variance |

### Algorithm Comparison

| Algorithm | Clusters | Silhouette Score | Cluster Sizes |
|---|---|---|---|
| K-Means | 3 | 0.5483 | 45 / 393 / 2 |
| Hierarchical (Ward) | 3 | 0.2646 | 6 / 281 / 153 |

### K-Means Cluster Profile

- **Cluster 0:** High spenders on Grocery, Milk and Detergents_Paper (retail-like customers).
- **Cluster 1:** Regular, lower-spending customers (the largest group).
- **Cluster 2:** Only 2 customers with extremely high Frozen and Delicassen spending (outliers).

---

## Conclusion

K-Means gave a higher silhouette score (0.5483) than Hierarchical Clustering (0.2646), so it formed better-separated clusters on this dataset. However, K-Means created one tiny cluster of 2 outlier customers, while Hierarchical Clustering gave more balanced groups. Handling outliers before clustering could improve both results.

---

## How to Run

1. Place `WHOLESALE.csv` in the same folder as the notebook.
2. Install the libraries: `pip install numpy pandas matplotlib seaborn scikit-learn scipy`
3. Open the notebook in Jupyter and run all cells in order.
