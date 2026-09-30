# ID3 Decision Tree -- Customer Segmentation

## 1. Project Overview

This notebook demonstrates **customer segmentation using an ID3-style
Decision Tree** on the Online Retail dataset.

The workflow is:

**Raw Online Retail data → Data cleaning → Customer-level features →
Customer segments → Train/Test split → Entropy-based Decision Tree →
Evaluation → Tree visualization + Feature importance**

> **Important implementation note:** The notebook uses
> `sklearn.tree.DecisionTreeClassifier(criterion="entropy")`. This is an
> entropy-based decision tree and follows the same Information Gain idea
> taught for ID3. It is not a hand-coded ID3 implementation.

------------------------------------------------------------------------

## 2. Objective

The notebook predicts a customer's segment:

-   **Low Value**
-   **Medium Value**
-   **High Value**

The segments are created from `TotalSpend` using three quantiles.

The model then learns customer behavior patterns from:

-   `TotalQuantity`
-   `NumberOfTransactions`
-   `NumberOfProducts`
-   `AverageOrderValue`

------------------------------------------------------------------------

## 3. Libraries Used

``` python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier, plot_tree
```

### Why these libraries?

-   **Pandas** -- data loading, cleaning and grouping
-   **NumPy** -- numerical operations
-   **Matplotlib** -- visualization
-   **Scikit-learn** -- train/test split, decision tree and evaluation
    metrics

------------------------------------------------------------------------

# 4. Dataset

The notebook reads:

``` python
df = pd.read_excel("Online Retail.xlsx")
```

The dataset contains online retail transaction information.

The notebook first displays:

``` python
df.head()
df.info()
```

This helps inspect the first records and the structure/data types.

------------------------------------------------------------------------

# 5. Data Cleaning

## 5.1 Remove missing Customer IDs

``` python
df = df.dropna(subset=["CustomerID"])
```

Customer-level segmentation requires a valid `CustomerID`, so rows
without one are removed.

------------------------------------------------------------------------

## 5.2 Remove cancelled invoices

``` python
df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
```

Invoices beginning with `C` are treated as cancelled invoices and
removed.

------------------------------------------------------------------------

## 5.3 Remove invalid quantity and price values

``` python
df = df[df["Quantity"] > 0]
df = df[df["UnitPrice"] > 0]
```

Only positive quantities and positive unit prices are retained.

------------------------------------------------------------------------

# 6. Feature Engineering

## Total Amount

The notebook creates:

``` python
df["TotalAmount"] = df["Quantity"] * df["UnitPrice"]
```

So:

**TotalAmount = Quantity × UnitPrice**

This represents the value of an individual transaction line.

------------------------------------------------------------------------

# 7. Creating Customer-Level Data

The transaction data is converted into customer-level information:

``` python
customer_data = df.groupby("CustomerID").agg(
    TotalSpend=("TotalAmount", "sum"),
    TotalQuantity=("Quantity", "sum"),
    NumberOfTransactions=("InvoiceNo", "nunique"),
    NumberOfProducts=("StockCode", "nunique"),
    AverageOrderValue=("TotalAmount", "mean"),
    Country=("Country", "first")
).reset_index()
```

Each customer gets the following information:

  Feature                  Meaning
  ------------------------ ---------------------------------------
  `CustomerID`             Unique customer
  `TotalSpend`             Total amount spent
  `TotalQuantity`          Total quantity purchased
  `NumberOfTransactions`   Number of distinct invoices
  `NumberOfProducts`       Number of distinct products purchased
  `AverageOrderValue`      Average transaction value
  `Country`                Customer's country

------------------------------------------------------------------------

# 8. Creating the Target Variable

The notebook creates three customer segments using:

``` python
customer_data["Segment"] = pd.qcut(
    customer_data["TotalSpend"],
    q=3,
    labels=["Low Value", "Medium Value", "High Value"]
)
```

### What is `qcut`?

`qcut` divides the values into **quantile-based groups**.

Here:

``` text
q = 3
```

so customers are divided into three groups:

``` text
Low Spend     → Low Value
Middle Spend  → Medium Value
High Spend    → High Value
```

The target variable is:

``` python
y = customer_data["Segment"]
```

------------------------------------------------------------------------

# 9. Features Used by the Model

The notebook uses:

``` python
features = [
    "TotalQuantity",
    "NumberOfTransactions",
    "NumberOfProducts",
    "AverageOrderValue"
]
```

Then:

``` python
X = customer_data[features]
y = customer_data["Segment"]
```

### Important Viva Question

### Why is `TotalSpend` not used as an input feature?

Because the target `Segment` is **created directly from `TotalSpend`**.

If `TotalSpend` were also given to the model as an input, it would
provide the model with the same information used to create the target,
causing **target leakage**.

------------------------------------------------------------------------

# 10. Train-Test Split

``` python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
```

### Meaning

-   **80%** → training data
-   **20%** → testing data
-   `random_state=42` → reproducible split
-   `stratify=y` → preserves the class distribution in train and test
    sets

------------------------------------------------------------------------

# 11. ID3 and Entropy

## What is ID3?

**ID3 = Iterative Dichotomiser 3**

ID3 is a supervised decision-tree algorithm for classification.

It chooses the attribute that gives the **highest Information Gain**.

The lecture describes ID3 as selecting the attribute that produces the
greatest reduction in uncertainty/entropy.

------------------------------------------------------------------------

# 12. What is Entropy?

Entropy measures the **impurity or uncertainty** of a dataset.

The formula is:

**Entropy(S) = - Σ pᵢ log₂(pᵢ)**

where `pᵢ` is the probability of class `i`.

### Interpretation

``` text
Entropy = 0
    ↓
Pure dataset
All samples belong to one class

Higher entropy
    ↓
More mixed classes
```

For a binary dataset containing equal numbers of positive and negative
samples:

**Entropy = 1**

The lecture's example with 4 Yes and 4 No gives entropy 1.0.

------------------------------------------------------------------------

# 13. What is Information Gain?

Information Gain tells us how much uncertainty is reduced after
splitting on an attribute.

**Information Gain = Parent Entropy − Weighted Child Entropy**

The ID3 algorithm chooses the feature with the **highest Information
Gain**.

Example from the lecture:

  Attribute     Information Gain
  ----------- ------------------
  Weather                  0.656
  Humidity                 0.049
  Wind                     0.049

Therefore:

**Weather becomes the root node.**

------------------------------------------------------------------------

# 14. Decision Tree in This Notebook

The model is created using:

``` python
id3_model = DecisionTreeClassifier(
    criterion="entropy",
    random_state=42,
    max_depth=4
)
```

### `criterion="entropy"`

This tells scikit-learn to use **entropy-based splitting**.

This matches the ID3 concept of selecting splits using information gain.

### `max_depth=4`

The tree is limited to a maximum depth of 4.

This helps prevent the tree from becoming unnecessarily large.

------------------------------------------------------------------------

# 15. Training the Model

``` python
id3_model.fit(X_train, y_train)
```

The model learns the relationship between customer behavior features and
the customer segment.

------------------------------------------------------------------------

# 16. Prediction

``` python
y_pred = id3_model.predict(X_test)
```

The trained tree predicts the segment of each test customer.

Possible predictions:

``` text
Low Value
Medium Value
High Value
```

------------------------------------------------------------------------

# 17. Model Evaluation

## Accuracy

``` python
accuracy = accuracy_score(y_test, y_pred)
```

Accuracy measures the proportion of test samples classified correctly.

**Accuracy = Correct Predictions / Total Predictions**

------------------------------------------------------------------------

## Classification Report

``` python
classification_report(y_test, y_pred)
```

This provides metrics such as:

-   Precision
-   Recall
-   F1-score
-   Support

------------------------------------------------------------------------

## Confusion Matrix

``` python
confusion_matrix(y_test, y_pred)
```

The confusion matrix shows how actual classes compare with predicted
classes.

It helps identify which customer segments are being confused with one
another.

------------------------------------------------------------------------

# 18. Feature Importance

The notebook obtains:

``` python
id3_model.feature_importances_
```

and creates a DataFrame:

``` python
importance = pd.DataFrame({
    "Feature": features,
    "Importance": id3_model.feature_importances_
})
```

The values indicate the relative contribution of each feature to the
trained decision tree's splits.

The features are then sorted and plotted as a horizontal bar chart.

------------------------------------------------------------------------

# 19. Decision Tree Visualization

The notebook uses:

``` python
plot_tree(
    id3_model,
    feature_names=features,
    class_names=id3_model.classes_,
    filled=True,
    rounded=True,
    proportion=False
)
```

This displays the learned tree.

The visualization shows:

-   Decision features
-   Split thresholds
-   Samples
-   Class distribution
-   Predicted class

------------------------------------------------------------------------

# 20. Complete Workflow

``` text
Online Retail.xlsx
        ↓
Load Data
        ↓
Remove missing CustomerID
        ↓
Remove cancelled invoices
        ↓
Remove invalid Quantity / UnitPrice
        ↓
Create TotalAmount
        ↓
Group transactions by CustomerID
        ↓
Create customer-level features
        ↓
Create Low / Medium / High Value segments
        ↓
Select X and y
        ↓
Train-Test Split
        ↓
DecisionTreeClassifier
criterion = entropy
        ↓
Train
        ↓
Predict
        ↓
Accuracy
Classification Report
Confusion Matrix
        ↓
Decision Tree Plot
        ↓
Feature Importance
```

------------------------------------------------------------------------

# 21. Important Viva Questions

### Q1. What is ID3?

ID3 is a decision-tree classification algorithm that selects attributes
using Information Gain.

### Q2. What is entropy?

Entropy measures impurity or uncertainty in a dataset.

### Q3. What is Information Gain?

It measures the reduction in entropy after a split.

### Q4. Which feature becomes the root in ID3?

The feature with the highest Information Gain.

### Q5. Why is entropy used?

To measure the impurity/uncertainty of the data.

### Q6. What does entropy = 0 mean?

The node is completely pure.

### Q7. Why `criterion="entropy"`?

To use entropy-based splitting, corresponding to the Information Gain
approach used by ID3.

### Q8. Why `max_depth=4`?

To limit tree depth and help control overfitting.

### Q9. Why is feature scaling not required?

Decision trees use feature-based splits rather than distance
calculations.

### Q10. Why is TotalSpend excluded from X?

Because Segment is created from TotalSpend. Including TotalSpend would
leak the information used to create the target.

------------------------------------------------------------------------

# 22. Important Note for Your Viva

If the examiner asks:

> **"Did you implement ID3 manually?"**

Say:

> **"No. I used `DecisionTreeClassifier` from scikit-learn with
> `criterion='entropy'`. This implements entropy-based decision-tree
> splitting, which follows the Information Gain principle used in
> ID3."**

Don't say that you coded the ID3 algorithm from scratch---the notebook
doesn't do that.

------------------------------------------------------------------------

## Technologies

-   Python
-   Pandas
-   NumPy
-   Matplotlib
-   Scikit-learn
-   Jupyter Notebook
-   Excel dataset

## Files

``` text
.
├── ID3 Decision Tree.ipynb
├── Online Retail.xlsx
└── README.md
```

## Conclusion

This notebook demonstrates an **entropy-based Decision Tree for customer
segmentation**. It starts with transaction-level retail data, cleans the
data, aggregates it to customer level, creates three spending-based
segments, trains an entropy-based decision tree, evaluates its
predictions, and visualizes both the tree and feature importance.
