# Customer Segmentation using K-Means Clustering

## Objective

Group retail customers into clusters based on:

- Annual Income
- Spending Score

## Dataset

Mall Customers Dataset (Kaggle)

## Libraries Used

- pandas
- matplotlib
- scikit-learn

## Steps

1. Load the dataset
2. Select relevant features
3. Use the Elbow Method to estimate the number of clusters
4. Train a K-Means model
5. Visualize customer groups
6. Save the clustered dataset

## Run

```bash
python kmeans_customer_segmentation.py
```

## Output

- Elbow Method graph
- Customer cluster visualization
- `Customer_Segments.csv` containing assigned cluster labels