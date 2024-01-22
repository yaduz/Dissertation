import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
import time

# Load the dataset
data = pd.read_csv('data.csv')

# Separate features and target variable
X = data.drop(columns=['class','ID'])
y = data['class']

# Initialize the number of features to select
k = 'all'  # Adjust 'k' as needed

# Chi-Squared Test
start_time = time.time()
chi2_selector = SelectKBest(chi2, k=k)
X_chi2 = chi2_selector.fit_transform(X, y)
selected_features_chi2 = X.columns[chi2_selector.get_support()]
num_selected_features_chi2 = len(selected_features_chi2)
end_time = time.time()
chi2_execution_time = end_time - start_time

# Mutual Information
start_time = time.time()
mi_selector = SelectKBest(mutual_info_classif, k=k)
X_mi = mi_selector.fit_transform(X, y)
selected_features_mi = X.columns[mi_selector.get_support()]
num_selected_features_mi = len(selected_features_mi)
end_time = time.time()
mi_execution_time = end_time - start_time

print(f"Chi-Squared Execution Time: {chi2_execution_time} seconds")
print(f"Mutual Information Execution Time: {mi_execution_time} seconds")

print(f"Chi-Squared Selected Features ({num_selected_features_chi2} features):")
print(selected_features_chi2)

print(f"Mutual Information Selected Features ({num_selected_features_mi} features):")
print(selected_features_mi)
