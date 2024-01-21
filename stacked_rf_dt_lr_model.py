import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import joblib
from sklearn.metrics import accuracy_score

# Load the original dataset
df = pd.read_csv('data.csv')

# Extract columns that end with '12' and the class label
selected_columns = [col for col in df.columns if col.endswith('17') or col == 'ID' or col == 'class']

# Create a new DataFrame with the selected columns
filtered_df = df[selected_columns]

# Save the filtered DataFrame to a new CSV file
filtered_df.to_csv('filtered_dataset.csv', index=False)

df = pd.read_csv('filtered_dataset.csv')
print(df)

X = df.drop(['ID', 'class'], axis=1)
print(X.columns)
y = df['class']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train)

# Define the stacked model
stacked_lr_model = StackingClassifier(
    estimators=[
        ('DT', DecisionTreeClassifier(random_state=42)),
        ('RF', RandomForestClassifier(n_estimators=100, random_state=42))
    ],
    final_estimator=LogisticRegression(random_state=42)
)

# Train the model
stacked_lr_model.fit(X_train, y_train)

# Evaluate the model
predictions = stacked_lr_model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Model Accuracy: {accuracy}")

# Save the model
joblib.dump(stacked_lr_model, 'stacked_lr_model_1.pkl')
