import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import lightgbm as lgb
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.feature_selection import SelectFromModel, SelectKBest, mutual_info_classif
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from deap import base, creator, tools, algorithms

# Load the dataset (replace 'data.csv' with your dataset)
data = pd.read_csv(r'Code\data.csv')

# Encode the target variable 'class' to numerical values
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(data['class'])

# Split the dataset into features (X) and target (y)
X = data.drop(columns=['ID', 'class'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a fitness function based on classification accuracy for Genetic Algorithm
def evaluate_features_ga(individual, X, y):
    selected_features = [bool(i) for i in individual]
    X_selected = X.iloc[:, selected_features]
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_selected, y)
    y_pred = clf.predict(X_selected)
    accuracy = accuracy_score(y, y_pred)
    return accuracy,

# Create a DEAP toolbox for Genetic Algorithm
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)
toolbox = base.Toolbox()

# Define individual attributes (binary features)
num_features = len(X.columns)
toolbox.register("attr_bool", np.random.choice, [0, 1])
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=num_features)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate_features_ga, X=X_train, y=y_train)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

# Create an initial population for Genetic Algorithm
population = toolbox.population(n=10)

# Define statistics to keep track of during evolution
stats_ga = tools.Statistics(lambda ind: ind.fitness.values)
stats_ga.register("max", np.max)
stats_ga.register("avg", np.mean)

# Run the genetic algorithm
results_ga, logbook_ga = algorithms.eaSimple(population, toolbox, cxpb=0.7, mutpb=0.2, ngen=10, stats=stats_ga, verbose=True)

# Get the best individual (selected features) from the final population
best_individual_ga = tools.selBest(results_ga, k=1)[0]
selected_features_ga = [bool(i) for i in best_individual_ga]
X_train_ga_selected = X_train.iloc[:, selected_features_ga]
X_test_ga_selected = X_test.iloc[:, selected_features_ga]

# Define a fitness function for Mutual Information-based feature selection
def evaluate_features_mi(X_train, y_train, X_test, y_test):
    mi_selector = SelectKBest(score_func=mutual_info_classif, k=5)  # You can adjust 'k' as needed
    X_train_mi_selected = mi_selector.fit_transform(X_train, y_train)
    X_test_mi_selected = mi_selector.transform(X_test)
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train_mi_selected, y_train)
    y_pred = clf.predict(X_test_mi_selected)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

# Evaluate Mutual Information-based feature selection
accuracy_mi = evaluate_features_mi(X_train, y_train, X_test, y_test)

# Baseline classifiers
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
knn_classifier = KNeighborsClassifier()
nb_classifier = GaussianNB()
lgbm_classifier = lgb.LGBMClassifier()
xgb_classifier = XGBClassifier(random_state=42)
lr_classifier = LogisticRegression(random_state=42)
SVM = SVC()
dt_classifier = DecisionTreeClassifier(random_state=42)

# Feature selection using Random Forest
rf_feature_selector = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42))
X_train_rf_selected = rf_feature_selector.fit_transform(X_train, y_train)
X_test_rf_selected = rf_feature_selector.transform(X_test)

# Stacked Ensemble Learning with Logistic Regression as final estimator
stacked_LR_estimator = StackingClassifier(estimators=[
    ('DT', dt_classifier),
    ('rf', rf_classifier)
], final_estimator=LogisticRegression(random_state=42))

lr_base_model = LogisticRegression(solver='liblinear', multi_class='auto', random_state=42)
# Create a BaggingClassifier with Logistic Regression as the base estimator
stacked_LR_bagging = BaggingClassifier(lr_base_model, n_estimators=100, random_state=42)

# Train and evaluate models
models = {
    "RF": rf_classifier,
    "KNN": knn_classifier,
    "NB": nb_classifier,
    "LGBM": lgbm_classifier,
    "XGB": xgb_classifier,
    "LR": lr_classifier,
    "SVM": SVM,
    "Stacked LR Estimator": stacked_LR_estimator,
    "Stacked LR Bagging": stacked_LR_bagging
}

results = {}
for name, model in models.items():
    if name == "Stacked LR Estimator":
        model.fit(X_train_ga_selected, y_train)  # Use GA-selected features for Stacked LR Estimator
        y_pred = model.predict(X_test_ga_selected)
    elif name == "Stacked LR Bagging":
        model.fit(X_train_rf_selected, y_train)  # Use RF-selected features for Stacked LR Bagging
        y_pred = model.predict(X_test_rf_selected)
    else:
        model.fit(X_train, y_train)  # No feature selection for other models
        y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy

# Print results
for name, accuracy in results.items():
    print(f"{name}: Accuracy = {accuracy:.4f}")

# Visualize the results
plt.figure(figsize=(10, 5))
plt.bar(results.keys(), results.values())
plt.xlabel("Classifier")
plt.ylabel("Accuracy")
plt.title("Classifier Comparison")
plt.ylim([0, 1])
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Compare the results
# print("Accuracy with Genetic Algorithm Feature Selection:", accuracy_ga)
print("Accuracy with Mutual Information Feature Selection:", accuracy_mi)
