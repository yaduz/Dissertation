import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc, confusion_matrix
from sklearn.feature_selection import SelectFromModel, mutual_info_classif, SelectKBest
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import lightgbm as lgb
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, StackingClassifier
from sklearn.metrics import precision_recall_curve
from deap import base, creator, tools, algorithms
import os
import seaborn as sns

# Load the dataset 
data = pd.read_csv(r'Code\data.csv')

# Encode the target variable 'class' to numerical values
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(data['class'])

# Split the dataset into features (X) and target (y)
X = data.drop(columns=['ID', 'class'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a function to evaluate classifiers and return metrics
def evaluate_classifier(classifier, X_train, y_train, X_test, y_test):
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    if hasattr(classifier, 'predict_proba'):
        y_prob = classifier.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_prob)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        return accuracy, precision, recall, f1, roc_auc, fpr, tpr
    else:
        return accuracy, precision, recall, f1, None, None, None

# Initialize classifiers
classifiers = {
    "RF": RandomForestClassifier(n_estimators=100, random_state=42),
    "KNN": KNeighborsClassifier(),
    "NB": GaussianNB(),
    "LGBM": lgb.LGBMClassifier(),
    "XGB": XGBClassifier(random_state=42),
    # "LR": LogisticRegression(random_state=42),
    "SVM": SVC(probability=True),
    "Stacked_LR_Estimator": StackingClassifier(estimators=[
        ('DT', DecisionTreeClassifier(random_state=42)),
        ('RF', RandomForestClassifier(n_estimators=100, random_state=42))
    ], final_estimator=LogisticRegression(random_state=42)),
}

# Create a 'Results' folder if it doesn't exist
if not os.path.exists('Results'):
    os.makedirs('Results')

# Initialize lists to store results
classifier_names = []
accuracies = []
precisions = []
recalls = []
f1_scores = []
roc_aucs = []
fprs = []
tprs = []

# Evaluate classifiers and store results
for name, classifier in classifiers.items():
    accuracy, precision, recall, f1, roc_auc, fpr, tpr = evaluate_classifier(classifier, X_train, y_train, X_test, y_test)
    classifier_names.append(name)
    accuracies.append(accuracy)
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)
    roc_aucs.append(roc_auc)
    fprs.append(fpr)
    tprs.append(tpr)

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
population = toolbox.population(n=15)

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

# Evaluate Mutual Information-based feature selection
mi_selector = SelectKBest(score_func=mutual_info_classif, k=5)  # You can adjust 'k' as needed
X_train_mi_selected = mi_selector.fit_transform(X_train, y_train)
X_test_mi_selected = mi_selector.transform(X_test)

# Train and evaluate models with GA-selected features
for name, classifier in classifiers.items():
    if name == "Stacked_LR_Estimator":
        accuracy, precision, recall, f1, roc_auc, fpr, tpr = evaluate_classifier(classifier, X_train_ga_selected, y_train, X_test_ga_selected, y_test)
    else:
        accuracy, precision, recall, f1, roc_auc, fpr, tpr = evaluate_classifier(classifier, X_train, y_train, X_test, y_test)
    
    classifier_names.append(f"{name} (GA)")
    accuracies.append(accuracy)
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)
    roc_aucs.append(roc_auc)
    fprs.append(fpr)
    tprs.append(tpr)

# Train and evaluate models with MI-selected features
for name, classifier in classifiers.items():
    if name == "Stacked_LR_Estimator":
        accuracy, precision, recall, f1, roc_auc, fpr, tpr = evaluate_classifier(classifier, X_train_mi_selected, y_train, X_test_mi_selected, y_test)
    else:
        accuracy, precision, recall, f1, roc_auc, fpr, tpr = evaluate_classifier(classifier, X_train, y_train, X_test, y_test)
    
    classifier_names.append(f"{name} (MI)")
    accuracies.append(accuracy)
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)
    roc_aucs.append(roc_auc)
    fprs.append(fpr)
    tprs.append(tpr)

def annotate_bars(ax):
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.2f}", (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', fontsize=10, color='black', xytext=(0, 5), textcoords='offset points')

# Plot and save results in separate windows
plt.figure(figsize=(8, 6))

# Plot Precision
plt.bar(classifier_names, precisions)
plt.title('Precision')
plt.ylim([0, 1])
ax = plt.gca()
annotate_bars(ax)
plt.xticks(rotation='vertical')
plt.savefig('Results/precision.png')
plt.show()

# Plot Recall
plt.figure(figsize=(8, 6))
plt.bar(classifier_names, recalls)
plt.title('Recall')
plt.ylim([0, 1])
ax = plt.gca()
annotate_bars(ax)
plt.xticks(rotation='vertical')
plt.savefig('Results/recall.png')
plt.show()

# Plot F1-Score
plt.figure(figsize=(8, 6))
plt.bar(classifier_names, f1_scores)
plt.title('F1-Score')
plt.ylim([0, 1])
ax = plt.gca()
annotate_bars(ax)
plt.xticks(rotation='vertical')
plt.savefig('Results/f1_score.png')
plt.show()

# Plot all AUC-ROC curves
plt.figure(figsize=(10, 8))
for i in range(len(classifier_names)):
    if roc_aucs[i] is not None:
        plt.plot(fprs[i], tprs[i], lw=2, label=f'{classifier_names[i]} (AUC = {roc_aucs[i]:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend(loc="lower right")
plt.savefig('Results/roc_auc_combined.png')
plt.show()

# Plot Accuracy
plt.figure(figsize=(8, 6))
plt.bar(classifier_names, accuracies)
plt.title('Accuracy')
plt.ylim([0, 1])
ax = plt.gca()
annotate_bars(ax)
plt.xticks(rotation='vertical')
plt.savefig('Results/accuracy.png')
plt.show()

# Calculate and plot Confusion Matrix for the best classifier
best_classifier = classifiers["Stacked_LR_Estimator"]
y_pred_best = best_classifier.predict(X_test_ga_selected)
labels = np.unique(y)
cm = confusion_matrix(y_test, y_pred_best, labels=labels)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig('Results/confusion_matrix.png')
plt.show()