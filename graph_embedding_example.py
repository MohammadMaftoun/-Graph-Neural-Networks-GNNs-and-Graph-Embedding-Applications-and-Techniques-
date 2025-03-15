# Import libraries
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import kneighbors_graph
from node2vec import Node2Vec
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, matthews_corrcoef, classification_report, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load and prepare the dataset
data = load_breast_cancer()
X = data.data
y = data.target
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
df = pd.DataFrame(X_scaled, columns=data.feature_names)
df['label'] = y
print("Dataset Shape:", df.shape)
print("Class Distribution:\n", df['label'].value_counts())

# Step 2: Create a graph network
k = 5
adj_matrix = kneighbors_graph(X_scaled, n_neighbors=k, mode='connectivity', include_self=False)
G = nx.from_scipy_sparse_array(adj_matrix)
G = nx.relabel_nodes(G, {i: str(i) for i in range(len(G.nodes))})
print("Number of nodes:", G.number_of_nodes())
print("Number of edges:", G.number_of_edges())

# Step 3: Generate node embeddings with Node2vec
node2vec = Node2Vec(G, dimensions=128, walk_length=100, num_walks=10, p=0.5, q=2.0, workers=4)
model = node2vec.fit(window=10, min_count=1, batch_words=4)
embeddings = np.array([model.wv[str(i)] for i in range(len(G.nodes))])
print("Embeddings Shape:", embeddings.shape)

# Step 4: Train and evaluate classifiers
X_train, X_test, y_train, y_test = train_test_split(embeddings, y, test_size=0.2, random_state=42)
classifiers = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "K-Neighbors": KNeighborsClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "GaussianNB": GaussianNB(),
    "Support Vector Classifier": SVC(probability=True, random_state=42)
}
results = {
    "Classifier": [],
    "Accuracy": [],
    "MCC": [],
    "Classification Report": [],
    "Confusion Matrix": []
}
for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["Malignant", "Benign"], output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    results["Classifier"].append(name)
    results["Accuracy"].append(accuracy)
    results["MCC"].append(mcc)
    results["Classification Report"].append(report)
    results["Confusion Matrix"].append(cm)

results_df = pd.DataFrame({
    "Classifier": results["Classifier"],
    "Accuracy": results["Accuracy"],
    "MCC": results["MCC"]
})
print("\nClassifier Performance:")
print(results_df)

# Step 5: Detailed classification reports
for i, name in enumerate(results["Classifier"]):
    print(f"\nClassification Report for {name}:")
    report = results["Classification Report"][i]
    report_df = pd.DataFrame(report).transpose()
    print(report_df)

# Step 6: Visualize results
# Bar chart for accuracy
plt.figure(figsize=(10, 6))
sns.barplot(x="Accuracy", y="Classifier", hue="Classifier", data=results_df)
plt.title("Classifier Performance Comparison (Accuracy)")
plt.xlabel("Accuracy")
plt.ylabel("Classifier")
plt.show()

# Heatmaps for confusion matrices
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.ravel()
for i, (name, cm) in enumerate(zip(results["Classifier"], results["Confusion Matrix"])):
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                xticklabels=["Malignant", "Benign"], yticklabels=["Malignant", "Benign"])
    axes[i].set_title(f"Confusion Matrix: {name}")
    axes[i].set_xlabel("Predicted")
    axes[i].set_ylabel("True")
for i in range(len(results["Classifier"]), len(axes)):
    fig.delaxes(axes[i])
plt.tight_layout()
plt.show()
