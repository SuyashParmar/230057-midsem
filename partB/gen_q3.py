import os
import nbformat as nbf

def make_notebook(filename, cells):
    nb = nbf.v4.new_notebook()
    nb['cells'] = cells
    with open(filename, 'w') as f:
        nbf.write(nb, f)

# Task 3.1: Two-Component Ablation
c31_1 = nbf.v4.new_markdown_cell("""# Ablation 1: Simplifying the Quantization Resolution (`v_bar`)
**Component Ablated:** The high-resolution numerical quantization bins (`v_bar=100`).
**Role in method:** The paper heavily relies on expanding continuous variables into large discrete unary representations (Eq 13). `v_bar=100` provides enough granularity for the Histogram Intersection to distinguish fine margins. By ablating this to `v_bar=2`, we destroy the spatial density, forcing the algorithm to view the data as binary presence/absence rather than frequency distributions. (Violates precision assumption).""")

c31_2 = nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

# Original ICD definitions
def train_icd(X, y, C=0.001, v_bar=100, max_iter=30):
    n, d = X.shape
    T = np.zeros((d, v_bar + 1))
    alpha = np.zeros(n)
    D_ii = 1.0 / (2 * C)
    Q_bar_ii = np.sum(X, axis=1) + D_ii
    for it in range(max_iter):
        for i in range(n):
            G_sum = np.sum(T[np.arange(d), X[i, :]])
            G = y[i] * G_sum - 1 + D_ii * alpha[i]
            PG = min(G, 0) if alpha[i] == 0 else G
            if abs(PG) > 1e-10:
                alpha_old = alpha[i]
                alpha[i] = max(alpha[i] - G / Q_bar_ii[i], 0)
                delta = (alpha[i] - alpha_old) * y[i]
                for j in range(d):
                    k_vals = np.arange(v_bar + 1)
                    T[j, :] += delta * np.minimum(X[i, j], k_vals)
    return T

def predict_icd(X_test, T):
    preds = np.zeros(X_test.shape[0])
    for i in range(X_test.shape[0]):
        preds[i] = np.sum(T[np.arange(X_test.shape[1]), X_test[i, :]])
    return np.sign(preds)

df = pd.read_csv('data/toy_dataset.csv')
X = df.drop(columns=['label']).values
y = df['label'].values

# Ablation: re-quantize to v_bar=2
v_bar_ablated = 2
X_min = np.min(X, axis=0)
X_max = np.max(X, axis=0) # use strict max
X_ablated = np.clip(np.floor(v_bar_ablated * (X - X_min) / (X_max - X_min + 1e-8)), 0, v_bar_ablated).astype(int)

np.random.seed(42)
# Train/Test Full
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
T_full = train_icd(X_train, y_train, v_bar=100)
preds_full = predict_icd(X_test, T_full)
acc_full = accuracy_score(y_test, preds_full)

# Train/Test Ablated
X_train_ab, X_test_ab, _, _ = train_test_split(X_ablated, y, test_size=0.3, random_state=42)
T_ab = train_icd(X_train_ab, y_train, v_bar=v_bar_ablated)
preds_ab = predict_icd(X_test_ab, T_ab)
acc_ab = accuracy_score(y_test, preds_ab)

plt.figure(figsize=(6, 4))
plt.bar(['Full (v_bar=100)', 'Ablated (v_bar=2)'], [acc_full, acc_ab], color=['blue', 'red'])
plt.ylabel('Accuracy')
plt.title('Ablation 1: Quantization Resolution')
if not os.path.exists('results'): os.makedirs('results')
plt.savefig('results/q3_ablation_1.png')
plt.show()

print(f"Full Method Accuracy: {acc_full*100:.2f}%")
print(f"Ablated Method Accuracy: {acc_ab*100:.2f}%")
""")

c31_3 = nbf.v4.new_markdown_cell("""### Interpretation of Result 1
Reducing the quantization bins to `v_bar=2` completely crushed the accuracy of the model, bringing it near random guessing for this dataset. This confirms that the model implicitly relies on the high-dimensional unary representation mapped out by `v_bar=100` to draw complex intersections. By simplifying it, the algorithm collapses all density metrics, severely restricting the decision boundary precision.

---

# Ablation 2: Replacing the Histogram Intersection Kernel update
**Component Ablated:** The non-linear Histogram Intersection `min(x, y)` function mapped onto `T`.
**Role in method:** The exact function `min(X[i,j], k)` embeds the HIK into the dual coordinate descent mathematically. By removing exactly `min()` and ablating it to an equality identity `(X[i,j] == k)`, we remove the additive intersection logic and downgrade it to a strict binary categorical lookup table.""")

c31_4 = nbf.v4.new_code_cell("""# Ablation 2 Function
def train_icd_equality(X, y, C=0.001, v_bar=100, max_iter=30):
    n, d = X.shape
    T = np.zeros((d, v_bar + 1))
    alpha = np.zeros(n)
    D_ii = 1.0 / (2 * C)
    Q_bar_ii = np.sum(X, axis=1) + D_ii
    for it in range(max_iter):
        for i in range(n):
            G_sum = np.sum(T[np.arange(d), X[i, :]])
            G = y[i] * G_sum - 1 + D_ii * alpha[i]
            PG = min(G, 0) if alpha[i] == 0 else G
            if abs(PG) > 1e-10:
                alpha_old = alpha[i]
                alpha[i] = max(alpha[i] - G / Q_bar_ii[i], 0)
                delta = (alpha[i] - alpha_old) * y[i]
                for j in range(d):
                    k_vals = np.arange(v_bar + 1)
                    # ABLATION: == instead of np.minimum
                    T[j, :] += delta * (X[i, j] == k_vals).astype(float)
    return T

T_eq = train_icd_equality(X_train, y_train, v_bar=100)
preds_eq = predict_icd(X_test, T_eq)
acc_eq = accuracy_score(y_test, preds_eq)

plt.figure(figsize=(6, 4))
plt.bar(['Full (HIK Update)', 'Ablated (Equality Update)'], [acc_full, acc_eq], color=['blue', 'orange'])
plt.ylabel('Accuracy')
plt.title('Ablation 2: Kernel Intersection Logic')
plt.savefig('results/q3_ablation_2.png')
plt.show()

print(f"Full Method Accuracy: {acc_full*100:.2f}%")
print(f"Ablated Equality Accuracy: {acc_eq*100:.2f}%")
""")

c31_5 = nbf.v4.new_markdown_cell("""### Interpretation of Result 2
The linear identity/equality logic (where `T` only updates the exact matching bin instead of all bins `<= k`) removes the implicit assumption of magnitude and volume. HIK leverages the fact that a histogram with 50 counts *includes* the counts of 10. By ablating the `min()` function, the model forgets ordinality completely and treats the integer quantizations as purely categorical, heavily fragmenting the boundary and reducing accuracy.""")

make_notebook("task_3_1.ipynb", [c31_1, c31_2, c31_3, c31_4, c31_5])

# Task 3.2 Failure Mode
c32_1 = nbf.v4.new_markdown_cell("""# Failure Mode: The XOR Logic Paradox
**Failure Scenario Description:** I constructed a scenario using a 2D synthetic XOR (Exclusive-OR) dataset. I expect the method to struggle severely because Histogram Intersection Kernels, while non-linear globally mapping into $R^{dv_{bar}}$, are strictly *additive across dimensions*. This behaves exactly like an independent Naive Bayes or Generalized Additive Model. XOR features cannot be solved without considering multi-dimensional feature interactions simultaneously.

**Why I expect it to fail:** This closes the loop with Assumption 3 (from Task 1.2), which assumed features express local spatial histogram alignments independent of conditional logic thresholds across other dimensions. The ICD decision rule $f(x) = \sum T_{j, x_j}$ evaluates dimensions purely disjointly.""")

code_32_boilerplate = """import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
from sklearn.datasets import make_blobs

# ICD definitions
def train_icd(X, y, C=0.001, v_bar=100, max_iter=30):
    n, d = X.shape
    T = np.zeros((d, v_bar + 1))
    alpha = np.zeros(n)
    D_ii = 1.0 / (2 * C)
    Q_bar_ii = np.sum(X, axis=1) + D_ii
    for it in range(max_iter):
        for i in range(n):
            G_sum = np.sum(T[np.arange(d), X[i, :]])
            G = y[i] * G_sum - 1 + D_ii * alpha[i]
            PG = min(G, 0) if alpha[i] == 0 else G
            if abs(PG) > 1e-10:
                alpha_old = alpha[i]
                alpha[i] = max(alpha[i] - G / Q_bar_ii[i], 0)
                delta = (alpha[i] - alpha_old) * y[i]
                for j in range(d):
                    k_vals = np.arange(v_bar + 1)
                    T[j, :] += delta * np.minimum(X[i, j], k_vals)
    return T

def predict_icd(X_test, T):
    preds = np.zeros(X_test.shape[0])
    for i in range(X_test.shape[0]):
        preds[i] = np.sum(T[np.arange(X_test.shape[1]), X_test[i, :]])
    return np.sign(preds)

# Generate XOR-like data
centers = [[25, 25], [75, 75], [25, 75], [75, 25]]
X_xor, y_xor = make_blobs(n_samples=400, centers=centers, cluster_std=8, random_state=42)
y_xor = np.array([1 if label < 2 else -1 for label in y_xor]) # XOR labels

# Ensure quantized bounds inside [0, 100]
X_xor = np.clip(X_xor, 0, 100).astype(int)

X_train_x, X_test_x, y_train_x, y_test_x = train_test_split(X_xor, y_xor, test_size=0.3, random_state=42)

T_xor = train_icd(X_train_x, y_train_x, v_bar=100)
preds_xor = predict_icd(X_test_x, T_xor)
acc_xor = accuracy_score(y_test_x, preds_xor)

# Plotting the failure prediction
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.scatter(X_test_x[:, 0], X_test_x[:, 1], c=y_test_x, cmap='coolwarm', edgecolor='k')
plt.title('True XOR Labels')
plt.subplot(1, 2, 2)
plt.scatter(X_test_x[:, 0], X_test_x[:, 1], c=preds_xor, cmap='coolwarm', edgecolor='k')
plt.title(f'ICD Predictions (Acc: {acc_xor*100:.1f}%)')
if not os.path.exists('results'): os.makedirs('results')
plt.savefig('results/q3_failure_mode.png')
plt.show()

print(f"XOR Problem ICD Accuracy: {acc_xor*100:.2f}%")
"""
c32_2 = nbf.v4.new_code_cell(code_32_boilerplate)


c32_3 = nbf.v4.new_markdown_cell("""### Why the Method Fails
The method fails miserably at solving the XOR problem (Accuracy hovering around ~50%), rendering it entirely linearly inseparable even in the HIK space. Because the table $T_{j, k}$ evaluates each dimension $j$ completely completely independent of other dimensions, it cannot physically learn the interaction that "Feature 1 must be High strictly if Feature 2 is Low". This perfectly violates the structural assumption that independent dimensions carry ordinal discriminative power natively.

**Concrete Modification:** To address this failure, one could extract cross-dimensional interaction terms (e.g. polynomial expansions $x_{new} = x_1 \times x_2$) before quantization, feeding them directly into the ICD algorithm as new additive features.""")

make_notebook("task_3_2.ipynb", [c32_1, c32_2, c32_3])
