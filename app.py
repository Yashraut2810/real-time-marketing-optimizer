import time
import random
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Suppress sklearn warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Generate synthetic data with 10,000 entries
np.random.seed(42)
n_samples = 10000
synthetic_data = pd.DataFrame({
    'customer_id': range(n_samples),
    'website_visits': np.random.randint(0, 30, n_samples),
    'prescription_purchases': np.random.randint(0, 10, n_samples),
})

# Generate realistic response labels
synthetic_data['responded'] = (
    0.7 * (synthetic_data['website_visits'] / 30) +
    0.3 * (synthetic_data['prescription_purchases'] / 10) +
    np.random.normal(0, 0.1, n_samples)
).clip(0, 1).round().astype(int)

# Smart Segmentation using KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
synthetic_data['segment'] = kmeans.fit_predict(synthetic_data[['website_visits', 'prescription_purchases']])
segment_labels = {0: "Low Engagement", 1: "Medium Engagement", 2: "High Engagement"}
synthetic_data['segment_label'] = synthetic_data['segment'].map(segment_labels)

# Data Integrity Check: anomaly flag
synthetic_data['anomaly'] = synthetic_data.apply(
    lambda row: 1 if row['website_visits'] == 0 and row['prescription_purchases'] > 0 else 0,
    axis=1
)
print(f"[!] {synthetic_data['anomaly'].sum()} data anomalies detected: purchases without visits.\n")

# Split data
feature_columns = ['website_visits', 'prescription_purchases']
X = synthetic_data[feature_columns]
y = synthetic_data['responded']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print("=== Model Performance Metrics ===")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}\n")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['No Response', 'Response'], yticklabels=['No Response', 'Response'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('confusion_matrix.png')
plt.close()

# ROC Curve
y_prob = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.savefig('roc_curve.png')
plt.close()

# Initialize customer features
customer_features = {i: {'website_visits': 0, 'prescription_purchases': 0} for i in range(100)}

def update_features(data):
    cid = data['customer_id']
    action = data['action']
    if action == 'visit_website':
        customer_features[cid]['website_visits'] += 1
    elif action == 'purchase_prescription':
        customer_features[cid]['prescription_purchases'] += 1

def make_prediction(customer_id):
    features = customer_features[customer_id]
    X_pred = pd.DataFrame([[
        features['website_visits'],
        features['prescription_purchases']
    ]], columns=feature_columns)
    prob = model.predict_proba(X_pred)[0][1]

    # Financial Logic
    promo_cost = 2
    expected_revenue = 10
    expected_profit = prob * expected_revenue - promo_cost

    if expected_profit > 0:
        action = random.choice([
            "email campaign with drug info",
            "discount on next prescription",
            "personalized consultation offer"
        ])
    else:
        action = "no promotion"

    return {
        'customer_id': customer_id,
        'website_visits': features['website_visits'],
        'prescription_purchases': features['prescription_purchases'],
        'probability': prob,
        'expected_profit': expected_profit,
        'recommendation': action
    }

def generate_data():
    actions = ['visit_website', 'purchase_prescription']
    while True:
        cid = random.randint(0, 99)
        action = random.choice(actions)
        yield {'customer_id': cid, 'action': action}
        time.sleep(0.1)

# Run simulation
print("=== Real-Time Marketing Campaign Optimization ===")
print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
data_stream = generate_data()
output_results = []

for i, data in enumerate(data_stream):
    update_features(data)
    result = make_prediction(data['customer_id'])

    # Inconsistency check
    if result['website_visits'] == 0 and result['prescription_purchases'] > 0:
        print(f"[!] Data inconsistency: Customer {result['customer_id']} has {result['prescription_purchases']} purchases but 0 visits.")

    output_results.append(result)
    print(f"Customer {result['customer_id']:>3} | Visits: {result['website_visits']:>2} | Purchases: {result['prescription_purchases']:>2} | "
          f"Response Prob: {result['probability']:.2f} | Profit: {result['expected_profit']:.2f} | Action: {result['recommendation']}")

    if i >= 9:
        df_output = pd.DataFrame(output_results)
        df_output.to_csv("real_time_predictions.csv", index=False)
        print(f"\nEnded at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=== Demo Complete ===")
        print("\nVisualizations saved as 'confusion_matrix.png', 'roc_curve.png', and results in 'real_time_predictions.csv'")
        break