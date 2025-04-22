import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os

# Load dataset
df = pd.read_csv("nse_data/final_gap_training_dataset.csv")

# Drop rows with missing values and reset index
df = df.dropna().reset_index(drop=True)

# Select features and label
features = ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'VWAP', 'TOTTRDQTY', 'CALL_OI', 'PUT_OI', 'PCR']
x = df[features]
y = df['GAP_LABEL']

# Encode labels
y = y.astype('category').cat.codes

# Split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train, y_train)

# Evaluate
y_pred = model.predict(x_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save model and label mapping
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/gap_prediction_model.pkl")
label_mapping = dict(enumerate(df['GAP_LABEL'].astype('category').cat.categories))
joblib.dump(label_mapping, "models/label_mapping.pkl")
print("\n✅ Model and label mapping saved to models directory")
