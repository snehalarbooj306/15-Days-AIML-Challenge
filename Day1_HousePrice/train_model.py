from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import joblib

print("Loading dataset...")

# Load dataset
data = fetch_california_housing()
X = data.data
y = data.target

print("Splitting data...")

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training model...")

# Create and train model
model = RandomForestRegressor()
model.fit(X_train, y_train)

print("Testing model...")

# Test model
predictions = model.predict(X_test)
score = r2_score(y_test, predictions)

print("Model R2 Score:", score)

print("Saving model...")

# Save model
joblib.dump(model, "model.pkl")

print("Model saved as model.pkl ✅")
