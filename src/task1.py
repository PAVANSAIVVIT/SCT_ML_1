import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# --- Paths (works no matter where you run from) ---
THIS_FILE = Path(__file__).resolve()
ROOT = THIS_FILE.parents[1]          # project root (SCT_ML_1/)
DATA = ROOT / "data"
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

# Plot theme
sns.set_theme(style="whitegrid")

print("\nLoading dataset from:", DATA / "train.csv")
df = pd.read_csv(DATA / "train.csv")
print("Dataset loaded successfully!")

print("\nFirst 5 rows of the data:")
print(df.head())

# Select features/target
X = df[['SquareFootage', 'Bedrooms', 'Bathrooms']]
y = df['Price']

print("\nDataset description:")
print(df.describe(include='all'))

# 1) Correlation heatmap
plt.figure(figsize=(8, 5))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap", fontsize=16, fontweight="bold")
out1 = RESULTS / "output1_correlation_heatmap.png"
plt.savefig(out1, bbox_inches="tight", dpi=150)
plt.close()
print(f"Saved: {out1}")

# 2) Pairplot
pairplot = sns.pairplot(df[['SquareFootage','Bedrooms','Bathrooms','Price']],
                        diag_kind="kde", corner=True, height=2.5)
pairplot.fig.suptitle("Feature Relationships", fontsize=16, fontweight="bold")
pairplot.fig.subplots_adjust(top=0.93, bottom=0.12)
out2 = RESULTS / "output2_pairplot.png"
pairplot.savefig(out2, dpi=150)
plt.close('all')
print(f"Saved: {out2}")

# Train-test split and model
print("\nTraining Linear Regression model...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = LinearRegression()
model.fit(X_train, y_train)
print("Model training completed.")

# Predictions & metrics
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"\nMean Squared Error: {mse:,.2f}")
print(f"R² Score: {r2:.3f}")

coeff_df = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
print("\nModel Coefficients:")
print(coeff_df)
print(f"Intercept: {model.intercept_:,.2f}")

# 3) Actual vs Predicted scatter
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, alpha=0.6, edgecolor='k', label="Predictions")
mn, mx = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
plt.plot([mn, mx], [mn, mx], 'r--', lw=2, label="Perfect Prediction")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices", fontsize=16, fontweight="bold")
plt.legend()
out3 = RESULTS / "output3_actual_vs_pred.png"
plt.savefig(out3, bbox_inches="tight", dpi=150)
plt.close()
print(f"Saved: {out3}")

# Sample predictions
sample_data = pd.DataFrame({
    'SquareFootage': [2000, 1500, 3000],
    'Bedrooms': [3, 2, 4],
    'Bathrooms': [2, 1, 3]
})
sample_preds = model.predict(sample_data).round(2)
sample_data['Predicted Price'] = [f"₹{v:,.2f}" for v in sample_preds]

print("\nSample Predictions:")
print(sample_data.to_string(index=False))
