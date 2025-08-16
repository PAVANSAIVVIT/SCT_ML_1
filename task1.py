import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Set a clean theme for plots
sns.set_theme(style="whitegrid")

# Load dataset
print("\nLoading dataset...")
df = pd.read_csv("train.csv")  # Make sure train.csv is in the same folder
print("Dataset loaded successfully!")

# Show first few rows
print("\nFirst 5 rows of the data:")
print(df.head())

# Features and target
X = df[['SquareFootage', 'Bedrooms', 'Bathrooms']]
y = df['Price']

# Basic stats
print("\nDataset description:")
print(df.describe())

# Correlation heatmap
plt.figure(figsize=(8, 5))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap", fontsize=16, fontweight="bold")
plt.show()

# Pairplot with fixed title and visible x-labels
pairplot = sns.pairplot(df, diag_kind="kde", corner=True, height=2.5)
pairplot.fig.suptitle("Feature Relationships", fontsize=16, fontweight="bold")
pairplot.fig.subplots_adjust(top=0.93, bottom=0.12)  # extra bottom space for labels
plt.show()

# Train-test split
print("\nTraining Linear Regression model...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = LinearRegression()
model.fit(X_train, y_train)
print("Model training completed.")

# Predictions
y_pred = model.predict(X_test)

# Metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"\nMean Squared Error: {mse:,.2f}")
print(f"R² Score: {r2:.3f}")

# Model coefficients
coeff_df = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
print("\nModel Coefficients:")
print(coeff_df)
print(f"Intercept: {model.intercept_:,.2f}")

# Plot Actual vs Predicted
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, alpha=0.6, edgecolor='k', label="Predictions")
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2, label="Perfect Prediction")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices", fontsize=16, fontweight="bold")
plt.legend()
plt.show()

# Sample predictions
sample_data = pd.DataFrame({
    'SquareFootage': [2000, 1500, 3000],
    'Bedrooms': [3, 2, 4],
    'Bathrooms': [2, 1, 3]
})
sample_data['Predicted Price'] = model.predict(sample_data).round(2)
sample_data['Predicted Price'] = sample_data['Predicted Price'].apply(lambda x: f"₹{x:,.2f}")

print("\nSample Predictions:")
print(sample_data.to_string(index=False))
