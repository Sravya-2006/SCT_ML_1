import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
train_df = pd.read_csv("/mnt/data/train.csv")
test_df = pd.read_csv("/mnt/data/test.csv")
features = ['GrLivArea', 'BedroomAbvGr', 'FullBath']
target = 'SalePrice'
train_df = train_df[features + [target]].dropna()
test_df = test_df[features].fillna(0)
X = train_df[features]
y = train_df[target]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_val)
rmse = mean_squared_error(y_val, y_pred, squared=False)
print(f"Validation RMSE: {rmse:.2f}")
test_preds = model.predict(test_df)
submission = pd.read_csv("/mnt/data/sample_submission.csv")
submission['SalePrice'] = test_preds
submission.to_csv("predictions.csv", index=False)
print("Predictions saved to 'predictions.csv'")
