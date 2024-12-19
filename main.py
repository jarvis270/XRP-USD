import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np

# Download historical data for Ethereum
eth = yf.Ticker("XRP-USD")
hist = eth.history(period="1y")[['Close']]

# Prepare the data
hist['Date'] = hist.index
hist['Date'] = pd.to_datetime(hist['Date'])
hist['Date'] = hist['Date'].map(pd.Timestamp.toordinal)
X = hist[['Date']]
y = hist['Close']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict on testing set to evaluate the model
predictions_test = model.predict(X_test)
mse = mean_squared_error(y_test, predictions_test)
r2 = r2_score(y_test, predictions_test)

print(f"\nMean Squared Error: {mse}")
print(f"R-squared (Confidence Score): {r2}")

# Predict the next 5 days
last_date = X['Date'].max()
future_dates = [last_date + i for i in range(1, 6)]
future_dates_df = pd.DataFrame(future_dates, columns=['Date'])
predictions = model.predict(future_dates_df)

# Show predictions for the next 5 days
for i, date in enumerate(future_dates):
    date_formatted = pd.Timestamp.fromordinal(date).strftime('%Y-%m-%d')
    print(f"Predicted Price for {date_formatted}: {predictions[i]}")
