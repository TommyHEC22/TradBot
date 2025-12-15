import investpy
import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
import requests
import csv
import shutil

Webhook_url = os.environ["DISCORD_WEBHOOK_URL"]
TWELVEDATA_API_KEY = os.environ["TWELVEDATA_API_KEY"]

pairs = ['EUR/USD', 'GBP/USD', 'USD/CHF', 'AUD/USD', 'USD/CAD', 'NZD/USD', 'EUR/GBP', 'AUD/CAD', 'EUR/AUD', 'GBP/CAD', 'GBP/CHF', 'AUD/CHF', 'EUR/CHF', 'EUR/CAD', 'NZD/CHF', 'USD/SGD']

long = []
short = []

CURRENCY_COUNTRIES = {
    'USD': ['United States'],
    'EUR': [
        'Austria', 'Belgium', 'Cyprus', 'Estonia', 'Finland', 'France', 
        'Germany', 'Greece', 'Ireland', 'Italy', 'Latvia', 'Lithuania', 
        'Luxembourg', 'Malta', 'Netherlands', 'Portugal', 'Slovakia', 
        'Slovenia', 'Spain'
    ],
    'GBP': ['United Kingdom'],
    'CHF': ['Switzerland'],
    'CAD': ['Canada'],
    'AUD': ['Australia'],
    'NZD': ['New Zealand'],
    'SGD': ['Singapore']
}

def currency_safety(currency):
    def fetch_events_next_24h():
        now = datetime.now()
        end_time = now + timedelta(hours=24)

        try:
            df = investpy.economic_calendar(
                from_date=now.strftime("%d/%m/%Y"),
                to_date=end_time.strftime("%d/%m/%Y"),
                countries=CURRENCY_COUNTRIES[currency],
                importances=["high", "medium", "low"]
            )

            # Filter to events within exact 24h window
            def in_window(row):
                dt_str = f"{row['date']} {row['time']}"
                try:
                    event_time = datetime.strptime(dt_str, "%d/%m/%Y %H:%M")
                except ValueError:
                    # Some events might have 'All Day' or blank time; skip those
                    return False
                return now <= event_time <= end_time

            df = df[df.apply(in_window, axis=1)]
            return df

        except Exception as e:
            print("Error fetching ", currency, " events:", e)
            return None

    def assess_safety(events_df):
        if events_df is None or events_df.empty:
            return "SAFE"  # No events → safe

        high_count = (events_df["importance"].str.lower() == "high").sum()
        medium_count = (events_df["importance"].str.lower() == "medium").sum()

        if high_count > 0:
            return "UNSAFE"
        if medium_count > 1:
            return "UNSAFE"
        return "SAFE"

    events_df = fetch_events_next_24h()
    result = assess_safety(events_df)
    return result

def safe_pairs(pairs):

    # Get safety status for currencies once to avoid multiple API calls
    usd_safety = currency_safety('USD')
    eur_safety = currency_safety('EUR')
    gbp_safety = currency_safety('GBP')
    chf_safety = currency_safety('CHF')
    cad_safety = currency_safety('CAD')
    aud_safety = currency_safety('AUD')
    nzd_safety = currency_safety('NZD')
    sgd_safety = currency_safety('SGD')

    if usd_safety == 'UNSAFE':
        to_remove = ["EUR/USD","GBP/USD","USD/CHF","AUD/USD","USD/CAD","NZD/USD","USD/SGD"]
        pairs = [s for s in pairs if s not in to_remove]

    if eur_safety == 'UNSAFE':
        to_remove = ["EUR/USD","EUR/GBP","EUR/AUD","EUR/CHF","EUR/CAD"]
        pairs = [s for s in pairs if s not in to_remove]

    if gbp_safety == 'UNSAFE':
        to_remove = ["GBP/USD","EUR/GBP","GBP/CAD","GBP/CHF"]
        pairs = [s for s in pairs if s not in to_remove]

    if chf_safety == 'UNSAFE':
        to_remove = ["USD/CHF","EUR/CHF","GBP/CHF","AUD/CHF","NZD/CHF"]
        pairs = [s for s in pairs if s not in to_remove]

    if cad_safety == 'UNSAFE':
        to_remove = ["AUD/CAD","EUR/CAD","GBP/CAD","USD/CAD"]
        pairs = [s for s in pairs if s not in to_remove]

    if aud_safety == 'UNSAFE':
        to_remove = ["AUD/USD","AUD/CAD","AUD/CHF","EUR/AUD"]
        pairs = [s for s in pairs if s not in to_remove]

    if nzd_safety == 'UNSAFE':
        to_remove = ["NZD/USD","NZD/CHF"]
        pairs = [s for s in pairs if s not in to_remove]

    if sgd_safety == 'UNSAFE':
        to_remove = ["USD/SGD"]
        pairs = [s for s in pairs if s not in to_remove]

    return pairs

def get_daily_data(currency_pair):
    api_key = TWELVEDATA_API_KEY
    url = 'https://api.twelvedata.com/time_series'

    params = {
        'symbol': currency_pair,
        'interval': '1day',
        'end_date': (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d'),
        'outputsize': 5000,
        'apikey': api_key,
    }

    response = requests.get(url, params=params)
    data = response.json()

    if 'values' in data:
        values = data['values']

        # Get the header keys from one of the entries
        headers = values[0].keys()

        # Save the original data.csv file
        with open('data.csv', 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()
            writer.writerows(values)

        print("Data saved to data.csv")

        # Create copies for the other models
        try:
            shutil.copy('data.csv', 'data2.csv')
            print("Copy created: data2.csv")
            
            shutil.copy('data.csv', 'data3.csv')
            print("Copy created: data3.csv")
            
            print("All data files ready:")
            print("- data.csv (original)")
            print("- data2.csv (copy for second model)")
            print("- data3.csv (copy for third model)")
            
        except Exception as e:
            print(f"Warning: Could not create copies - {e}")
            print("Only data.csv was created successfully")

    else:
        print("Error fetching data:", data.get('message', 'Unknown error'))

    return

def funnydaything_prediction():
  print("starting funnydaything")
  # Load the data
  file_path = 'data.csv'
  print("loaded data")
  data = pd.read_csv(file_path)
  print("read csv")
  data = data.round(4)

  # Handle missing values
  data.fillna(0, inplace=True)

  # Sort data by Date
  data.sort_values('datetime', inplace=True)
  print("sorted data")

  # Calculate the percentage change for the 'close' column and multiply by 100
  data['Change %'] = data['close'].pct_change() * 100

  # Optionally round the values to 2 decimal places
  data['Change %'] = data['Change %'].round(2)

  # Create new features: Moving Averages
  data['MA_5'] = data['close'].rolling(window=5).mean()
  data['MA_10'] = data['close'].rolling(window=10).mean()
  print("created moving averages")
  # Create lag features
  for i in range(1, 6):
      data[f'close_Lag_{i}'] = data['close'].shift(i)
  print("created lag features")
  # Drop rows with NaN values (resulting from lag features)
  data.dropna(inplace=True)

  # Specify how many recent entries to remove for testing
  num_recent_entries_to_remove = 1  # Change this value as needed

  # Split the data into training and testing sets
  data_train = data.iloc[:-num_recent_entries_to_remove]
  data_test = data.iloc[-num_recent_entries_to_remove:]

  # Prepare the feature matrix and target vector for training
  X_train = data_train[['open', 'high', 'low', 'Change %', 'MA_5', 'MA_10'] + [f'close_Lag_{i}' for i in range(1, 6)]].values
  y_train = data_train['close'].values

  # Prepare the feature matrix for testing
  X_test = data_test[['open', 'high', 'low', 'Change %', 'MA_5', 'MA_10'] + [f'close_Lag_{i}' for i in range(1, 6)]].values
  print("prepared features and target")
  # Standardize the features
  scaler = StandardScaler()
  X_train_scaled = scaler.fit_transform(X_train)
  X_test_scaled = scaler.transform(X_test)
  print("standardized features")
  # Initialize the XGBoost model
  xgb_model = XGBRegressor()

  # Define the parameter grid for hyperparameter tuning
  param_grid = {
      'n_estimators': [50, 100, 200],
      'max_depth': [3, 5, 7],
      'learning_rate': [0.01, 0.1, 0.2]
  }
  print("defined parameter grid")
  # Perform grid search
  print("starting grid search")
  grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error')
  grid_search.fit(X_train_scaled, y_train)
  print("completed grid search")
  # Best parameters
  best_params = grid_search.best_params_

  # Train the best XGBoost model
  best_xgb_model = grid_search.best_estimator_
  best_xgb_model.fit(X_train_scaled, y_train)
  print("trained best model")
  # Predict the next day's close using the removed entries for testing
  next_day_close_prediction = best_xgb_model.predict(X_test_scaled)

  ndpp = next_day_close_prediction[0]

  newest_close = data.loc[0, 'close']

  def pChange(ndpp, newest_close):
    pChange = ndpp/newest_close

    if pChange > 1:
      pChange = pChange - 1
      pChange = round(pChange*100, 2)
      return pChange

    elif pChange < 100:
      pChange = pChange - 1
      pChange = round(pChange*100, 2)
      return pChange

  print(pChange(ndpp, newest_close))

  if -0.08 > pChange(ndpp, newest_close) > -0.33:
      return -1  
  elif 0.08 < pChange(ndpp, newest_close) < 0.33:
      return 1
  else:
      return 0
  
def boxstock_prediction():
    # Load the data
    file_path = 'data2.csv'
    data = pd.read_csv(file_path)

    data = data.round(4)

    # Handle missing values
    data.fillna(0, inplace=True)

    # Sort data by Date
    data.sort_values('datetime', inplace=True)

      # Calculate the percentage change for the 'close' column and multiply by 100
    data['Change %'] = data['close'].pct_change() * 100

    # Optionally round the values to 2 decimal places
    data['Change %'] = data['Change %'].round(2)

    # Create new features: Moving Averages
    data['MA_5'] = data['close'].rolling(window=5).mean()
    data['MA_10'] = data['close'].rolling(window=10).mean()

    # Create lag features
    for i in range(1, 6):
        data[f'close_Lag_{i}'] = data['close'].shift(i)

    # Drop rows with NaN values (resulting from lag features)
    data.dropna(inplace=True)

    # Drop rows with NaN values (resulting from new indicators)
    data.dropna(inplace=True)

    # Prepare the feature matrix and target vector for training
    X = data[['open', 'high', 'low', 'Change %', 'MA_5', 'MA_10'] + [f'close_Lag_{i}' for i in range(1, 6)]].values
    y = data['close'].values

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Initialize the XGBoost model
    xgb_model = XGBRegressor()

    # Define the parameter grid for hyperparameter tuning
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2]
    }

    # Perform grid search
    grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error')
    grid_search.fit(X_scaled, y)

    # Best parameters
    best_params = grid_search.best_params_

    # Train the best XGBoost model
    best_xgb_model = grid_search.best_estimator_
    best_xgb_model.fit(X_scaled, y)

    # Use the latest available data point for prediction
    latest_data_point = X[-1].reshape(1, -1)
    latest_data_point_scaled = scaler.transform(latest_data_point)

    # Predict the next day's close
    next_day_close_prediction = best_xgb_model.predict(latest_data_point_scaled)

    ndpp = next_day_close_prediction[0]

    newest_close = data.loc[0, 'close']

    def pChange(ndpp, newest_close):
        pChange = ndpp/newest_close

        if pChange > 1:
            pChange = pChange - 1
            pChange = round(pChange*100, 2)
            return pChange

        elif pChange < 100:
            pChange = pChange - 1
            pChange = round(pChange*100, 2)
            return pChange
        
    print(pChange(ndpp, newest_close))

    if -0.08 > pChange(ndpp, newest_close) > -0.33:
      return -1  
    elif 0.08 < pChange(ndpp, newest_close) < 0.33:
      return 1
    else:
      return 0

def techindicate_prediction():
    # Load the data
    file_path = 'data3.csv'
    data = pd.read_csv(file_path)

    # Handle missing values
    data.fillna(0, inplace=True)

    data = data.round(4)

    # Sort data by Date
    data.sort_values('datetime', inplace=True)

        # Sort data by Date
    data.sort_values('datetime', inplace=True)

      # Calculate the percentage change for the 'close' column and multiply by 100
    data['Change %'] = data['close'].pct_change() * 100

    # Create new features: Moving Averages
    data['MA_5'] = data['close'].rolling(window=5).mean()
    data['MA_10'] = data['close'].rolling(window=10).mean()
    data['MA_20'] = data['close'].rolling(window=20).mean()
    data['EMA_12'] = data['close'].ewm(span=12, adjust=False).mean()
    data['EMA_26'] = data['close'].ewm(span=26, adjust=False).mean()

    # Create lag features
    for i in range(1, 6):
        data[f'close_Lag_{i}'] = data['close'].shift(i)

    # Drop rows with NaN values (resulting from lag features)
    data.dropna(inplace=True)

    # Relative Strength Index (RSI)
    window_length = 14
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window_length).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window_length).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))

    # Moving Average Convergence Divergence (MACD)
    ema_12 = data['close'].ewm(span=12, adjust=False).mean()
    ema_26 = data['close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = ema_12 - ema_26
    data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()

    # Bollinger Bands
    data['Bollinger_Mid'] = data['close'].rolling(window=20).mean()
    data['Bollinger_Upper'] = data['Bollinger_Mid'] + (data['close'].rolling(window=20).std() * 2)
    data['Bollinger_lower'] = data['Bollinger_Mid'] - (data['close'].rolling(window=20).std() * 2)

    # Average True Range (ATR)
    high_low = data['high'] - data['low']
    high_close = np.abs(data['high'] - data['close'].shift())
    low_close = np.abs(data['low'] - data['close'].shift())
    tr = high_low.combine(high_close, max).combine(low_close, max)
    data['ATR'] = tr.rolling(window=14).mean()

    # Stochastic Oscillator
    low_14 = data['low'].rolling(window=14).min()
    high_14 = data['high'].rolling(window=14).max()
    data['%K'] = 100 * (data['close'] - low_14) / (high_14 - low_14)
    data['%D'] = data['%K'].rolling(window=3).mean()

    # Commodity Channel Index (CCI)
    typical_close = (data['high'] + data['low'] + data['close']) / 3
    mean_typical_close = typical_close.rolling(window=20).mean()
    mean_deviation = (typical_close - mean_typical_close).abs().rolling(window=20).mean()
    data['CCI'] = (typical_close - mean_typical_close) / (0.015 * mean_deviation)

    # Rate of Change (ROC)
    data['ROC'] = ((data['close'] - data['close'].shift(12)) / data['close'].shift(12)) * 100

    # Williams %R
    high_n = data['high'].rolling(window=14).max()
    low_n = data['low'].rolling(window=14).min()
    data['Williams_%R'] = (high_n - data['close']) / (high_n - low_n) * -100

    # Chande Momentum Oscillator (CMO)
    up_days = delta.where(delta > 0, 0).rolling(window=14).sum()
    down_days = -delta.where(delta < 0, 0).rolling(window=14).sum()
    data['CMO'] = 100 * (up_days - down_days) / (up_days + down_days)

    # Keltner Channel
    data['Keltner_Mid'] = data['close'].rolling(window=20).mean()
    data['Keltner_Upper'] = data['Keltner_Mid'] + 2 * data['ATR']
    data['Keltner_lower'] = data['Keltner_Mid'] - 2 * data['ATR']

    # Pivot Points
    data['Pivot'] = (data['high'].shift(1) + data['low'].shift(1) + data['close'].shift(1)) / 3
    data['Pivot_R1'] = 2 * data['Pivot'] - data['low'].shift(1)
    data['Pivot_R2'] = data['Pivot'] + (data['high'].shift(1) - data['low'].shift(1))
    data['Pivot_S1'] = 2 * data['Pivot'] - data['high'].shift(1)
    data['Pivot_S2'] = data['Pivot'] - (data['high'].shift(1) - data['low'].shift(1))

    # Drop rows with NaN values (resulting from new indicators)
    data.dropna(inplace=True)

    # Prepare the feature matrix and target vector for training
    features = ['open', 'high', 'low', 'Change %', 'MA_5', 'MA_10', 'MA_20', 'EMA_12', 'EMA_26',
                'RSI', 'MACD', 'Signal_Line', 'Bollinger_Mid', 'Bollinger_Upper', 'Bollinger_lower',
                'ATR', '%K', '%D', 'CCI', 'ROC', 'Williams_%R', 'CMO',
                'Keltner_Mid', 'Keltner_Upper', 'Keltner_lower', 'Pivot', 'Pivot_R1', 'Pivot_R2',
                'Pivot_S1', 'Pivot_S2'] + [f'close_Lag_{i}' for i in range(1, 6)]
    X = data[features].values
    y = data['close'].values

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Initialize the XGBoost model
    xgb_model = XGBRegressor()

    # Define the parameter grid for hyperparameter tuning
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2]
    }

    # Perform grid search
    grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error')
    grid_search.fit(X_scaled, y)

    # Best parameters
    best_params = grid_search.best_params_

    # Train the best XGBoost model
    best_xgb_model = grid_search.best_estimator_
    best_xgb_model.fit(X_scaled, y)

    # Use the latest available data point for prediction
    latest_data_point = X[-1].reshape(1, -1)
    latest_data_point_scaled = scaler.transform(latest_data_point)

    # Predict the next day's close
    next_day_close_prediction = best_xgb_model.predict(latest_data_point_scaled)

    ndpp = next_day_close_prediction[0]

    newest_close = data.loc[0, 'close']

    def pChange(ndpp, newest_close):
        pChange = ndpp/newest_close

        if pChange > 1:
            pChange = pChange - 1
            pChange = round(pChange*100, 2)
            return pChange

        elif pChange < 100:
            pChange = pChange - 1
            pChange = round(pChange*100, 2)
            return pChange
        
    print(pChange(ndpp, newest_close))

    if -0.08 > pChange(ndpp, newest_close) > -0.33:
      return -1  
    elif 0.08 < pChange(ndpp, newest_close) < 0.33:
      return 1
    else:
      return 0
    

print(currency_safety('USD'))

usable_pairs = safe_pairs(pairs)

if len(usable_pairs) == 0:
    print("No safe pairs found. Please check the currency safety status.")
    exit()

for i in range (len(usable_pairs)):

    get_daily_data(usable_pairs[i])

    num = funnydaything_prediction() + boxstock_prediction() + techindicate_prediction()

    if num == 3:
        long.append(usable_pairs[i])
    elif num == -3:
        short.append(usable_pairs[i])



if len(long) == 0 and len(short) == 0:
    message = "Unfortunately there are no trades available today"
else:
    message = "Long Pairs: " + ", ".join(long) + "\nShort Pairs: " + ", ".join(short)

data = {
    "content": "<@708745881213599795> Hope you had a good day!\n" + message,
    "allowed_mentions": {
        "users": ["708745881213599795"]
    }
}

response = requests.post(Webhook_url, json=data)
if response.status_code == 204:
    print("Message sent successfully!")
else:
    print(f"Failed to send message: {response.text}")