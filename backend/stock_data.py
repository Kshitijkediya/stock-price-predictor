import os
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime, timedelta
from functools import lru_cache
import shap
import plotly.graph_objects as go

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

@lru_cache(maxsize=10)
def get_stock_data(ticker):
    """
    Fetches stock data, handles MultiIndex columns, and includes timeout.
    """
    print(f"Fetching data for {ticker} with increased timeout (60s)...")
    try:
        data_multi = yf.download(
            ticker,
            period='1y',
            interval='1d',
            timeout=60
        )

        if data_multi.empty:
            raise ValueError(f"No data returned for ticker {ticker} after download attempt.")

        if isinstance(data_multi.columns, pd.MultiIndex):
            print("Detected MultiIndex columns, flattening...")
            required_tuples = [('Open', ticker), ('High', ticker), ('Low', ticker), ('Close', ticker), ('Volume', ticker)]
            if not all(col_tuple in data_multi.columns for col_tuple in required_tuples):
                 missing_cols = [str(t) for t in required_tuples if t not in data_multi.columns]
                 raise ValueError(f"Standard OHLCV columns missing in fetched MultiIndex data for {ticker}. Missing: {missing_cols}. Found: {data_multi.columns.tolist()}")

            data = pd.DataFrame({
                'Open': data_multi[('Open', ticker)],
                'High': data_multi[('High', ticker)],
                'Low': data_multi[('Low', ticker)],
                'Close': data_multi[('Close', ticker)],
                'Volume': data_multi[('Volume', ticker)]
            })
        else:
             print("Detected flat columns.")
             required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
             if not all(col in data_multi.columns for col in required_cols):
                 missing_cols = [c for c in required_cols if c not in data_multi.columns]
                 raise ValueError(f"Standard OHLCV columns missing in fetched flat data for {ticker}. Missing: {missing_cols}. Found: {data_multi.columns.tolist()}")
             data = data_multi[required_cols].copy()


        data.reset_index(inplace=True)
        if 'Date' in data.columns:
             data['Date'] = pd.to_datetime(data['Date'])
        else:
             raise ValueError("Date column not found after processing index.")

        print(f"Data processed for {ticker}. Shape: {data.shape}")
        return data

    except Exception as e:
        print(f"Error during yfinance download or processing for {ticker}: {type(e).__name__} - {e}")
        import traceback
        traceback.print_exc()
        raise ValueError(f"Failed to download or process data for {ticker}. Check network or ticker validity. Error: {type(e).__name__}") from e

def prepare_data(data):
    if data.empty or 'Close' not in data.columns:
        raise ValueError("Data is empty or missing 'Close' column for LSTM preparation.")

    data = data.dropna(subset=['Close'])
    if data.empty:
        raise ValueError("Data became empty after dropping NaNs in 'Close' column.")

    scaler = MinMaxScaler(feature_range=(0, 1))
    close_values = data[['Close']].values
    if len(close_values) == 0:
         raise ValueError("No 'Close' values available to scale.")
    scaled_data = scaler.fit_transform(close_values)

    prediction_days = 60

    if len(scaled_data) <= prediction_days:
        raise ValueError(f"Not enough data ({len(scaled_data)} points) to create training sequences with lookback of {prediction_days} days.")

    x_train = []
    y_train = []

    for x in range(prediction_days, len(scaled_data)):
        x_train.append(scaled_data[x - prediction_days:x, 0])
        y_train.append(scaled_data[x, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    if x_train.size == 0:
        raise ValueError("x_train array is empty before reshaping.")
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    return x_train, y_train, scaler

def build_lstm_model(x_train, y_train, ticker):
    model_filename = os.path.join(MODELS_DIR, f'{ticker}_lstm.keras')
    if os.path.exists(model_filename):
        model = load_model(model_filename)
    else:
        model = Sequential()
        if x_train.shape[1] == 0:
             raise ValueError("Cannot build LSTM model with sequence length 0.")
        model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(LSTM(units=50))
        model.add(Dense(units=1))

        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(x_train, y_train, epochs=5, batch_size=32, verbose=0) # Consider increasing epochs
        model.save(model_filename)
    return model

def predict_next_day_lstm(model, data, scaler):
    prediction_days = 60
    if len(data) < prediction_days:
        raise ValueError(f"Not enough data ({len(data)} points) for LSTM prediction sequence of {prediction_days} days.")

    # Prepare the actual input for the prediction
    real_data_for_pred = data.iloc[-prediction_days:].values
    if real_data_for_pred.shape[0] != prediction_days or real_data_for_pred.shape[1] != 1:
         raise ValueError(f"Incorrect shape {real_data_for_pred.shape} for scaling prediction input. Expected ({prediction_days}, 1).")

    scaled_real_data_for_pred = scaler.transform(real_data_for_pred)
    input_data = np.reshape(scaled_real_data_for_pred, (1, prediction_days, 1))

    # Make the prediction
    prediction = model.predict(input_data)
    prediction_inversed = scaler.inverse_transform(prediction)

    return prediction_inversed[0][0], input_data


def arima_prediction_func(close_data_series): 
    """
    Fits an ARIMA model and forecasts the next step.
    Expects a pandas Series of closing prices.
    """
    if not isinstance(close_data_series, pd.Series):
        raise TypeError("Input must be a pandas Series.")

    # Check if the Series is empty after dropping NaNs
    cleaned_series = close_data_series.dropna()
    if cleaned_series.empty:
        raise ValueError("Data Series is empty after dropping NaNs for ARIMA prediction.")

    if len(cleaned_series) < 10:
         raise ValueError(f"Not enough non-NaN data points ({len(cleaned_series)}) for ARIMA model.")

    try:
        model = ARIMA(cleaned_series, order=(5, 1, 0)) # Standard ARIMA order
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=1)
        return forecast.iloc[0]
    except Exception as e:
        print(f"Error fitting ARIMA(5,1,0) model: {e}")
        try:
            print("Trying simpler ARIMA(1,1,0) order...")
            model = ARIMA(cleaned_series, order=(1, 1, 0))
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=1)
            return forecast.iloc[0]
        except Exception as e2:
            print(f"Error fitting simpler ARIMA model: {e2}")
            raise ValueError(f"ARIMA model failed to converge. Error: {e2}") from e2


def calculate_technical_indicators(data):
    required_cols = ['Open', 'High', 'Low', 'Close', 'Date']
    if not isinstance(data, pd.DataFrame) or not all(col in data.columns for col in required_cols):
        missing = [col for col in required_cols if col not in data.columns]
        raise ValueError(f"Input data must be a pandas DataFrame with standard OHLC columns. Missing: {missing}")

    print(f"Shape before dropping OHLC NaNs: {data.shape}")
    data = data.dropna(subset=['Open', 'High', 'Low', 'Close']).copy()
    print(f"Shape after dropping OHLC NaNs: {data.shape}")

    if data.empty:
         print("Warning: Data became empty after dropping initial OHLC NaNs.")
         return data 

    for col in ['Open', 'High', 'Low', 'Close']:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    data = data.dropna(subset=['Open', 'High', 'Low', 'Close']).copy()
    if data.empty:
         print("Warning: Data became empty after ensuring OHLC columns are numeric.")
         return data

    # Moving Average
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['MA50'] = data['Close'].rolling(window=50).mean()
    # Bollinger Bands
    data['STD20'] = data['Close'].rolling(window=20).std()
    data['UpperBand'] = data['MA20'] + (data['STD20'] * 2)
    data['LowerBand'] = data['MA20'] - (data['STD20'] * 2)
    # RSI
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.ewm(com=13, adjust=False).mean()
    avg_loss = loss.ewm(com=13, adjust=False).mean()
    rs = np.where(avg_loss == 0, np.inf, avg_gain / avg_loss) # Avoid division by zero
    data['RSI'] = 100 - (100 / (1 + rs))
    data['RSI'] = data['RSI'].replace([np.inf, -np.inf], 100) 
    data['RSI'] = data['RSI'].fillna(50) 

    # MACD
    data['EMA12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['EMA26'] = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = data['EMA12'] - data['EMA26']
    data['Signal Line'] = data['MACD'].ewm(span=9, adjust=False).mean()

    # Heikin-Ashi 
    try:
        ha_data = pd.DataFrame(index=data.index)
        ha_data['HA_Close'] = (data['Open'] + data['High'] + data['Low'] + data['Close']) / 4
        ha_data['HA_Open'] = np.nan
        if not data.empty:
             ha_data.loc[ha_data.index[0], 'HA_Open'] = (data.iloc[0]['Open'] + data.iloc[0]['Close']) / 2
             for i in range(1, len(data)):
                  prev_ha_open = pd.to_numeric(ha_data.iloc[i-1]['HA_Open'], errors='coerce')
                  prev_ha_close = pd.to_numeric(ha_data.iloc[i-1]['HA_Close'], errors='coerce')
                  if not pd.isna(prev_ha_open) and not pd.isna(prev_ha_close):
                       ha_data.loc[ha_data.index[i], 'HA_Open'] = (prev_ha_open + prev_ha_close) / 2
                  else:
                       ha_data.loc[ha_data.index[i], 'HA_Open'] = np.nan
        ha_data['HA_High'] = data['High'].combine(ha_data['HA_Open'], max).combine(ha_data['HA_Close'], max)
        ha_data['HA_Low'] = data['Low'].combine(ha_data['HA_Open'], min).combine(ha_data['HA_Close'], min)
        data[['HA_Open', 'HA_High', 'HA_Low', 'HA_Close']] = ha_data[['HA_Open', 'HA_High', 'HA_Low', 'HA_Close']]
    except Exception as ha_e:
        print(f"Warning: Could not calculate Heikin-Ashi indicators: {ha_e}")
        for col in ['HA_Open', 'HA_High', 'HA_Low', 'HA_Close']: data[col] = np.nan

    # Fibonacci
    high_numeric = pd.to_numeric(data['High'], errors='coerce').dropna()
    low_numeric = pd.to_numeric(data['Low'], errors='coerce').dropna()
    fib_cols = ['Fib_0.0', 'Fib_23.6', 'Fib_38.2', 'Fib_50.0', 'Fib_61.8', 'Fib_100.0']
    if not high_numeric.empty and not low_numeric.empty:
         max_price = high_numeric.max()
         min_price = low_numeric.min()
         diff = max_price - min_price
         if diff > 0:
             for p, col in zip([0.0, 0.236, 0.382, 0.5, 0.618, 1.0], fib_cols):
                 data[col] = max_price - p * diff
         else:
             for col in fib_cols: data[col] = max_price
    else:
         for col in fib_cols: data[col] = np.nan

    data['Close_Lag1'] = data['Close'].shift(1)

    return data

def get_lstm_shap_explanation(model, background_data, input_instance):

    
    print("--- Generating SHAP Explanations for LSTM ---")
    try:

        background_sample_size = min(100, background_data.shape[0])
        background_sample_indices = np.random.choice(background_data.shape[0], background_sample_size, replace=False)
        background_sample = background_data[background_sample_indices]

        # Using KernelExplainer for model-agnostic approach, suitable for complex models like LSTM

        def predict_wrapper(data_batch_reshaped):
            num_samples = data_batch_reshaped.shape[0]
            sequence_length = model.input_shape[1]
            data_batch_lstm_shape = data_batch_reshaped.reshape(num_samples, sequence_length, 1)
            return model.predict(data_batch_lstm_shape, verbose=0)

        background_flat = background_sample.reshape(background_sample.shape[0], -1)
        instance_flat = input_instance.reshape(1, -1)

        explainer = shap.KernelExplainer(predict_wrapper, background_flat)

        shap_values = explainer.shap_values(instance_flat, nsamples=50)

        
        shap_values_array = shap_values[0] if isinstance(shap_values, list) else shap_values
        abs_shap = np.abs(shap_values_array.flatten()) 
        sequence_length = input_instance.shape[1]
        feature_names = [f"Day -{sequence_length - i}" for i in range(sequence_length)]

       
        feature_importance = sorted(zip(feature_names, abs_shap), key=lambda x: x[1], reverse=True)


        print("--- SHAP Explanation Generated ---")
        return {
            'feature_importance': feature_importance[:10] # Return top 10 important features
        }

    except ImportError:
        print("SHAP library not found. Please install it: pip install shap")
        return {"error": "SHAP library not installed."}
    except Exception as e:
        print(f"--- Error generating SHAP values: {type(e).__name__}: {e} ---")
        import traceback
        traceback.print_exc()
        return {"error": f"Could not generate SHAP explanation: {type(e).__name__}"}


def generate_prediction_chart(data, predictions, ticker):
    try:
        display_data = data.tail(90)
        dates = [d.strftime('%Y-%m-%d') if hasattr(d, 'strftime') else str(d) for d in display_data['Date'].tolist()]
        closes = [float(c) for c in display_data['Close'].tolist()]

        last_date = data['Date'].iloc[-1]
        next_date = (last_date + pd.Timedelta(days=1)).strftime('%Y-%m-%d')

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=dates, y=closes,
            mode='lines', name='Historical Close',
            line=dict(color='#00ff00', width=2)
        ))

        if 'LSTM' in predictions and not isinstance(predictions['LSTM'], str):
            fig.add_trace(go.Scatter(
                x=[next_date], y=[float(predictions['LSTM'])],
                mode='markers+text', name='LSTM Prediction',
                marker=dict(color='cyan', size=14, symbol='star'),
                text=['LSTM'], textposition='top center',
                textfont=dict(color='cyan')
            ))

        if 'ARIMA' in predictions and not isinstance(predictions['ARIMA'], str):
            fig.add_trace(go.Scatter(
                x=[next_date], y=[float(predictions['ARIMA'])],
                mode='markers+text', name='ARIMA Prediction',
                marker=dict(color='orange', size=14, symbol='diamond'),
                text=['ARIMA'], textposition='bottom center',
                textfont=dict(color='orange')
            ))

        fig.update_layout(
            title=dict(text=f'{ticker} — Last 90 Days & Next Day Prediction', font=dict(color='#00ff00')),
            xaxis_title='Date', yaxis_title='Price',
            plot_bgcolor='#0a0a0a', paper_bgcolor='#000000',
            font=dict(color='#00ff00', family='Courier New'),
            xaxis=dict(gridcolor='#1a3300', zerolinecolor='#1a3300'),
            yaxis=dict(gridcolor='#1a3300', zerolinecolor='#1a3300'),
            legend=dict(bgcolor='#0a0a0a', bordercolor='#00ff00', borderwidth=1),
            margin=dict(l=60, r=40, t=60, b=60)
        )

        return fig.to_html(full_html=False, include_plotlyjs='cdn')
    except Exception as e:
        print(f"Error generating chart: {type(e).__name__}: {e}")
        return None


def make_prediction(ticker, model_choice):
    print(f"--- Running prediction for {ticker} ---")
    try:
        raw_data = get_stock_data(ticker)
        if raw_data.empty:
             return {'error': f"No data retrieved for ticker {ticker}. It might be invalid."}, None

        data = calculate_technical_indicators(raw_data) # Use data with indicators

        min_required_data = 100 

        essential_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'MA20', 'MA50', 'RSI'] 
        cols_to_check = [col for col in essential_cols if col in data.columns]
        core_ohlc = ['Open', 'High', 'Low', 'Close']
        if not all(col in cols_to_check for col in core_ohlc):
             missing_core = [c for c in core_ohlc if c not in cols_to_check]
             return {'error': f"Core OHLC data ({missing_core}) missing after indicator calculation for {ticker}."}, None

        data_cleaned = data.dropna(subset=cols_to_check).copy()

        if len(data_cleaned) < min_required_data:
            print(f"Warning: Data points after cleaning ({len(data_cleaned)}) < {min_required_data}. Models might be less accurate.")
            if len(data_cleaned) < 65:
                 return {'error': f"Insufficient valid data points ({len(data_cleaned)}) after cleaning for {ticker}."}, None

        data = data_cleaned
        if data.empty:
            return {'error': f"No valid data remaining after cleaning for {ticker}."}, None

        predictions = {}
        explanations = {} 

        if model_choice in ['LSTM', 'Both']:
            print("--- Processing LSTM ---")
            lstm_explanation_data = None 
            try:
                lstm_data_input_df = data[['Close']].copy() 
                x_train, y_train, scaler = prepare_data(lstm_data_input_df)
                lstm_model = build_lstm_model(x_train, y_train, ticker)

                
                lstm_prediction, lstm_input_instance = predict_next_day_lstm(lstm_model, lstm_data_input_df, scaler)
                predictions['LSTM'] = lstm_prediction

                lstm_explanation_data = get_lstm_shap_explanation(lstm_model, x_train, lstm_input_instance)
                explanations['LSTM'] = lstm_explanation_data # Store explanation dict

            except Exception as lstm_e:
                 print(f"--- Error within LSTM block: {type(lstm_e).__name__}: {lstm_e} ---")
                 import traceback
                 traceback.print_exc()
                 predictions['LSTM'] = f"LSTM Error: {lstm_e}"
                 explanations['LSTM'] = {"error": f"LSTM prediction failed: {lstm_e}"}


        if model_choice in ['ARIMA', 'Both']:
             print("--- Processing ARIMA ---")
             try:
                arima_prediction = arima_prediction_func(data['Close']) 
                predictions['ARIMA'] = arima_prediction
                explanations['ARIMA'] = {"info": "ARIMA model explanation focuses on Autoregressive (AR) and Moving Average (MA) components based on past values. Feature importance plots like SHAP are less standard for ARIMA."}
             except Exception as arima_e:
                 print(f"--- Error within ARIMA block: {type(arima_e).__name__}: {arima_e} ---")
                 import traceback
                 traceback.print_exc()
                 predictions['ARIMA'] = f"ARIMA Error: {arima_e}"
                 explanations['ARIMA'] = {"error": f"ARIMA prediction failed: {arima_e}"}

        predictions['Current Price'] = data['Close'].iloc[-1]
        predictions['Last Date'] = data['Date'].iloc[-1].strftime('%Y-%m-%d')
        predictions['chart_html'] = generate_prediction_chart(data, predictions, ticker)
        print(f"--- Prediction successful for {ticker} ---")
        return predictions, explanations

    except Exception as e:
        print(f"--- Error in make_prediction (outer catch) for {ticker}: {type(e).__name__}: {e} ---")
        import traceback
        traceback.print_exc()
        return {'error': f"An error occurred processing ticker {ticker}. Details: {type(e).__name__}: {e}"}, {}


if __name__ == "__main__":
     pass 