"""
 All modules and functions required for back_test are added in requirements.txt.
"""

# Add the imports here
import uuid
import pandas as pd
from untrade.client import Client
import numpy as np
from scipy.stats import chisquare
import os
import warnings
warnings.simplefilter(action='ignore')



def process_data(data, start_date="2000-01-01", end_date="2030-01-01"):
    """
    Process the input data and return a dataframe with all the necessary indicators and data for making signals.

    Parameters:
    data (pandas.DataFrame): The input data to be processed.

    Returns:
    pandas.DataFrame: The processed dataframe with all the necessary indicators and data.
    """

    # calculate ATR 
    def ATR(data, n=14):
        data['TR'] = np.maximum(data['high'] - data['low'],
                            np.maximum(abs(data['high'] - data['close'].shift(1)),
                                        abs(data['low'] - data['close'].shift(1)))
                            )
        data[f'ATR'] = data['TR'].ewm(span=n, min_periods=n).mean()
        data['ATRdiff']=np.abs(data['ATR']-data['ATR'].shift(1))
        data.drop('TR', axis=1, inplace=True)

    # calculate HA candles and add columns with name "HA_Close" "HA_Open" "HA_High" "HA_Low"
    def HA(data):
        data['HA_close'] = (data['open'] + data['high'] + data['low'] + data['close']) / 4
        data['HA_open'] = (data['open'].shift(1) + data['close'].shift(1)) / 2
        data['HA_high'] = data[['high', 'low', 'HA_open', 'HA_close']].max(axis=1)
        data['HA_low'] = data[['high', 'low', 'HA_open', 'HA_close']].min(axis=1)

        # Calculate Heikin-Ashi wicks
        data['HA_upper_wick'] = data['HA_high'] - data['HA_close']
        data['HA_lower_wick'] = data['HA_close'] - data['HA_low']
    
    def get_trend_alignment_indicator(data):
        high = data["HA_close"]
        low = data["HA_open"]
        median_price = (high + low) / 2

        # trend_alignment_indicator's Jaw
        jaw_period = 8
        jaw_shift = 5
        jaw = median_price.rolling(window=jaw_period).mean().shift(jaw_shift)

        # trend_alignment_indicator's Teeth
        teeth_period = 5
        teeth_shift = 3
        teeth = median_price.rolling(window=teeth_period).mean().shift(teeth_shift)

        # trend_alignment_indicator's Lips
        lips_period = 3
        lips_shift = 2
        lips = median_price.rolling(window=lips_period).mean().shift(lips_shift)

        trend_alignment_indicator = []

        for i in range(len(jaw)):
            if (lips.iloc[i] > teeth.iloc[i])  and (teeth.iloc[i] > jaw.iloc[i]):
                trend_alignment_indicator.append(1)

            elif (lips.iloc[i] < teeth.iloc[i]) and (teeth.iloc[i] < jaw.iloc[i]):
                trend_alignment_indicator.append(-1)
            elif  (teeth.iloc[i] < jaw.iloc[i]):
                trend_alignment_indicator.append(-2)
            elif  (lips.iloc[i] < teeth.iloc[i]) :
                trend_alignment_indicator.append(-3)
            else:
                trend_alignment_indicator.append(0)

        data["trend_alignment_indicator"] = trend_alignment_indicator

    def ADX(data,lookback=14):
        high = data['high']
        low = data['low']
        close = data['close']
        plus_dm = high.diff()
        minus_dm = low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0

        tr1 = pd.DataFrame(high - low)
        tr2 = pd.DataFrame(abs(high - close.shift(1)))
        tr3 = pd.DataFrame(abs(low - close.shift(1)))
        frames = [tr1, tr2, tr3]
        tr = pd.concat(frames, axis = 1, join = 'inner').max(axis = 1)
        atr = tr.rolling(lookback).mean()

        plus_di = 100 * (plus_dm.ewm(alpha = 1/lookback).mean() / atr)
        minus_di = abs(100 * (minus_dm.ewm(alpha = 1/lookback).mean() / atr))
        dx = (abs(plus_di - minus_di) / abs(plus_di + minus_di)) * 100
        adx = ((dx.shift(1) * (lookback - 1)) + dx) / lookback
        adx_smooth = adx.ewm(alpha = 1/lookback).mean()
        data["ADX"] = adx_smooth

    def supertrend(data, multiplier=2):
        # Calculate basic upper and lower bands
        data['upperband'] = (data['high'] + data['low']) / 2 + (multiplier * data['ATR'])
        data['lowerband'] = (data['high'] + data['low']) / 2 - (multiplier * data['ATR'])

        # Initialize Supertrend
        data['supertrend'] = np.nan
        data['direction'] = 0  # 1 for Long, -1 for Short

        for i in range(len(data)):
            if i == 0:
                data.loc[i, 'supertrend'] = data.loc[i, 'upperband']
                continue

            # Determine Supertrend direction
            if data['close'][i] > data['upperband'][i - 1]:
                data.loc[i, 'direction'] = 1
            elif data['close'][i] < data['lowerband'][i - 1]:
                data.loc[i, 'direction'] = -1
            else:
                data.loc[i, 'direction'] = data['direction'][i - 1]

            # Update Supertrend
            if data['direction'][i] == 1:
                data.loc[i, 'supertrend'] = min(data['lowerband'][i], data['supertrend'][i - 1])
            elif data['direction'][i] == -1:
                data.loc[i, 'supertrend'] = max(data['upperband'][i], data['supertrend'][i - 1])

    def dynamic_adxr(data, base_lookback=16, atr_period=5):
        # Step 1: Calculate ATR mean
        # Rolling mean of ATR to determine the baseline volatility
        weights = np.arange(1, atr_period + 1)
        data['ATR_mean'] = data['ATR'].rolling(window=atr_period, min_periods=1).mean()
        # Step 2: Compute dynamic lookback
        # Avoid division by zero by replacing zero ATR_mean with a small number (e.g., 1e-6)
        data['ATR_mean_safe'] = data['ATR_mean'].replace(0, 1e-6)

        # Compute the dynamic lookback
        data['Dynamic_Lookback'] = (base_lookback * (1 + data['ATR'] / data['ATR_mean_safe'])).round().astype(float)

        # Handle any remaining infinite or NaN values by setting them to base_lookback
        data['Dynamic_Lookback'].replace([np.inf, -np.inf], base_lookback, inplace=True)
        data['Dynamic_Lookback'].fillna(base_lookback, inplace=True)

        # Convert Dynamic_Lookback to integer safely
        data['Dynamic_Lookback'] = data['Dynamic_Lookback'].astype(int)

        # Ensure lookback is at least 1 to avoid invalid shifts
        data['Dynamic_Lookback'] = data['Dynamic_Lookback'].clip(lower=1)

        # Step 3: Calculate ADXR_dynamic
        # Initialize ADXR_dynamic with NaN
        data['ADXR_dynamic'] = np.nan


        # Iterate over the DataFrame to compute ADXR_dynamic
        for idx in data.index:
            lookback = data.at[idx, 'Dynamic_Lookback']

            # Calculate the shifted index
            shifted_idx = idx - lookback

            if shifted_idx >= data.index.min():
                try:
                    current_adx = data.at[idx, 'ADX']
                    shifted_adx = data.at[shifted_idx, 'ADX']
                    if pd.notna(current_adx) and pd.notna(shifted_adx):
                        data.at[idx, 'ADXR_dynamic'] = 0.5*current_adx + 0.5* shifted_adx
                    else:
                        data.at[idx, 'ADXR_dynamic'] = np.nan
                except KeyError:
                    # In case shifted_idx is out of bounds
                    data.at[idx, 'ADXR_dynamic'] = np.nan 
            else:
                # Insufficient data for dynamic ADXR calculation
                data.at[idx, 'ADXR_dynamic'] = np.nan

        # Optional: Drop intermediate columns if no longer needed
        data.drop(columns=['ATR_mean', 'ATR_mean_safe', 'Dynamic_Lookback'], inplace=True)

    def PVO(data, short_period=15, long_period=30, signal_period=5):
        # Calculate short and long EMAs
        short_ema = data['volume'].ewm(span=short_period, adjust=False).mean()
        long_ema = data['volume'].ewm(span=long_period, adjust=False).mean()

        # Calculate PVO
        pvo = (short_ema - long_ema) / long_ema * 100

        # Calculate Signal Line
        signal = pvo.ewm(span=signal_period, adjust=False).mean()

        # Add PVO and Signal Line as new columns
        pvo_col_name = f"PVO_{short_period}_{long_period}"
        signal_col_name = f"Signal_{short_period}_{long_period}"
        data[pvo_col_name] = pvo
        data[signal_col_name] = signal

    def calculate_smoothened_PSAR(data, step=0.02, max_step=0.2):
        # Ensure the DataFrame has the necessary columns
        if not {'HA_high', 'HA_low', 'HA_close'}.issubset(data.columns):
            raise ValueError("DataFrame must contain 'High', 'Low', and 'Close' columns")

        high = data['HA_high'].values
        low = data['HA_low'].values
        close = data['HA_close'].values
        dates = data['datetime'].values
        dates = [pd.to_datetime(date) for date in dates]
        length = len(data)
        
        # Initialize arrays
        psar = [0] * length
        trend = [0] * length  # 1 for uptrend, -1 for downtrend
        ep = [0] * length     # Extreme point
        af = [step] * length  # Acceleration factor



        for i in range(length):
            if dates[i].month%2 == 1 and dates[i].day == 1:
                psar[i] = close[i]
                trend[i] = 0
                ep[i] = close[i]
                continue
            # Determine trend based on prior data
            if close[i] > close[i - 1]:
                trend[i] = 1    # Uptrend
                psar[i] = low[i - 1]
                ep[i] = high[i]
            else:
                trend[i] = -1   # Downtrend
                psar[i] = high[i - 1]
                ep[i] = low[i]
            af[i] = step

            # Calculate PSAR
            prior_psar = psar[i - 1]
            prior_ep = ep[i - 1]
            prior_af = af[i - 1]
            prior_trend = trend[i - 1]

            psar[i] = prior_psar + prior_af * (prior_ep - prior_psar)

            # Adjust PSAR to not penetrate extremes
            if prior_trend == 1:
                psar[i] = min(psar[i], low[i - 1], low[i - 2])
            else:
                psar[i] = max(psar[i], high[i - 1], high[i - 2])

            # Determine trend and adjust EP and AF
            if prior_trend == 1:
                if low[i] < psar[i]:
                    # Reversal to downtrend
                    trend[i] = -1
                    psar[i] = prior_ep
                    ep[i] = low[i]
                    af[i] = step
                else:
                    trend[i] = prior_trend
                    if high[i] > prior_ep:
                        ep[i] = high[i]
                        af[i] = min(prior_af + step, max_step)
                    else:
                        ep[i] = prior_ep
                        af[i] = prior_af
            else:
                if high[i] > psar[i]:
                    # Reversal to uptrend
                    trend[i] = 1
                    psar[i] = prior_ep
                    ep[i] = high[i]
                    af[i] = step
                else:
                    trend[i] = prior_trend
                    if low[i] < prior_ep:
                        ep[i] = low[i]
                        af[i] = min(prior_af + step, max_step)
                    else:
                        ep[i] = prior_ep
                        af[i] = prior_af

        # Assign the calculated PSAR values to the DataFrame
        data['smoothened_PSAR'] = psar


    ATR(data, 14)
    HA(data)
    get_trend_alignment_indicator(data)
    ADX(data, 14)
    supertrend(data)
    dynamic_adxr(data)
    PVO(data)
    calculate_smoothened_PSAR(data)

    

    # Filter the dataframe to values between start and end
    data = data[(data['datetime'] >= start_date) & (data['datetime'] <= end_date)]
    data.reset_index(drop=True, inplace=True)

    # Ensure 'datetime' column is in datetime format
    data['datetime'] = pd.to_datetime(data['datetime'])

    # Keep only the columns needed for the strat

    # Initialize Position and signals
    data['Position'] = 0
    data['signals'] = 0

    return data

# -------STRATEGY LOGIC--------#
def strat(data):
    """
    Create a strategy based on indicators or other factors.

    Parameters:
    - data: DataFrame
        The input data containing the necessary columns for strategy creation.

    Returns:
    - DataFrame
        The modified input data with an additional 'signal' column representing the strategy signals.
    """
    stoploss =0
    for i in range(1,len(data) - 1):
        curr_atr=data['ATR'][i]
        if data.loc[i, "Position"] == 1:
            if (data.loc[i, "close"] <= stoploss):
                data.loc[i+1, "Position"] = 0
            elif data.loc[i, "close"] < data.loc[i, "supertrend"] and not ( data.loc[i, "HA_close"] > data.loc[i, "HA_open"]) :
                data.loc[i+1, "Position"] = 0
            elif data.loc[i, "HA_close"] < data.loc[i, "HA_open"] and data.loc[i-1, "HA_close"] <data.loc[i-1, "HA_open"] and data['ATRdiff'][i]/data['ATR'][i]>0.25:
                data.loc[i+1, "Position"] = 0
            else:
                data.loc[i+1, "Position"] = 1
        elif data.loc[i, "Position"] == -1:
            if (data.loc[i, "close"] <= take_profit):
                data.loc[i+1, "Position"] = 0
            elif data.loc[i, "close"] > data.loc[i, "supertrend"]:
                data.loc[i+1, "Position"] = 0
            elif data.loc[i, "HA_close"] > data.loc[i, "HA_open"] and data.loc[i-1, "HA_close"] > data.loc[i-1, "HA_open"] and  data['ATRdiff'][i]/data['ATR'][i]>0.25 :
                 data.loc[i+1, "Position"] = 0
            else:
                data.loc[i+1, "Position"] = -1
                take_profit = min(take_profit, data.loc[i, "close"] -curr_atr)
        else:
            if (( data.loc[i, "HA_close"] > data.loc[i, "HA_open"] or data['HA_close'][i]>data['smoothened_PSAR'][i] ) and (data.loc[i, "trend_alignment_indicator"] == 1)) and data['PVO_15_30'][i]>1.1*data['Signal_15_30'][i]:
                data.loc[i+1, "Position"] = 1
                stoploss = data.loc[i, "close"] - curr_atr*3
            elif (data.loc[i, "trend_alignment_indicator"] <0) and (data.loc[i, "HA_close"] < data.loc[i, "HA_open"]) and data['PVO_15_30'][i]>1.1*data['Signal_15_30'][i] and data['ADXR_dynamic'][i]<25 and data['ADXR_dynamic'][i]>15:
                data.loc[i+1, "Position"] = -1
                take_profit = data.loc[i, "close"] - 1.5* curr_atr 

    def post_process(data):   # convert Position column into signals and trade_type
        for i in range(1, len(data)):
            # Handle signal generation
            if (data.loc[i, 'Position'] != data.loc[i-1, 'Position']):
                if data.loc[i, 'Position'] == 0:
                    if data.loc[i-1, 'Position'] == 1:
                        data.loc[i-1, 'signals'] = -1
                    else:
                        data.loc[i-1, 'signals'] = 1

                if data.loc[i, 'Position'] == 1:
                    if data.loc[i-1, 'Position'] == 0:
                        data.loc[i-1, 'signals'] = 1
                    else:
                        data.loc[i-1, 'signals'] = 2

                elif data.loc[i, 'Position'] == -1:
                    if data.loc[i-1, 'Position'] == 0:
                        data.loc[i-1, 'signals'] = -1
                    else:
                        data.loc[i-1, 'signals'] = -2

            else:
                data.loc[i-1, 'signals'] = 0

        # Create trade_type column in dataframe
        data['trade_type'] = "hold"
        in_trade = False
        for i in range(1, len(data)):
            if data.loc[i, 'signals'] != 0:
                if data.loc[i, 'signals'] in [2, -2]:
                    if data.loc[i, 'signals'] == 2:
                        data.loc[i, 'trade_type'] = "short_reversal"
                    else:
                        data.loc[i, 'trade_type'] = "long_reversal"
                else:
                    if in_trade:
                        data.loc[i, 'trade_type'] = "square_off"
                        in_trade = False
                    else:
                        if data.loc[i, 'signals'] == 1:
                            data.loc[i, 'trade_type'] = "long"
                            in_trade = True
                        else:
                            data.loc[i, 'trade_type'] = "short"
                            in_trade = True

        
        return data
    
    data = post_process(data)

    return data


def perform_backtest(csv_file_path, leverage = 1):
    client = Client()
    result = client.backtest(
        jupyter_id="team62_zelta_hpps",  # the one you use to login to jupyter.untrade.io
        file_path=csv_file_path,
        leverage=leverage,  # Adjust leverage as needed
    )
    return result

# Following function can be used for every size of file, specially for large files(time consuming,depends on upload speed and file size)


def perform_backtest_large_csv(csv_file_path):
    client = Client()
    file_id = str(uuid.uuid4())
    chunk_size = 90 * 1024 * 1024
    total_size = os.path.getsize(csv_file_path)
    total_chunks = (total_size + chunk_size - 1) // chunk_size
    chunk_number = 0
    if total_size <= chunk_size:
        total_chunks = 1
        # Normal Backtest
        result = client.backtest(
            file_path=csv_file_path,
            leverage=1,
            jupyter_id="team62_zelta_hpps",
            # result_type="Q",
        )
        for value in result:
            print(value)

        return result

    with open(csv_file_path, "rb") as f:
        while True:
            chunk_data = f.read(chunk_size)
            if not chunk_data:
                break
            chunk_file_path = f"/tmp/{file_id}_chunk_{chunk_number}.csv"
            with open(chunk_file_path, "wb") as chunk_file:
                chunk_file.write(chunk_data)

            # Large CSV Backtest
            result = client.backtest(
                file_path=chunk_file_path,
                leverage=1,
                jupyter_id="team62_zelta_hpps",
                file_id=file_id,
                chunk_number=chunk_number,
                total_chunks=total_chunks,
                # result_type="Q",
            )

            for value in result:
                print(value)

            os.remove(chunk_file_path)

            chunk_number += 1

    return result

def perform_backtest_quarterly(csv_file_path="output.csv", leverage=1):
    # Create an instance of the untrade client
    client = Client()

    # Perform backtest using the provided CSV file path
    result = client.backtest(
        jupyter_id="team62_zelta_hpps",  # the one you use to login to jupyter.untrade.io
        file_path=csv_file_path,
        leverage=leverage,  # Adjust leverage as needed
        result_type = "Q"
    )

    for value in result:
        print(value)


def main():
    data = pd.read_csv("ETHUSDT_1d.csv")

    processed_data = process_data(data, start_date="2018-01-01", end_date="2024-12-31") 
    '''Add start and end date as per requirement. All dates within the start and the end date will be a part of the final dataframe and signals will be generated for these dates.
    If start_date is before the first date in csv and end_date is greater than last date in csv, complete csv will be a part of dataframe and strategy will run for this duration.
    Please put csv file with atleast 3 months of prior data than the dates on which you want to test the strategy.
    For e.g. if you want to test a strategy from start_date=2023-01-01 to end_date=2023-12-31 then, the csv you provide should contain data from atleast 2022-10-01 to 2023-12-31.
    This helps us stabilise indicator values that are initialised based on starting point.'''

    result_data = strat(processed_data)

    csv_file_path = "results_SPSAR+TAI.csv"

    result_data.to_csv(csv_file_path, index=False)


    # Results for 1x leverage quarterly
    backtest_result = perform_backtest(csv_file_path)
    print("Results for 1x leverage")
    for value in backtest_result:
        print(value)

    # Results for 1x leverage quarterly
    perform_backtest_quarterly(csv_file_path, leverage=1)


    # Results for 2x leverage overall
    backtest_result = perform_backtest(csv_file_path, leverage = 2)
    print("Results for 2x leverage")
    for value in backtest_result:
        print(value)

    # Results for 2x leverage quarterly
    perform_backtest_quarterly(csv_file_path, leverage=2)
    


if __name__ == "__main__":
    main()
