# IITR Inter IIT Tech Meet Zeltalabs problem statement submission

This submission includes two strategies, one for BTC and one for ETH, implemented in the following Python files:  
**1)** **BTC_TSI+KAMA.py**  
**2)** **ETH_SPSAR+TAI.py**

## Strategy Implementation

To implement these strategies, we have adhered to the required format by using two functions:  
**1)** **process_data()**  
   - All indicator calculations are performed within this function.  
   - We have incorporated two additional parameters: **start_date** and **end_date**. These parameters allow you to specify the date range for testing the strategy.  
   - Data within the specified range will be part of the final DataFrame, and signals will be generated for these dates.  
   - If **start_date** is earlier than the first date in the CSV file and **end_date** is later than the last date, the entire CSV file will be included in the DataFrame, and the strategy will run for this duration.  
   - Ensure the CSV file contains at least three months of prior data to stabilize indicator values initialized at the start of calculations and account for the lookback periods of indicators.  
     Example: If you wish to test a strategy from **start_date=2023-01-01** to **end_date=2023-12-31**, the provided CSV file should include data from at least **2022-10-01** to **2023-12-31**.  

**2)** **strat()**  
   - This function contains the core strategy logic for marking positions and signals.

For detailed information about the logic used in these strategies, refer to the submission document.

## Logs  
Running these files will save logs for each strategy as separate CSV files in the working directory:  
- BTC strategy logs: **results_TSI+KAMA.csv**  
- ETH strategy logs: **results_SPSAR+TAI.csv**

## Setup Instructions  

1. Ensure all required Python modules are installed by running the following command:  
   ```bash
   pip install -r requirements.txt
2. Verify the path to the dataset required for backtesting. Provide the following files in the same working directory:
- BTC-USDT 1-day timeframe: BTC_2019_2023_1d.csv
- ETH-USDT 1-day timeframe: ETHUSDT_1d.csv

    You can also run the strategy on other datasets by providing the correct path in the code line:
    ```python 
    data=pd.read_csv('your_csv_file_path')
3. Please do ensure that you follow instructions for process_data() function to ensure no signal deviations.
4. In order to run properly ensure that the untrade sdk is installed on your testing device:
    ```bash
    git clone https://github.com/ztuntrade/untrade-sdk.git && cd untrade-sdk
    pip3 install ./untrade-sdk
5. On running these files, you will get metrics for overall as well as quarterly performance of the strategies with both 1x and 2x leverage. In case you face technical error and the SDK aborts connection, please rerun the file.
