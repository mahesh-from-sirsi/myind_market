import os
import requests
import zipfile
import io
import pandas as pd
from datetime import datetime, timedelta
import joblib
from sklearn.preprocessing import LabelEncoder

# Paths
nse_data_dir = "nse_data"
latest_features_path = os.path.join(nse_data_dir, "latest_intraday_features.csv")

# Target symbols
symbols = ['NIFTY', 'BANKNIFTY', 'RELIANCE', 'HDFCBANK', 'ICICIBANK', 'INFY', 'SBIN', 'TCS', 'LT', 'AXISBANK', 'ITC', 'KOTAKBANK']

# Define function to get URLs for equity and F&O bhavcopy
def get_bhavcopy_url(date):
    day = date.strftime("%d")
    month = date.strftime("%b").upper()
    year = date.strftime("%Y")

    # URLs for equity and F&O bhavcopy
    equity_url = f"https://www1.nseindia.com/content/historical/EQUITIES/{year}/{month}/cm{day}{month}{year}bhav.csv.zip"
    fno_url = f"https://www1.nseindia.com/content/historical/DERIVATIVES/{year}/{month}/fo{day}{month}{year}bhav.csv.zip"
    return equity_url, fno_url

# Helper function to download and extract data
def download_and_extract(url, save_dir):
    headers = {
        "User-Agent": "Mozilla/5.0"
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                z.extractall(save_dir)
            return True
        else:
            return False
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return False

# Download and prepare the data for today
def prepare_data():
    # Define today's date
    end_date = datetime.today()

    eq_url, fno_url = get_bhavcopy_url(end_date)

    # Download and extract data
    eq_success = download_and_extract(eq_url, "nse_data/equity")
    fno_success = download_and_extract(fno_url, "nse_data/fno")

    if not eq_success and not fno_success:
        print(f"No data for: {end_date.strftime('%d-%m-%Y')}")
        return pd.DataFrame()

    # Process equity data
    all_eq = pd.DataFrame()
    equity_files = os.listdir("nse_data/equity")
    for file in equity_files:
        if file.endswith(".csv"):
            df = pd.read_csv(f"nse_data/equity/{file}")
            df = df[df['SYMBOL'].isin(symbols)]
            df['DATE'] = pd.to_datetime(file[2:11], format="%d%b%Y")
            all_eq = pd.concat([all_eq, df], ignore_index=True)

    # Process F&O data
    all_fno = pd.DataFrame()
    fno_files = os.listdir("nse_data/fno")
    for file in fno_files:
        if file.endswith(".csv"):
            df = pd.read_csv(f"nse_data/fno/{file}")
            df = df[df['SYMBOL'].isin(symbols)]
            df['DATE'] = pd.to_datetime(file[2:11], format="%d%b%Y")
            all_fno = pd.concat([all_fno, df], ignore_index=True)

    # Filter and aggregate F&O data (Open Interest calculations)
    fno_filtered = all_fno[all_fno['INSTRUMENT'].isin(['OPTIDX', 'OPTSTK'])]
    call_oi = fno_filtered[fno_filtered['OPTION_TYP'] == 'CE'].groupby(['SYMBOL', 'DATE'])['OPEN_INT'].sum().reset_index(name='CALL_OI')
    put_oi = fno_filtered[fno_filtered['OPTION_TYP'] == 'PE'].groupby(['SYMBOL', 'DATE'])['OPEN_INT'].sum().reset_index(name='PUT_OI')

    # Merge OI data
    oi_data = pd.merge(call_oi, put_oi, on=['SYMBOL', 'DATE'], how='outer').fillna(0)
    oi_data['PCR'] = oi_data['PUT_OI'] / oi_data['CALL_OI'].replace(0, 1)

    # Process equity data for VWAP and other features
    all_eq['VWAP'] = (all_eq['TOTTRDVAL'] / all_eq['TOTTRDQTY']).round(2)
    equity_features = all_eq[['SYMBOL', 'DATE', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VWAP', 'TOTTRDQTY']].copy()
    equity_features = equity_features.sort_values(['SYMBOL', 'DATE'])

    # Add next day open for gap labeling
    equity_features['NEXT_OPEN'] = equity_features.groupby('SYMBOL')['OPEN'].shift(-1)
    equity_features['GAP'] = equity_features['NEXT_OPEN'] - equity_features['CLOSE']
    equity_features['GAP_PCT'] = (equity_features['GAP'] / equity_features['CLOSE'] * 100).round(2)
    equity_features['GAP_LABEL'] = equity_features['GAP_PCT'].apply(lambda x: 'GAP_UP' if x > 0.3 else ('GAP_DOWN' if x < -0.3 else 'FLAT'))

    # Merge with OI data
    final_data = pd.merge(equity_features, oi_data, on=['SYMBOL', 'DATE'], how='left')
    final_data = final_data.dropna()

    # Save the data for later predictions
    final_data.to_csv(latest_features_path, index=False)
    print(f"✅ Latest features data saved to {latest_features_path}")

# Run the data preparation
prepare_data()
