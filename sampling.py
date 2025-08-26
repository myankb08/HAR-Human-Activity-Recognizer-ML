import pandas as pd

# Load your CSV file into a DataFrame (done manually)
df = pd.read_csv('/Collected Data/laying_5.csv')


df["time"] = pd.to_datetime(df["time"], unit='s')  # Assuming 'time' is in seconds

# Set the 'time' column as the index
df.set_index("time", inplace=True)

# Resample to 50Hz and interpolate to fill missing values
df_resampled = df.resample('20ms').mean()  # 50Hz
df_resampled = df_resampled.interpolate(method='linear')

# If needed, reset the index to have 'time' as a regular column
df_resampled.reset_index(inplace=True)
# rewriting the original files
df_resampled.to_csv('/Collected Data/laying_5.csv')