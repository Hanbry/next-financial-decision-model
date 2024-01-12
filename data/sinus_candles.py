import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mplfinance.original_flavor import candlestick_ohlc

num_candles = 4000

start_time = datetime.datetime.now(datetime.timezone.utc)  # Setze die Zeitzone auf UTC

t = np.arange(num_candles)
sin_data = (np.sin(t / 10.0) + 1) * 50

data = []

for i in range(num_candles):
    time = start_time + datetime.timedelta(minutes=i)
    open_price = sin_data[i]
    close_price = sin_data[i]
    high_price = max(sin_data[i], sin_data[i - 1])
    low_price = min(sin_data[i], sin_data[i - 1])
    volume = np.random.randint(1000, 5000)
    time_utc = int(time.timestamp())  # Konvertiere das Datum in einen Unix-Zeitstempel
    data.append([open_price, close_price, high_price, low_price, volume, time_utc])

df = pd.DataFrame(data, columns=["open", "close", "high", "low", "volume", "time"])

df.to_csv("sinus_minute_candlestick_data.csv", index=False)

# Für die Visualisierung müssen wir das Datum zurück in ein datetime-Objekt konvertieren
df['time'] = pd.to_datetime(df['time'], unit='s')

df = df.sort_values(by='time')

df_last_200_minutes = df.iloc[-200:]

df_last_200_minutes['time_num'] = mdates.date2num(df_last_200_minutes['time'])

fig, ax = plt.subplots(figsize=(12, 6))

candlestick_ohlc(ax, df_last_200_minutes[['time_num', 'open', 'high', 'low', 'close']].values, width=0.0005, colorup='g', colordown='r')

ax.xaxis_date()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
plt.xticks(rotation=45)
plt.xlabel('Zeit')
plt.ylabel('Preis')
plt.title('Candlestick-Chart (Last 200 Minutes)')

plt.show()
