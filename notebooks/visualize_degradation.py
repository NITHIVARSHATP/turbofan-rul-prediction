import matplotlib
matplotlib.use('TkAgg')  # Force GUI backend

import pandas as pd
import matplotlib.pyplot as plt

columns = [
    'engine_id', 'cycle',
    'setting_1', 'setting_2', 'setting_3',
    'sensor_1', 'sensor_2', 'sensor_3', 'sensor_4', 'sensor_5',
    'sensor_6', 'sensor_7', 'sensor_8', 'sensor_9', 'sensor_10',
    'sensor_11', 'sensor_12', 'sensor_13', 'sensor_14', 'sensor_15',
    'sensor_16', 'sensor_17', 'sensor_18', 'sensor_19', 'sensor_20',
    'sensor_21'
]

df = pd.read_csv('../data/raw/train_FD001.txt', sep='\s+', header=None, names=columns)
df.dropna(axis=1, inplace=True)
df['RUL'] = df.groupby('engine_id')['cycle'].transform(max) - df['cycle']

selected_sensors = ['sensor_2', 'sensor_3', 'sensor_15']
engine_ids = [1, 2, 3]

for sensor in selected_sensors:
    plt.figure(figsize=(10, 5))
    for eid in engine_ids:
        engine_data = df[df['engine_id'] == eid]
        plt.plot(engine_data['cycle'], engine_data[sensor], label=f'Engine {eid}')
    plt.title(f'{sensor} Trend Over Cycles')
    plt.xlabel('Cycle')
    plt.ylabel(f'{sensor} Reading')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{sensor}_engine_plot.png')
    plt.show(block=True)
