import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent

flights = pd.read_csv(DATA_DIR / 'flights_cut.csv', usecols=['AIRLINE'])
airlines = pd.read_csv(DATA_DIR / 'airlines.csv')

counts = (
    flights['AIRLINE']
    .value_counts()
    .rename_axis('IATA_CODE')
    .reset_index(name='COUNT')
    .merge(airlines, on='IATA_CODE')
    .sort_values('COUNT', ascending=False)
    [['IATA_CODE', 'AIRLINE', 'COUNT']]
)

counts['SHARE'] = (counts['COUNT'] / counts['COUNT'].sum() * 100).round(1)

print(counts.to_string(index=False))
