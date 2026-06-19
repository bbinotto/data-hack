# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

ML project analyzing 2015 US domestic flight data to predict arrival/departure delays.

## Data

| File | Rows | Key columns |
|------|------|-------------|
| `flights.csv` | ~5.8M | YEAR, MONTH, DAY, DAY_OF_WEEK, AIRLINE, ORIGIN_AIRPORT, DESTINATION_AIRPORT, SCHEDULED_DEPARTURE, DEPARTURE_DELAY, ARRIVAL_DELAY, CANCELLED, CANCELLATION_REASON, DISTANCE, AIR_SYSTEM_DELAY, SECURITY_DELAY, AIRLINE_DELAY, LATE_AIRCRAFT_DELAY, WEATHER_DELAY |
| `airports.csv` | 322 | IATA_CODE, AIRPORT, CITY, STATE, COUNTRY, LATITUDE, LONGITUDE |
| `airlines.csv` | 14 | IATA_CODE, AIRLINE |

## Critical: loading flights.csv

At 592 MB the file needs dtype optimization — always load with explicit dtypes:

```python
FLIGHT_DTYPES = {
    'YEAR': 'int16', 'MONTH': 'int8', 'DAY': 'int8', 'DAY_OF_WEEK': 'int8',
    'FLIGHT_NUMBER': 'int32', 'DEPARTURE_DELAY': 'float32', 'ARRIVAL_DELAY': 'float32',
    'SCHEDULED_DEPARTURE': 'int32', 'SCHEDULED_TIME': 'float32', 'ELAPSED_TIME': 'float32',
    'AIR_TIME': 'float32', 'DISTANCE': 'int32', 'CANCELLED': 'int8', 'DIVERTED': 'int8',
}
df = pd.read_csv('flights.csv', dtype=FLIGHT_DTYPES)
```

## ML target

- **Regression**: predict `ARRIVAL_DELAY` (minutes)
- **Classification**: binary late/on-time (threshold 15 min)
- Drop `CANCELLED == 1` rows before modeling
- Delay breakdown columns (`AIRLINE_DELAY`, `WEATHER_DELAY`, etc.) are leakage — exclude from features

## Dependencies

```
pip install -r requirements.txt
```

## Notebooks

Primary notebook: `notebooks/01_flight_analysis.ipynb`

Structure: imports → data loading → EDA → feature engineering → modelling → evaluation
