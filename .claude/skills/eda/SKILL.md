---
name: eda
description: Run exploratory data analysis on the flight dataset. Use when the user wants to understand the data, find patterns, or prepare for modelling.
---

Perform EDA on the 2015 US flight dataset. Work inside `notebooks/01_flight_analysis.ipynb` unless the user specifies otherwise.

Steps:
1. Load flights.csv using the FLIGHT_DTYPES dict from CLAUDE.md. Merge with airlines.csv and airports.csv on IATA codes.
2. **Data quality**: count nulls per column, show % missing, identify columns with >30% missing (delay breakdown columns are sparse by design — only populated when that delay type occurred).
3. **Target distribution**: plot histogram of ARRIVAL_DELAY; note skew; show % of flights delayed >15 min, >60 min; show % cancelled.
4. **Time patterns**: average delay by month, day of week, hour of day (derive from SCHEDULED_DEPARTURE as int HHMM).
5. **Airline comparison**: mean/median ARRIVAL_DELAY per airline (join IATA_CODE → AIRLINE name); bar chart sorted by delay.
6. **Airport comparison**: top 20 busiest origin airports by flight count; overlay mean delay.
7. **Distance vs delay**: scatter plot DISTANCE vs ARRIVAL_DELAY (sample 50k rows); correlation.
8. **Delay cause breakdown**: for delayed flights only, pie chart of mean AIR_SYSTEM_DELAY, SECURITY_DELAY, AIRLINE_DELAY, LATE_AIRCRAFT_DELAY, WEATHER_DELAY.

Print a 5-bullet summary of the most important findings at the end.
