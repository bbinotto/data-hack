"""Compare distributions of AIRLINE, ORIGIN_AIRPORT, DESTINATION_AIRPORT and MONTH
across flights_cut, flights_train, flights_val and flights_test."""

import pandas as pd

DTYPES = {
    "MONTH": "int8",
    "DAY": "int8",
    "DAY_OF_WEEK": "int8",
    "FLIGHT_NUMBER": "int32",
    "DEPARTURE_DELAY": "float32",
    "ARRIVAL_DELAY": "float32",
    "SCHEDULED_DEPARTURE": "string",
    "SCHEDULED_TIME": "float32",
    "SCHEDULED_ARRIVAL": "string",
    "ELAPSED_TIME": "float32",
    "DISTANCE": "int32",
}

DATASETS = {
    "cut": "flights_cut.csv",
    "train": "flights_train.csv",
    "val": "flights_val.csv",
    "test": "flights_test.csv",
}

COMPARE_COLS = ["AIRLINE", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "MONTH"]


def load_datasets():
    return {name: pd.read_csv(path, dtype=DTYPES) for name, path in DATASETS.items()}


def compare_column(dfs, column, top_n=15):
    dists = {name: df[column].value_counts(normalize=True) for name, df in dfs.items()}
    table = pd.DataFrame(dists).fillna(0.0).sort_values("cut", ascending=False)
    print(f"\n=== {column} (proportion of rows, top {top_n} by 'cut') ===")
    print(table.head(top_n).round(4))
    print(f"\n{column} max abs deviation from 'cut' across datasets:")
    for name in dfs:
        if name == "cut":
            continue
        diff = (table[name] - table["cut"]).abs().max()
        print(f"  cut vs {name}: {diff:.4f}")


def main():
    dfs = load_datasets()

    print("Row counts:")
    for name, df in dfs.items():
        print(f"  {name}: {len(df):,}")

    for col in COMPARE_COLS:
        compare_column(dfs, col)


if __name__ == "__main__":
    main()
