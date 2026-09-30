import pandas as pd

RAW = "data/raw/compas-scores-two-years.csv"
OUT = "data/processed/compas_clean.csv"

def clean():
    df = pd.read_csv(RAW)
    print("Raw rows:", len(df))

    df = df[
        # screening within 30 days of arrest, so the score belongs to this arrest
        (df["days_b_screening_arrest"] <= 30)
        & (df["days_b_screening_arrest"] >= -30)
        # drop rows where recidivism data could not be found
        & (df["is_recid"] != -1)
        # drop ordinary traffic offences
        & (df["c_charge_degree"] != "O")
        # drop rows with no COMPAS score text
        & (df["score_text"] != "N/A")
    ]

    cols = ["age", "sex", "race", "priors_count",
            "c_charge_degree", "decile_score", "two_year_recid"]
    df = df[cols].reset_index(drop=True)

    df.to_csv(OUT, index=False)
    print(f"Saved {len(df)} rows to {OUT}")

if __name__ == "__main__":
    clean()