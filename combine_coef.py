from pathlib import Path
import pandas as pd

INPUT_DIR = Path( "logs/simple_classification" )

TYPE = [ "logistic_regression_coefficients", "random_forest_importances" ]

for name in TYPE:

    dfs = []

    for file in INPUT_DIR.glob( f"*_{name}.csv" ):
        horizon = file.stem.replace( f"_{name}", "" )

        df = pd.read_csv( file )

        # Remove the unnamed index column if present
        if df.columns[0].startswith( "Unnamed" ) or df.columns[0] == "":
            df = df.drop( columns=df.columns[0] )

        df["horizon"] = horizon
        dfs.append(df)

    combined = pd.concat( dfs, ignore_index=True )

    # Optional ordering
    order = ["1d", "1w", "1m", "6m", "1y", "3y", "5y"]
    combined["horizon"] = pd.Categorical( combined["horizon"], categories=order, ordered=True )
    combined = combined.sort_values( ["features", "horizon"] )

    combined.to_csv( INPUT_DIR / f"combined_{name}.csv", index=False )

    print( combined )