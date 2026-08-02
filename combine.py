from pathlib import Path
import pandas as pd

INPUT_DIR = Path( "logs/classification/ranking" )

TYPE = [
    "model_comparison",
    "logistic_regression_coefficients",
    "random_forest_importances"
]

for name in TYPE:

    dfs = []

    for file in INPUT_DIR.glob( f"*_{name}.csv" ):
        horizon = file.stem.replace( f"_{name}", "" )

        df = pd.read_csv( file )

        # Remove the unnamed index column if present
        if df.columns[0].startswith( "Unnamed" ) or df.columns[0] == "":
            df = df.drop( columns=df.columns[0] )

        horizon = file.stem.repalce( f"_{name}", "" )
        df["horizon"] = horizon

        dfs.append( df )

    if not dfs:
        print( f"No files found for {name}, skipping." )
        continue

    combined = pd.concat( dfs, ignore_index=True )

    # Ordering
    combined["horizon"] = pd.Categorical( combined["horizon"], categories=HORIZONS, ordered=True )
    combined = combined.sort_values( ["features", "horizon"] )

    # Sorting
    if name == "model_comparison":
        combined = combined.sort_values( ["horizon", "model"] )
    else:
        combined = combined.sort_values( ["features", "horizon"] )

    output = INPUT_DIR / f"combined_{name}.csv"

    combined.to_csv( output, index=False )

    print( f"\nSaved {output}" )
    print( combined )