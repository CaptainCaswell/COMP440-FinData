from pathlib import Path
import pandas as pd

LOG_FOLDER = Path("logs/simple_classification")

OUTPUT_FILE = LOG_FOLDER / "combined_model_comparison.csv"

HORIZONS = [
    "1d",
    "1w",
    "1m",
    "6m",
    "1y",
    "3y",
    "5y",
]


def main():
    results = []

    for horizon in HORIZONS:
        file = LOG_FOLDER / f"{horizon}_model_comparison.csv"

        if not file.exists():
            raise FileNotFoundError( f"Missing result file: {file}" )

        print( f"Loading {file}" )

        df = pd.read_csv( file )
        results.append( df )

    combined = pd.concat( results, ignore_index=True )

    print( "\nCombined results:" )
    print( combined.to_string( index=False ) )

    combined.to_csv( OUTPUT_FILE, index=False )

    print( f"\nSaved combined results to {OUTPUT_FILE}" )


if __name__ == "__main__":
    main()