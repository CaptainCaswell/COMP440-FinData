import pandas as pd
import joblib
import os
from pathlib import Path

# Files
FEATURE_FILE = "data/recent_features.parquet"
MODEL_DIR = "models"
OUTPUT_FILE = "data/recent_predictions.parquet"

TOP_PERCENT = 0.10
TOP_N = 20

# Models to load
MODELS = {
    "linear_regression": True,
    "random_forest": False,
    "gradient_boosting": False,
    "neural_network": True,
}

# Target features used during training
FEATURES = [
    "ret_1d",
    "ret_1w",
    "ret_1m",
    "ret_6m",
    "ret_1y",
    "ret_3y",
    "ret_5y",
]


def load_models():
    models = {}

    for name in MODELS:
        path = Path( MODEL_DIR ) / f"{name}.joblib"

        if not path.exists():
            print( f"Missing model: {path}" )
            continue

        models[name] = joblib.load( path )
        print( f"Loaded {name}" )

    return models


def load_scaler():
    path = Path( MODEL_DIR ) / "scaler.joblib"

    if path.exists():
        return joblib.load( path )

    return None

def prepare_features( df ):
    X = df.reindex( columns = FEATURES, fill_value = 0 ).copy()

    for col in X.columns:
        X[col] = X[col].fillna( X[col].median() )

    return X


def combined_score( results ):
    # Normalize models then create combined score

    score_cols = [ col for col in results.columns if col.endswith( "_score" )]

    if not score_cols:
        raise ValueError( "No model score columns found." )

    print( "\nNormalizing model scores..." )

    rank_cols = []

    for col in score_cols:
        rank_col = f"{col}_rank"
        results[rank_col] = results[col].rank( pct=True )

        rank_cols.append( rank_col )

    results["combined_score"] = results[rank_cols].mean( axis=1 )

    return results


def predict( df, models, scaler ):

    X = prepare_features( df )

    results = df.copy()

    for name, model in models.items():
        if MODELS[name]:
            X_input = scaler.transform( X )
        else:
            X_input = X

        prediction = model.predict( X_input )

        results[f"{name}_score"] = prediction

    results = combined_score(results)

    return results


def evaluate( predictions ):
    predictions = predictions.sort_values( "combined_score", ascending=False )

    print( f"\nTop {TOP_N} predictions:" )

    display_cols = [
        "ticker",
        "date",
        "combined_score"
    ]

    display_cols += [ col for col in predictions.columns if col.endswith( "_score_rank" ) ]

    print( predictions[display_cols].head( TOP_N ) .to_string( index=False ) )

    if "future_ret_1y" in predictions.columns:
        n = max( 1, int( len( predictions ) * TOP_PERCENT) )

        top = predictions.head( n )

        print(  "\n" + "=" * 60 )
        print( f"TOP {TOP_PERCENT:.0%} PERFORMANCE" )
        print(  "=" * 60 )

        print( f"Stocks: {n}" )
        print( f"Mean return:   {top['future_ret_1y'].mean():.2%}" )
        print( f"Median return: {top['future_ret_1y'].median():.2%}" )


    return predictions


def main():

    print("Loading feature data...")
    data = pd.read_parquet(FEATURE_FILE)

    print( f"Loaded {len(data)} stocks" )

    models = load_models()
    scaler = load_scaler()

    if not models:
        print("No models loaded")
        return

    if not scaler:
            print("No scaler loaded")
            return

    predictions = predict( data, models, scaler )

    predictions = evaluate( predictions )

    predictions.to_parquet( OUTPUT_FILE )

    print( f"\nSaved {OUTPUT_FILE}")


if __name__ == "__main__":
    main()