"""Fold-fitted preprocessing for the T-14 initial experiments."""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(
    selected_features: dict,
    feature_contract: dict,
    model_name: str,
    engineered_numeric_features: tuple[str, ...] = (),
) -> ColumnTransformer | str:
    """Return an *unfitted* transformer for the Gold predictor roles.

    ``selected_features`` maps numeric, binary and categorical roles to names.
    The numeric imputer uses fold-training medians; categorical imputation and
    one-hot categories are learned on that fold's training rows only. Unknown
    validation categories become all-zero indicators. LR scales numeric values;
    tree models keep their original scale. Dummy ignores predictors.

    Args:
        selected_features: Names grouped as numeric, binary and categorical.
        feature_contract: Gold role lists and excluded metadata/outcome names.
        model_name: One of Dummy, LR, RF or XGB.
        engineered_numeric_features: Explicit numeric columns produced within
            a screening fold. Empty for the original 30-predictor pipeline.

    Returns:
        An unfitted ColumnTransformer, or passthrough for Dummy.
    """
    if model_name not in {"Dummy", "LR", "RF", "XGB"}:
        raise ValueError(f"Unsupported initial model: {model_name}")
    groups = ("numeric", "binary", "categorical")
    if set(selected_features) != set(groups):
        raise ValueError("Predictor groups must include numeric, binary and categorical")
    columns = [name for group in groups for name in selected_features[group]]
    if not columns or len(columns) != len(set(columns)):
        raise ValueError("Feature names are empty or repeated")
    forbidden = set(feature_contract["metadata_only"]) | set(
        feature_contract["excluded_predictors"]
    ) | {
        feature_contract["target"], "ReservationStatusDate", "ReservationStatus",
        "LabelAvailableDate", "Partition", "ValFold", "FinalRefit",
    }
    if forbidden.intersection(columns):
        raise ValueError(f"Outcome or metadata in predictors: {sorted(forbidden.intersection(columns))}")
    if set(engineered_numeric_features) - set(selected_features["numeric"]):
        raise ValueError("Engineered columns must be in the numeric role")
    if not set(selected_features["numeric"] + selected_features["binary"]).issubset(
        set(feature_contract["numeric_predictors"]) | set(engineered_numeric_features)
    ):
        raise ValueError("Numeric/binary feature outside Gold contract")
    if not set(selected_features["categorical"]).issubset(
        feature_contract["categorical_predictors"]
    ):
        raise ValueError("Categorical feature outside Gold contract")
    if model_name == "Dummy":
        return "passthrough"

    numeric_steps = [("imputer", SimpleImputer(strategy="median", keep_empty_features=True))]
    if model_name == "LR":
        numeric_steps.append(("scaler", StandardScaler()))
    numeric = Pipeline(numeric_steps)
    binary = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric, selected_features["numeric"]),
            ("binary", binary, selected_features["binary"]),
            ("categorical", categorical, selected_features["categorical"]),
        ],
        remainder="drop",
        sparse_threshold=0,
    )
