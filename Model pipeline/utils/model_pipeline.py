from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(feature_config, scale_numeric=True):
    # Check that feature groups contain no duplicate or overlapping columns
    group_names = ('numeric', 'binary', 'categorical')
    missing_groups = set(group_names) - set(feature_config)
    if missing_groups:
        raise ValueError(f'Missing feature groups: {sorted(missing_groups)}')
    columns = {}
    for group in group_names:
        values = feature_config[group]
        if not isinstance(values, (list, tuple)) or any(
            not isinstance(value, str) or not value for value in values
        ):
            raise ValueError(f'{group} must be a list of column names')
        columns[group] = list(values)
    all_columns = [column for group in group_names for column in columns[group]]
    if not all_columns:
        raise ValueError('Configure at least one feature')
    if len(all_columns) != len(set(all_columns)):
        raise ValueError('Feature groups contain duplicate or overlapping columns')
    if not isinstance(scale_numeric, bool):
        raise ValueError('scale_numeric must be True or False')

    # Build numeric processing: median imputation and optional scaling
    numeric_steps = [('imputer', SimpleImputer(strategy='median'))]
    if scale_numeric:
        numeric_steps.append(('scaler', StandardScaler()))
    numeric = Pipeline(numeric_steps)

    # Build binary processing: most-frequent imputation
    binary = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
    ])

    # Build categorical processing: missing token and one-hot encoding
    categorical = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='__MISSING__')),
        ('encoder', OneHotEncoder(
            handle_unknown='ignore', drop=None, sparse_output=False,
        )),
    ])

    # Combine the three branches and exclude unlisted columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('numeric', numeric, columns['numeric']),
            ('binary', binary, columns['binary']),
            ('categorical', categorical, columns['categorical']),
        ],
        remainder='drop',
        sparse_threshold=0,
    )

    # Return the unfitted preprocessor
    return preprocessor


def build_pipeline(model_name, feature_config, model_params):
    # Create a new estimator from the supplied parameters
    if model_name == 'dummy':
        estimator = DummyClassifier(**model_params)
    elif model_name == 'logistic_regression':
        estimator = LogisticRegression(**model_params)
    elif model_name == 'random_forest':
        estimator = RandomForestClassifier(**model_params)
    elif model_name == 'xgboost':
        from xgboost import XGBClassifier

        estimator = XGBClassifier(**model_params)
    else:
        raise ValueError(f'Unsupported model: {model_name}')

    # Scale numeric inputs for LR and keep their original scale for tree models
    steps = []
    if model_name in ('logistic_regression', 'random_forest', 'xgboost'):
        steps.append(('preprocessor', build_preprocessor(
            feature_config, scale_numeric=(model_name == 'logistic_regression'),
        )))
    steps.append(('model', estimator))

    # Return an unfitted pipeline for this model and fold
    return Pipeline(steps)
