"""Shared paths, parameters, Gold loading and table I/O for the DVC stages."""
import json
import sys
from pathlib import Path

import pandas as pd
import yaml
from threadpoolctl import threadpool_limits

# LR (lbfgs) results change in the last digits with the BLAS thread count; the notebook runs
# match exactly at 2 threads, and pinning keeps results independent of the machine's cores
threadpool_limits(limits=2, user_api='blas')

MODEL_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = MODEL_DIR.parent
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

GOLD_ROOT = PROJECT_ROOT / 'Data pipeline' / 'hotel_booking_pipeline' / 'datamart' / 'gold'
ARTIFACTS = MODEL_DIR / 'artifacts'
METRICS = MODEL_DIR / 'metrics'
TUNED_PARAMS = MODEL_DIR / 'tuned_params'
TARGET = 'IsCanceled'
KEYS = ['Hotel', 'Booking_ID']
FEATURE_GROUPS = ('numeric', 'binary', 'categorical')
GOLD_DATE_COLUMNS = ('ArrivalDate', 'BookingCreatedDate', 'CutoffDate', 'FXObservationDate',
                     'TrendWeekStart', 'TrendWeekEnd', 'TrendAvailableDate',
                     'ReservationStatusDate')
TOLERANCE = 1e-12


def load_params():
    return yaml.safe_load((PROJECT_ROOT / 'params.yaml').read_text(encoding='utf-8'))


def load_contract():
    return json.loads((GOLD_ROOT / 'feature_contract.json').read_text(encoding='utf-8'))


def load_gold(hotel):
    # Read the model dataset and outcome metadata (notebook 2.1-2.2). The pyarrow reader was
    # checked against the notebook's Spark toPandas() frames: same values, dtypes and row order
    model = pd.read_parquet(GOLD_ROOT / 'model_dataset' / f'gold_model_dataset_{hotel}.parquet',
                            engine='pyarrow')
    labels = pd.read_parquet(GOLD_ROOT / 'labels' / f'gold_label_store_{hotel}.parquet',
                             engine='pyarrow')
    loaded = model.merge(labels[['Booking_ID', 'ReservationStatus', 'ReservationStatusDate']],
                         on='Booking_ID', how='left', validate='one_to_one')
    for column in GOLD_DATE_COLUMNS:
        loaded[column] = pd.to_datetime(loaded[column], errors='raise')
    if len(loaded) != len(model) or not loaded['Booking_ID'].is_unique:
        raise ValueError(f'{hotel}: outcome metadata join changed the modelling grain.')
    return loaded


def write_parquet(table, path):
    # Save a data partition and require an exact read-back
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(path, index=False, engine='pyarrow')
    pd.testing.assert_frame_equal(table.reset_index(drop=True), read_parquet(path), check_exact=True)


def read_parquet(path):
    return pd.read_parquet(path, engine='pyarrow')


def table_schema(table):
    return {'dtypes': {column: str(dtype) for column, dtype in table.dtypes.items()},
            'date_columns': [column for column in table
                             if pd.api.types.is_datetime64_any_dtype(table[column])]}


def write_tables(directory, tables, extra=None):
    # Write CSVs in the notebook's format plus schema.json, then verify every read-back
    directory.mkdir(parents=True, exist_ok=True)
    schema = {}
    for name, table in tables.items():
        schema[name] = table_schema(table)
        (directory / name).write_bytes(table.to_csv(
            index=False, date_format='%Y-%m-%d', lineterminator='\n').encode('utf-8'))
    document = {**(extra or {}), 'csv_schema': schema}
    (directory / 'schema.json').write_text(json.dumps(document, indent=2) + '\n', encoding='utf-8')
    for name, table in tables.items():
        pd.testing.assert_frame_equal(table.reset_index(drop=True),
                                      read_table(directory / name, schema[name]), check_exact=True)


def read_table(path, schema):
    # Restore saved dtypes and dates exactly as the notebook's read-back checks did
    dtype_map = {column: dtype for column, dtype in schema['dtypes'].items()
                 if column not in schema['date_columns']}
    nullable = {column: [''] for column, dtype in dtype_map.items()
                if dtype in ('Int64', 'Float64', 'boolean') or dtype.startswith('float')}
    table = pd.read_csv(path, dtype=dtype_map, parse_dates=schema['date_columns'],
                        keep_default_na=False, na_values=nullable, float_precision='round_trip')
    for column in schema['date_columns']:
        table[column] = table[column].astype(schema['dtypes'][column])
    return table


def read_tables(directory):
    schema = json.loads((directory / 'schema.json').read_text(encoding='utf-8'))['csv_schema']
    return {name: read_table(directory / name, columns) for name, columns in schema.items()}


def write_json(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2) + '\n', encoding='utf-8')


def feature_config(params, name):
    groups = params['feature_sets'][name]
    return {group: list(groups[group]) for group in FEATURE_GROUPS}


def feature_columns(groups):
    return [column for group in FEATURE_GROUPS for column in groups[group]]


def check_features(groups, contract):
    # Predictors exclude IDs, outcome fields and metadata, and match the contract types
    columns = feature_columns(groups)
    forbidden = set(contract['metadata_only']) | {
        contract['target'], 'ReservationStatus', 'ReservationStatusDate', 'LabelAvailableDate',
    }
    if set(columns) & forbidden:
        raise ValueError('Predictors must exclude IDs, outcome fields and metadata')
    if not set(groups['numeric'] + groups['binary']).issubset(contract['numeric_predictors']):
        raise ValueError('Numeric feature types do not match the contract')
    if not set(groups['categorical']).issubset(contract['categorical_predictors']):
        raise ValueError('Categorical feature types do not match the contract')
    if len(columns) != len(set(columns)):
        raise ValueError('Feature set repeats a column')


def manual_params(params, model_key, hotel):
    spec = params['models'][model_key]
    overrides = (spec.get('hotel_overrides') or {}).get(hotel) or {}
    return {**spec['params'], **overrides}


def model_settings(params, hotel, tuned=None):
    # Resolve the evaluated model's estimator, parameters and their origin
    model_key = params['model']
    spec = params['models'][model_key]
    if spec['estimator'] == 'dummy' or params['hyperparams'] == 'manual':
        return {'key': model_key, 'estimator': spec['estimator'],
                'params': manual_params(params, model_key, hotel), 'config_id': 'manual'}
    if params['hyperparams'] != 'tuned':
        raise ValueError("hyperparams must be 'tuned' or 'manual'")
    if model_key not in tuned:
        raise ValueError(f'No tuned parameters for model {model_key}')
    return {'key': model_key, 'estimator': spec['estimator'],
            'params': tuned[model_key]['params'], 'config_id': tuned[model_key]['config_id']}


def label_dates(fit_eligibility):
    # One label-availability date per booking from the saved eligibility records
    dates = fit_eligibility[KEYS + ['LabelAvailableDate']].drop_duplicates()
    if dates.duplicated(KEYS).any():
        raise AssertionError('Conflicting label-availability dates')
    return dates
