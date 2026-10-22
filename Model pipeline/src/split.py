"""Notebook 01 split of development bookings using the unchanged eligibility utility."""
import pandas as pd

import common
from utils import model_split

MEMBERSHIP_TABLES = ('partition_membership', 'cv_membership', 'fit_eligibility',
                     'refit_membership', 'calibration_membership')


def run_split(hotel, development, config, output_root):
    if not development['CutoffDate'].lt(pd.Timestamp(config['split']['test_start'])).all():
        raise ValueError('Notebook 01 split accepts development bookings only')
    result = model_split.make_splits(development, dict(config['split']))
    # The utility also supports later notebooks. Do not publish its empty held-out summary.
    result['audit']['partition_summary'] = result['audit']['partition_summary'].loc[
        lambda table: table['Cohort'].ne('Test')].reset_index(drop=True)
    directory = output_root / 'split' / hotel
    common.write_tables(directory, {f'{name}.csv': result[name] for name in MEMBERSHIP_TABLES},
                        extra={'hotel': hotel, 'resolved_config': result['config'],
                               'scope': 'development only'})
    common.write_tables(directory / 'summary', {
        'cv_summary.csv': result['audit']['cv_summary'],
        'partition_summary.csv': result['audit']['partition_summary'],
    })
    return {'tables': result, 'dev': development}
