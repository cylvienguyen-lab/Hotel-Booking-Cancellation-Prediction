"""Temporal split and label eligibility for one hotel."""
import json

import pandas as pd

import common
from utils import model_split

MEMBERSHIP_TABLES = ('partition_membership', 'cv_membership', 'fit_eligibility',
                     'refit_membership', 'calibration_membership')


def audit_split(hotel, source, result, split_config):
    # Re-derive every membership with independent filters (notebook 3.4)
    checks = []

    def record(name, passed):
        checks.append({'Check': name, 'Passed': bool(passed)})

    def same_ids(name, actual, expected):
        actual = list(actual)
        record(name, len(actual) == len(set(actual)) and set(actual) == set(expected))

    column = split_config['split_column']
    refit_date = pd.Timestamp(split_config['test_start'])
    test_end = pd.Timestamp(split_config['test_end'])
    source = source.copy()
    expected_checkout = source['ArrivalDate'] + pd.to_timedelta(source['TotalNights'], unit='D')
    source['LabelAvailableDate'] = source['ReservationStatusDate']
    checkout = source['ReservationStatus'].eq('Check-Out')
    source.loc[checkout, 'LabelAvailableDate'] = pd.concat(
        [source.loc[checkout, 'ReservationStatusDate'], expected_checkout.loc[checkout]], axis=1
    ).max(axis=1)
    development = source.loc[source[column].lt(refit_date)]
    test = source.loc[source[column].ge(refit_date) & source[column].lt(test_end)]
    partitions = result['partition_membership']
    cv = result['cv_membership']
    eligibility = result['fit_eligibility']
    refit_ids = result['refit_membership']['Booking_ID']
    calibration = result['calibration_membership']

    same_ids('Partition covers input once', partitions['Booking_ID'], source['Booking_ID'])
    for name, expected in [('Development', development), ('Test', test)]:
        same_ids(f'{name} membership',
                 partitions.loc[partitions['Partition'].eq(name), 'Booking_ID'],
                 expected['Booking_ID'])
    record('Membership hotel keys match the selected hotel',
           all(result[name]['Hotel'].eq(hotel).all() for name in MEMBERSHIP_TABLES))

    validation_ids = []
    for fold in range(1, split_config['n_folds'] + 1):
        fit_date = pd.Timestamp(split_config['first_fit_date']) + pd.DateOffset(
            months=(fold - 1) * split_config['validation_months'])
        validation_end = fit_date + pd.DateOffset(months=split_config['validation_months'])
        candidates = development.loc[development[column].lt(fit_date)]
        expected_train = candidates.loc[candidates['LabelAvailableDate'].lt(fit_date)]
        expected_validation = development.loc[
            development[column].ge(fit_date) & development[column].lt(validation_end)]
        train_ids = cv.loc[cv['Fold'].eq(fold) & cv['Role'].eq('Train'), 'Booking_ID']
        valid_ids = cv.loc[cv['Fold'].eq(fold) & cv['Role'].eq('Validation'), 'Booking_ID']
        same_ids(f'Fold {fold}: eligible train', train_ids, expected_train['Booking_ID'])
        same_ids(f'Fold {fold}: complete validation', valid_ids, expected_validation['Booking_ID'])
        record(f'Fold {fold}: train and validation disjoint',
               not set(train_ids).intersection(valid_ids))
        fold_eligibility = eligibility.loc[
            eligibility['Stage'].eq('CV') & eligibility['Fold'].eq(fold).fillna(False)]
        same_ids(f'Fold {fold}: eligibility candidates',
                 fold_eligibility['Booking_ID'], candidates['Booking_ID'])
        same_ids(f'Fold {fold}: eligibility agrees with train',
                 fold_eligibility.loc[fold_eligibility['Eligible'], 'Booking_ID'], train_ids)
        record(f'Fold {fold}: nonempty train and validation',
               len(train_ids) > 0 and len(valid_ids) > 0)
        record(f'Fold {fold}: correct fit date',
               cv.loc[cv['Fold'].eq(fold), 'FitDate'].eq(fit_date).all()
               and fold_eligibility['FitDate'].eq(fit_date).all())
        record(f'Fold {fold}: validation labels ready before refit',
               expected_validation['LabelAvailableDate'].lt(refit_date).all())
        validation_ids.extend(valid_ids.tolist())

    # Check final refit membership and recorded label dates
    expected_refit = development.loc[development['LabelAvailableDate'].lt(refit_date)]
    final_eligibility = eligibility.loc[eligibility['Stage'].eq('FinalRefit')]
    same_ids('Final refit uses every eligible development row', refit_ids, expected_refit['Booking_ID'])
    same_ids('Final eligibility agrees with refit membership',
             final_eligibility.loc[final_eligibility['Eligible'], 'Booking_ID'], refit_ids)
    recorded = eligibility.merge(source[['Booking_ID', 'LabelAvailableDate']], on='Booking_ID',
                                 suffixes=('_recorded', '_expected'), validate='many_to_one')
    record('Recorded label availability matches the agreed proxy',
           recorded['LabelAvailableDate_recorded'].eq(recorded['LabelAvailableDate_expected']).all())
    same_ids('Calibration is the unique union of validation rows',
             calibration['Booking_ID'], validation_ids)
    test_ids = set(partitions.loc[partitions['Partition'].eq('Test'), 'Booking_ID'])
    record('Test excluded from CV, refit and calibration memberships',
           not test_ids & (set(cv['Booking_ID']) | set(refit_ids) | set(calibration['Booking_ID'])))
    return pd.DataFrame(checks)


def run_split(hotel, model_data, config, output_root):
    split_config = dict(config['split'])
    before = model_data.copy(deep=True)
    result = model_split.make_splits(model_data, split_config)
    pd.testing.assert_frame_equal(model_data, before)
    checks = audit_split(hotel, model_data, result, split_config)
    checks.insert(0, 'Hotel', hotel)
    if not checks['Passed'].all():
        print(checks.loc[~checks['Passed']].to_string())
        raise AssertionError(f'{hotel} split audit failed')

    # Save memberships in the notebook's CSV format; the audit omits Test outcome counts
    split_directory = output_root / 'split' / hotel
    partition_summary = result['audit']['partition_summary'].copy()
    partition_summary['Positives'] = partition_summary['Positives'].astype('Int64')
    partition_summary.loc[partition_summary['Cohort'].eq('Test'), 'Positives'] = pd.NA
    common.write_tables(split_directory, {f'{name}.csv': result[name] for name in MEMBERSHIP_TABLES},
                        extra={'hotel': hotel, 'resolved_config': result['config']})
    def records(table):
        return json.loads(table.to_json(orient='records', date_format='iso', date_unit='s'))

    common.write_json(split_directory / 'audit.json', {
        'hotel': hotel, 'note': 'Test Positives withheld until the final stage',
        'cv_summary': records(result['audit']['cv_summary']),
        'partition_summary': records(partition_summary), 'checks': records(checks),
    })

    # Keep data partitions in memory; write only the audit and memberships.
    partition = result['partition_membership'].set_index('Booking_ID')['Partition']
    labels = model_data['Booking_ID'].map(partition)
    dev = model_data.loc[labels.eq('Development')].reset_index(drop=True)
    test = model_data.loc[labels.eq('Test')].reset_index(drop=True)
    print(f'{hotel}: {len(checks)} split checks passed | development {labels.eq("Development").sum():,}'
          f' | test {labels.eq("Test").sum():,}')
    return {'tables': result, 'checks': checks, 'dev': dev, 'test': test}
