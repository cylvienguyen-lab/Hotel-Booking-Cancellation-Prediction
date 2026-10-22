"""Stage error_analysis: booking- and segment-level loss of the cv candidate against Dummy.

Rebuilt from reports/rf17_error_analysis_v1 (its original code was not kept); the rules
below reproduce that run's booking_errors.csv and group_errors.csv.
"""
import argparse

import numpy as np
import pandas as pd

import common


def individual_log_loss(labels, probabilities):
    labels, probabilities = np.asarray(labels), np.asarray(probabilities, dtype=float)
    return -np.where(labels == 1, np.log(probabilities), np.log1p(-probabilities))


def analyse(oof, dev, membership, settings):
    # LeadTime bands use quartiles of the fold-1 eligible training cohort
    initial_keys = membership.loc[membership['Fold'].eq(1) & membership['Role'].eq('Train'), common.KEYS]
    initial_train = initial_keys.merge(dev[common.KEYS + ['LeadTime']], on=common.KEYS,
                                       how='left', validate='one_to_one')
    inner_edges = np.unique(initial_train['LeadTime'].quantile([0.25, 0.5, 0.75]).to_numpy())
    edges = np.r_[-np.inf, inner_edges, np.inf]

    # Score every OOF booking against the Dummy prior of the same window
    bookings = oof[['Hotel', 'Booking_ID', 'Model', 'Fold', 'PredictionDate', 'IsCanceled',
                    'RawProbability', 'FeatureSet', 'ConfigID', 'DummyProbability']].rename(
        columns={'ConfigID': 'ReferenceConfigID', 'DummyProbability': 'RawProbability_dummy'})
    bookings = bookings.merge(dev[common.KEYS + ['CutoffDate', 'LeadTime', 'CustomerType', 'MarketSegment']],
                              on=common.KEYS, how='left', validate='one_to_one')
    if bookings['CutoffDate'].isna().any() or not bookings['CutoffDate'].eq(bookings['PredictionDate']).all():
        raise AssertionError('OOF bookings must match development rows and prediction dates')
    if bookings['LeadTime'].isna().any():
        raise AssertionError('LeadTime bands require complete LeadTime values')
    bands = pd.cut(bookings['LeadTime'], bins=edges, labels=False, include_lowest=True) + 1
    bookings['LeadTimeBand'] = 'G' + bands.astype(int).astype(str)
    bookings['IndividualLogLoss'] = individual_log_loss(bookings['IsCanceled'], bookings['RawProbability'])
    bookings['DummyIndividualLogLoss'] = individual_log_loss(bookings['IsCanceled'],
                                                             bookings['RawProbability_dummy'])
    bookings['ExcessLogLoss'] = bookings['IndividualLogLoss'] - bookings['DummyIndividualLogLoss']
    threshold = bookings.groupby(['Hotel', 'IsCanceled'])['IndividualLogLoss'].transform(
        lambda losses: losses.quantile(settings['tail_quantile']))
    bookings['HighLossWithinClass'] = bookings['IndividualLogLoss'].ge(threshold)
    bookings = bookings.sort_values(['Hotel', 'Fold', 'Booking_ID']).reset_index(drop=True)

    # Aggregate loss by segment and window; contribution is scaled by full window size
    window_rows = bookings.groupby(['Hotel', 'Fold']).size().rename('Rows').reset_index()
    parts = []
    for dimension in settings['dimensions']:
        column = 'LeadTimeBand' if dimension == 'LeadTime' else dimension
        grouped = bookings.groupby(['Hotel', 'Fold', column]).agg(
            Rows=('Booking_ID', 'size'), Positives=('IsCanceled', 'sum'),
            MeanProbability=('RawProbability', 'mean'), LogLoss=('IndividualLogLoss', 'mean'),
            DummyLogLoss=('DummyIndividualLogLoss', 'mean'), TotalLoss=('IndividualLogLoss', 'sum'),
            TotalExcessLoss=('ExcessLogLoss', 'sum'), CancelRate=('IsCanceled', 'mean'),
        ).reset_index().rename(columns={column: 'Group'})
        grouped.insert(2, 'Dimension', dimension)
        parts.append(grouped)
    groups = pd.concat(parts, ignore_index=True)
    groups['ProbabilityBias'] = groups['MeanProbability'] - groups['CancelRate']
    groups['ExcessLogLoss'] = groups['LogLoss'] - groups['DummyLogLoss']
    groups = groups.merge(window_rows, on=['Hotel', 'Fold'], suffixes=('', '_window'),
                          validate='many_to_one')
    groups['CVExcessContribution'] = groups['TotalExcessLoss'] / groups['Rows_window']
    groups = groups.sort_values(['Hotel', 'Dimension', 'Group', 'Fold']).reset_index(drop=True)
    return bookings, groups, inner_edges


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hotel', required=True)
    hotel = parser.parse_args().hotel
    settings = common.load_params()['error_analysis']
    dev = common.read_parquet(common.ARTIFACTS / 'data' / hotel / 'dev.parquet')
    membership = common.read_tables(common.ARTIFACTS / 'split' / hotel)['cv_membership.csv']
    oof = common.read_tables(common.ARTIFACTS / 'cv' / hotel)['oof_predictions.csv']
    bookings, groups, inner_edges = analyse(oof, dev, membership, settings)
    common.write_tables(common.ARTIFACTS / 'error_analysis' / hotel,
                        {'booking_errors.csv': bookings, 'group_errors.csv': groups},
                        extra={'lead_time_inner_edges': inner_edges.tolist(),
                               'tail_quantile': settings['tail_quantile']})
    print(f'{hotel}: {len(bookings):,} bookings | LeadTime edges {inner_edges.tolist()} | '
          f'{int(bookings["HighLossWithinClass"].sum()):,} high-loss bookings')


if __name__ == '__main__':
    main()
