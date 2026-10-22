from numbers import Integral

import pandas as pd


def make_splits(hotel_data, split_config):

    # Prepare the working data and fold calendar
    required_columns = [
        'Hotel', 'Booking_ID', 'CutoffDate', 'ArrivalDate', 'TotalNights',
        'ReservationStatus', 'ReservationStatusDate', 'IsCanceled',
    ]
    missing_columns = set(required_columns) - set(hotel_data.columns)
    if missing_columns:
        raise ValueError(f'Missing split inputs: {sorted(missing_columns)}')
    data = hotel_data[required_columns].copy().reset_index(drop=True)
    if data.empty or data.isna().any().any():
        raise ValueError('Split inputs must be nonempty and contain no missing metadata')
    if data['Hotel'].nunique() != 1 or data['Booking_ID'].duplicated().any():
        raise ValueError('Pass one hotel with one row per Booking_ID')
    if not data['ReservationStatus'].isin(['Check-Out', 'Canceled', 'No-Show']).all():
        raise ValueError('Unknown ReservationStatus prevents label availability calculation')
    if not data['IsCanceled'].isin([0, 1]).all():
        raise ValueError('IsCanceled must contain binary observed labels')
    for column in ('CutoffDate', 'ArrivalDate', 'ReservationStatusDate'):
        data[column] = pd.to_datetime(data[column], errors='raise').dt.normalize()
    if data[['CutoffDate', 'ArrivalDate', 'ReservationStatusDate']].isna().any().any():
        raise ValueError('Split inputs contain missing dates after date conversion')
    data = data.sort_values(['CutoffDate', 'Booking_ID']).reset_index(drop=True)

    if split_config['split_column'] != 'CutoffDate':
        raise ValueError('This split specification requires CutoffDate')
    validation_months = split_config['validation_months']
    n_folds = split_config['n_folds']
    for name, value in [('validation_months', validation_months), ('n_folds', n_folds)]:
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f'{name} must be a positive integer')
    first_fit = pd.Timestamp(split_config['first_fit_date']).normalize()
    test_start = pd.Timestamp(split_config['test_start']).normalize()
    test_end = pd.Timestamp(split_config['test_end']).normalize()
    if pd.isna(first_fit) or pd.isna(test_start) or pd.isna(test_end):
        raise ValueError('Split calendar dates must not be missing')
    if not first_fit < test_start < test_end:
        raise ValueError('Require first_fit_date < test_start < test_end')
    calendar = []
    for fold in range(1, n_folds + 1):
        fit_date = first_fit + pd.DateOffset(months=(fold - 1) * validation_months)
        validation_end = first_fit + pd.DateOffset(months=fold * validation_months)
        if validation_end > test_start:
            raise ValueError(f'Fold {fold} validation overlaps the final test')
        calendar.append({
            'Fold': fold, 'FitDate': fit_date,
            'ValidationStart': fit_date, 'ValidationEnd': validation_end,
        })
    if data['CutoffDate'].ge(test_end).any():
        raise ValueError('Input contains bookings outside the configured test end')

    # Calculate label availability dates
    expected_checkout = data['ArrivalDate'] + pd.to_timedelta(data['TotalNights'], unit='D')
    later_checkout = pd.concat(
        [data['ReservationStatusDate'], expected_checkout], axis=1
    ).max(axis=1, skipna=False)
    data['LabelAvailableDate'] = data['ReservationStatusDate'].where(
        ~data['ReservationStatus'].eq('Check-Out'), later_checkout
    )

    # Assign development and test membership
    keys = ['Hotel', 'Booking_ID']
    development = data.loc[data['CutoffDate'].lt(test_start)].copy()
    test = data.loc[data['CutoffDate'].ge(test_start)].copy()
    partition_membership = data[keys].copy()
    partition_membership['Partition'] = data['CutoffDate'].lt(test_start).map(
        {True: 'Development', False: 'Test'}
    )
    cv_parts = []
    eligibility_parts = []
    cv_summary_rows = []

    # Build training, validation and eligibility records for each CV fold
    for window in calendar:
        fold = window['Fold']
        fit_date = window['FitDate']
        candidates = development.loc[development['CutoffDate'].lt(fit_date)]
        eligible = candidates['LabelAvailableDate'].lt(fit_date)
        train = candidates.loc[eligible]
        validation = development.loc[
            development['CutoffDate'].ge(window['ValidationStart'])
            & development['CutoffDate'].lt(window['ValidationEnd'])
        ]

        eligibility = candidates[keys + ['LabelAvailableDate']].copy()
        eligibility['Stage'] = 'CV'
        eligibility['Fold'] = fold
        eligibility['FitDate'] = fit_date
        eligibility['Eligible'] = eligible
        eligibility['ExclusionReason'] = eligible.map(
            {True: '', False: 'Label not available before fit'}
        )
        eligibility_parts.append(eligibility)
        for role, cohort in [('Train', train), ('Validation', validation)]:
            membership = cohort[keys].copy()
            membership['Fold'] = fold
            membership['FitDate'] = fit_date
            membership['Role'] = role
            cv_parts.append(membership)

        cv_summary_rows.append({
            'Hotel': data['Hotel'].iloc[0], 'Fold': fold, 'FitDate': fit_date,
            'ValidationEnd': window['ValidationEnd'],
            'TrainCandidates': len(candidates), 'TrainRows': len(train),
            'TrainPositives': int(train['IsCanceled'].sum()),
            'ValidationRows': len(validation),
            'ValidationPositives': int(validation['IsCanceled'].sum()),
            'TrainLabelsUnavailable': int((~eligible).sum()),
            'ValidationLabelsUnavailableAtRefit': int(
                validation['LabelAvailableDate'].ge(test_start).sum()
            ),
        })
    cv_membership = pd.concat(cv_parts, ignore_index=True)

    # Build final-refit membership and eligibility records
    refit_eligible = development['LabelAvailableDate'].lt(test_start)
    refit = development.loc[refit_eligible]
    refit_excluded = development.loc[~refit_eligible]
    refit_membership = refit[keys].copy()
    refit_membership['FitDate'] = test_start
    final_eligibility = development[keys + ['LabelAvailableDate']].copy()
    final_eligibility['Stage'] = 'FinalRefit'
    final_eligibility['Fold'] = pd.NA
    final_eligibility['FitDate'] = test_start
    final_eligibility['Eligible'] = refit_eligible
    final_eligibility['ExclusionReason'] = refit_eligible.map(
        {True: '', False: 'Label not available before fit'}
    )
    eligibility_parts.append(final_eligibility)
    fit_eligibility = pd.concat(eligibility_parts, ignore_index=True)
    fit_eligibility['Fold'] = fit_eligibility['Fold'].astype('Int64')

    # Build calibration membership from the validation folds
    calibration_membership = cv_membership.loc[
        cv_membership['Role'].eq('Validation'), keys + ['Fold']
    ].merge(data[keys + ['CutoffDate']], on=keys, how='left', validate='many_to_one')
    calibration_membership = calibration_membership.rename(
        columns={'CutoffDate': 'PredictionDate'}
    )

    # Build split audit summaries
    calibration = data.loc[
        data['Booking_ID'].isin(calibration_membership['Booking_ID'])
    ]
    partition_summary_rows = []
    for name, cohort in [
        ('Development', development), ('Final refit eligible', refit),
        ('Final refit excluded', refit_excluded), ('Calibration', calibration),
        ('Test', test),
    ]:
        partition_summary_rows.append({
            'Hotel': data['Hotel'].iloc[0], 'Cohort': name,
            'Rows': len(cohort), 'Positives': int(cohort['IsCanceled'].sum()),
        })
    actual_config = dict(split_config)
    actual_config.update({
        'hotel': data['Hotel'].iloc[0], 'input_rows': len(data),
        'development_start': data['CutoffDate'].min().strftime('%Y-%m-%d'),
        'first_fit_date': first_fit.strftime('%Y-%m-%d'),
        'test_start': test_start.strftime('%Y-%m-%d'),
        'test_end': test_end.strftime('%Y-%m-%d'),
        'final_refit_date': test_start.strftime('%Y-%m-%d'),
        'interval_convention': '[start, end)',
        'training_rule': 'CutoffDate < FitDate and LabelAvailableDate < FitDate',
        'label_availability_rule': {
            'Check-Out': 'max(ReservationStatusDate, ArrivalDate + TotalNights)',
            'Canceled / No-Show': 'ReservationStatusDate',
        },
        'folds': [
            {name: value.strftime('%Y-%m-%d') if isinstance(value, pd.Timestamp) else value
             for name, value in window.items()}
            for window in calendar
        ],
    })

    # Return the configuration, memberships and audit summaries
    return {
        'config': actual_config,
        'partition_membership': partition_membership.reset_index(drop=True),
        'cv_membership': cv_membership,
        'fit_eligibility': fit_eligibility,
        'refit_membership': refit_membership.reset_index(drop=True),
        'calibration_membership': calibration_membership,
        'audit': {
            'cv_summary': pd.DataFrame(cv_summary_rows),
            'partition_summary': pd.DataFrame(partition_summary_rows),
        },
    }
