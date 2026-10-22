"""Descriptive diagnostics ported from ef05228 cells 64, 67, 120, 126 and 129."""
import pickle

import numpy as np
import pandas as pd

import common
import folds


def initial_diagnosis(workflow):
    contribution_parts, score_rows, cohort_parts, ranges = [], [], [], []
    config = workflow.config
    groups = common.feature_config(config, 'core21')
    columns = common.feature_columns(groups)
    for hotel in config['hotels']:
        dev = workflow.data[hotel]
        run = workflow.runs[(hotel, 'LR21')]
        pipeline = run['models'][1]
        before = pickle.dumps(pipeline)
        processor = pipeline.named_steps['preprocessor']
        estimator = pipeline.named_steps['model']
        fit_date, train, valid = folds.fold_cohorts(dev, workflow.splits[hotel]['cv_membership'], 1, columns)
        a, b = processor.transform(train[columns]), processor.transform(valid[columns])
        source_names = list(groups['numeric'] + groups['binary'])
        encoder = processor.named_transformers_['categorical'].named_steps['encoder']
        for column, categories in zip(groups['categorical'], encoder.categories_):
            source_names += [column] * len(categories)
        names, coefficients = processor.get_feature_names_out(), estimator.coef_[0]
        if len(names) != len(source_names):
            raise AssertionError('Incomplete encoded-feature mapping')
        shifts = (b.mean(axis=0) - a.mean(axis=0)) * coefficients
        contribution_parts.append(pd.DataFrame({
            'Hotel': hotel, 'EncodedFeature': names, 'SourceFeature': source_names,
            'Coefficient': coefficients, 'MeanContributionShift': shifts}))
        score_shift = float(pipeline.decision_function(valid[columns]).mean()
                            - pipeline.decision_function(train[columns]).mean())
        if not np.isclose(shifts.sum(), score_shift) or pickle.dumps(pipeline) != before:
            raise AssertionError('LR contributions failed reconciliation or changed fitted state')
        score_rows.append({'Hotel': hotel, 'MeanScoreShift': score_shift,
                           'SumOfFeatureShifts': float(shifts.sum()),
                           'ReconciliationDifference': float(abs(shifts.sum() - score_shift))})
        for feature in ('ArrivalDateYear', 'ArrivalDateWeekNumber'):
            ranges.append({'Hotel': hotel, 'Feature': feature, 'TrainMin': train[feature].min(),
                           'TrainMax': train[feature].max(), 'ValidationMin': valid[feature].min(),
                           'ValidationMax': valid[feature].max(),
                           'UnseenValidationValues': int((~valid[feature].isin(train[feature].unique())).sum())})
        candidates = dev.loc[dev.CutoffDate.lt(fit_date), common.KEYS + ['ArrivalDate', 'IsCanceled']].copy()
        eligibility = workflow.splits[hotel]['fit_eligibility']
        eligibility = eligibility.loc[eligibility.Stage.eq('CV') & eligibility.Fold.eq(1).fillna(False),
                                       common.KEYS + ['Eligible']]
        candidates = candidates.merge(eligibility, on=common.KEYS, validate='one_to_one')
        candidates['ArrivalMonth'] = candidates.ArrivalDate.dt.to_period('M').astype(str)
        candidates['Cohort'] = np.where(~candidates.Eligible, 'UnavailableRows',
                                        np.where(candidates.IsCanceled.eq(1), 'EligiblePositives', 'EligibleCheckOut'))
        counts = candidates.groupby(['ArrivalMonth', 'Cohort']).size().unstack(fill_value=0).reindex(
            columns=['EligibleCheckOut', 'EligiblePositives', 'UnavailableRows'], fill_value=0).reset_index()
        counts.insert(0, 'Hotel', hotel)
        counts['EligibleRows'] = counts.EligibleCheckOut + counts.EligiblePositives
        counts['EligibleCancelRate'] = counts.EligiblePositives / counts.EligibleRows.replace(0, np.nan)
        cohort_parts.append(counts)
    encoded = pd.concat(contribution_parts, ignore_index=True)
    contributions = encoded.groupby(['Hotel', 'SourceFeature'], sort=False).MeanContributionShift.sum().reset_index()
    rates = pd.concat([workflow.runs[(hotel, variant)]['fold_metrics']
                       for hotel in config['hotels'] for variant in ('Dummy', 'LR21')], ignore_index=True)
    return {'probability_rates.csv': rates[['Hotel', 'Variant', 'Fold', 'ValidationCancelRate', 'MeanProbability']],
            'arrival_cohorts.csv': pd.concat(cohort_parts, ignore_index=True),
            'encoded_contributions.csv': encoded, 'feature_contributions.csv': contributions,
            'score_reconciliation.csv': pd.DataFrame(score_rows), 'calendar_ranges.csv': pd.DataFrame(ranges)}


def model_diagnostics(workflow):
    from baseline import reliability
    config = workflow.config
    groups = common.feature_config(config, 'no_time18')
    columns = common.feature_columns(groups)
    cohorts, rate_rows, shift_rows, detail_rows, distribution_rows, customer_rows = {}, [], [], [], [], []
    metrics = pd.concat([workflow.runs[(hotel, variant)]['fold_metrics']
                         for hotel in config['hotels'] for variant in ('LR18', 'RF18', 'XGB18')], ignore_index=True)
    oof = pd.concat([workflow.runs[(hotel, variant)]['oof_predictions']
                     for hotel in config['hotels'] for variant in ('LR18', 'RF18', 'XGB18')], ignore_index=True)
    for hotel in config['hotels']:
        for fold in range(1, config['split']['n_folds'] + 1):
            _, train, valid = folds.fold_cohorts(workflow.data[hotel], workflow.splits[hotel]['cv_membership'], fold, columns)
            cohorts[(hotel, fold, 'Train')], cohorts[(hotel, fold, 'Validation')] = train, valid
            rate_rows.append({'Hotel': hotel, 'Fold': fold, 'TrainRows': len(train), 'ValidationRows': len(valid),
                              'TrainCancelRate': float(train.IsCanceled.mean()),
                              'ValidationCancelRate': float(valid.IsCanceled.mean())})
            for feature in groups['numeric']:
                a, b = train[feature].dropna(), valid[feature].dropna()
                difference, scale = abs(b.mean() - a.mean()), a.std(ddof=0)
                shift = difference / scale if scale > 0 else (0 if difference == 0 else np.nan)
                shift_rows.append({'Hotel': hotel, 'Fold': fold, 'Feature': feature,
                                   'Type': 'Numeric', 'Shift': shift})
            for feature in groups['categorical'] + groups['binary']:
                a = train[feature].fillna('__MISSING__').astype(str).value_counts(normalize=True)
                b = valid[feature].fillna('__MISSING__').astype(str).value_counts(normalize=True)
                levels = a.index.union(b.index)
                shift = .5 * (a.reindex(levels, fill_value=0) - b.reindex(levels, fill_value=0)).abs().sum()
                shift_rows.append({'Hotel': hotel, 'Fold': fold, 'Feature': feature,
                                   'Type': 'Categorical', 'Shift': shift})
            row = {'Hotel': hotel, 'Fold': fold}
            for role, cohort in [('Train', train), ('Validation', valid)]:
                selected = cohort.loc[cohort.CustomerType.eq('Transient')]
                row.update({f'{role}Rows': len(selected), f'{role}Share': len(selected) / len(cohort),
                            f'{role}Rate': float(selected.IsCanceled.mean())})
            customer_rows.append(row)
        initial = cohorts[(hotel, 1, 'Train')]
        windows = {'Initial train': initial, **{f'Fold {fold}': cohorts[(hotel, fold, 'Validation')]
                                                for fold in range(1, config['split']['n_folds'] + 1)}}
        for feature in ('EURUSD', 'LeadTime'):
            inner = np.unique(initial[feature].quantile([.25, .5, .75]).to_numpy())
            edges = np.r_[-np.inf, inner, np.inf]
            for window, cohort in windows.items():
                values = cohort[feature]
                if values.isna().any():
                    raise AssertionError('Missing numeric feature before fixed-bin diagnosis')
                distribution_rows.append({'Hotel': hotel, 'Feature': feature, 'Window': window,
                    'Minimum': float(values.min()), 'Maximum': float(values.max()),
                    'OutsideInitialRange': float((values.lt(initial[feature].min())
                                                  | values.gt(initial[feature].max())).mean()),
                    **{f'Q{int(q * 100)}': float(values.quantile(q)) for q in [.05, .25, .5, .75, .95]}})
                assigned = pd.cut(values, bins=edges, labels=False, include_lowest=True) + 1
                table = cohort.assign(ValueGroup=assigned).groupby('ValueGroup').agg(
                    Rows=('IsCanceled', 'size'), Positives=('IsCanceled', 'sum'))
                table = table.reindex(range(1, len(edges)), fill_value=0).reset_index()
                table['CancelRate'] = table.Positives / table.Rows.replace(0, np.nan)
                table['Share'] = table.Rows / len(cohort)
                detail_rows.append(table.assign(Hotel=hotel, Feature=feature, Window=window))
    return {'train_validation.csv': metrics[['Hotel', 'Variant', 'Fold', 'TrainLogLoss', 'LogLoss',
                                              'DummyTrainLogLoss', 'DummyLogLoss', 'GeneralizationGap']],
            'cancellation_rates.csv': pd.DataFrame(rate_rows),
            'probability_rates.csv': metrics[['Hotel', 'Variant', 'Fold', 'ValidationCancelRate',
                                               'MeanProbability', 'ProbabilityBias']],
            'reliability.csv': reliability(oof, groups=config['diagnostics']['score_groups']),
            'feature_shift.csv': pd.DataFrame(shift_rows),
            'numeric_groups.csv': pd.concat(detail_rows, ignore_index=True),
            'numeric_distributions.csv': pd.DataFrame(distribution_rows),
            'transient.csv': pd.DataFrame(customer_rows)}
