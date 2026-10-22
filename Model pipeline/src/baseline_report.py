"""Notebook displays and English observations, sourced exclusively from current outputs."""
import json

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display

import common

METRICS = ['LogLoss', 'BrierScore', 'ROC_AUC', 'AveragePrecision']


def read(flow, stage, filename):
    return common.read_tables(flow.output_root / stage)[filename]


def hotels(flow, stage, filename):
    return pd.concat([read(flow, f'{stage}/{hotel}', filename)
                      for hotel in flow.config['hotels']], ignore_index=True)


def show(frame):
    display(frame.style.format(precision=4, na_rep='—').hide(axis='index'))


def fold_view(metrics, value='LogLoss'):
    return metrics.pivot(index=['Hotel', 'Fold'], columns='Variant', values=value).reset_index()


def show_section(flow, section):
    if section == '0':
        show(read(flow, 'setup', 'config_summary.csv'))
        return
    if section == '2':
        show(read(flow, 'load', 'gold_summary.csv'))
        show(read(flow, 'load', 'held_out_calendar.csv'))
        return
    if section == '3':
        checks = read(flow, 'checks', 'split_checks.csv').iloc[0]
        print(f"{int(checks.ChecksPassed)} checks passed (checks are implemented in tests/).")
        cv = read(flow, 'split_summary', 'cv_summary.csv')
        show(cv[['Hotel', 'Fold', 'FitDate', 'TrainRows', 'TrainLabelsUnavailable',
                 'ValidationRows', 'ValidationPositives']])
        show(read(flow, 'split_summary', 'partition_summary.csv'))
    elif section == '4':
        show(read(flow, 'preprocessing', 'audit.csv'))
        show(read(flow, 'preprocessing', 'fallback_checks.csv'))
        return
    elif section in ('5.1', '5.3', '6'):
        stage = {'5.1': 'baseline', '5.3': 'calendar_comparison', '6': 'comparison'}[section]
        summary = hotels(flow, stage, 'cv_summary.csv')
        show(summary[['Hotel', 'Variant'] + METRICS + ['PooledOOFLogLoss']])
        metrics = hotels(flow, stage, 'fold_metrics.csv')
        show(fold_view(metrics))
        if section == '5.1':
            print(f"{len(read(flow, 'metric_checks', 'checks.csv'))} known-answer metric checks passed")
        else:
            show(metrics[['Hotel', 'Variant', 'Fold', 'ValidationCancelRate', 'MeanProbability']])
        if section == '6':
            show(read(flow, 'alignment', 'cohort_alignment.csv'))
    elif section == '5.2':
        for filename in ('probability_rates.csv', 'arrival_cohorts.csv', 'feature_contributions.csv',
                         'score_reconciliation.csv', 'calendar_ranges.csv'):
            print(filename.removesuffix('.csv').replace('_', ' ').capitalize())
            show(read(flow, 'diagnosis', filename))
    elif section == '7':
        for filename in ('train_validation.csv', 'cancellation_rates.csv', 'probability_rates.csv'):
            print(filename.removesuffix('.csv').replace('_', ' ').capitalize())
            show(read(flow, 'diagnostics', filename))
        plot_reliability(flow, 'diagnostics', 'Variant')
        shifts = read(flow, 'diagnostics', 'feature_shift.csv')
        print('Numeric shift: |validation mean − train mean| / train SD. Categorical shift: total variation.')
        display(shifts.pivot(index=['Feature', 'Type'], columns=['Hotel', 'Fold'], values='Shift').style.format(precision=3))
        for filename in ('numeric_distributions.csv', 'numeric_groups.csv', 'transient.csv'):
            print(filename.removesuffix('.csv').replace('_', ' ').capitalize())
            show(read(flow, 'diagnostics', filename))
    elif section == '8':
        show(read(flow, 'tuning_summary', 'winners.csv'))
        show(fold_view(read(flow, 'tuning_summary', 'fold_metrics.csv')))
    elif section == '9':
        show(read(flow, 'feature_summary', 'cv_summary.csv'))
        metrics = hotels(flow, 'features', 'fold_metrics.csv')
        show(metrics[['Hotel', 'Variant', 'Fold', 'GainVsF0', 'ValidationCancelRate', 'MeanProbability', 'ProbabilityBias']])
    elif section == '10':
        show(read(flow, 'calibration', 'cv_summary.csv'))
        metrics = read(flow, 'calibration', 'fold_metrics.csv')
        show(metrics[['Hotel', 'Method', 'Fold', 'ValidationCancelRate', 'MeanProbability',
                      'LogLoss', 'GainVsRaw', 'ExactZeroRows']])
        plot_reliability(flow, 'calibration', 'Method')
    elif section == '11':
        show(hotels(flow, 'frozen_comparison', 'cv_summary.csv')[['Hotel', 'Variant'] + METRICS])
        show(fold_view(hotels(flow, 'frozen_comparison', 'fold_metrics.csv')))
        show(read(flow, 'frozen', 'hard_groups.csv'))
        show(read(flow, 'frozen', 'rehearsal_check.csv'))
    else:
        raise ValueError(section)
    write_observation(flow, section)


def plot_reliability(flow, stage, label):
    table = read(flow, stage, 'reliability.csv')
    fold_ids = sorted(table.Fold.unique())
    fig, axes = plt.subplots(len(flow.config['hotels']), len(fold_ids),
                              figsize=(4 * len(fold_ids), 7), squeeze=False, sharex=True, sharey=True)
    limit = min(1., max(table.MeanProbability.max(), table.ActualRate.max()) * 1.12)
    for row, hotel in enumerate(flow.config['hotels']):
        for col, fold in enumerate(fold_ids):
            ax = axes[row, col]
            ax.plot([0, limit], [0, limit], '--', color='gray', linewidth=1)
            panel = table.loc[table.Hotel.eq(hotel) & table.Fold.eq(fold)]
            for variant, group in panel.groupby(label, sort=False):
                ax.plot(group.MeanProbability, group.ActualRate, marker='o', label=variant)
            ax.set(title=f'{hotel} | Fold {fold}', xlim=(0, limit), ylim=(0, limit),
                   xlabel='Mean predicted probability', ylabel='Actual cancellation rate')
            ax.grid(alpha=.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(labels), frameon=False)
    fig.tight_layout(rect=[0, .06, 1, 1])
    fig.savefig(flow.output_root / stage / 'reliability.png', dpi=130)
    plt.show()
    plt.close(fig)


def markdown_table(table):
    def cell(value):
        if pd.isna(value):
            return '—'
        if isinstance(value, float):
            return f'{value:.4f}'
        return str(value).replace('|', '/')
    header = '| ' + ' | '.join(str(c) for c in table.columns) + ' |'
    separator = '| ' + ' | '.join('---' for _ in table.columns) + ' |'
    rows = ['| ' + ' | '.join(cell(value) for value in row) + ' |' for row in table.itertuples(index=False, name=None)]
    return '\n'.join([header, separator] + rows)


def one(table, **filters):
    for key, value in filters.items():
        table = table.loc[table[key].eq(value)]
    return table.iloc[0]


def observation(flow, section):
    paragraphs = ['**Observation & Decision**']
    if section == '3':
        cv = read(flow, 'split_summary', 'cv_summary.csv')
        starts = []
        for hotel, group in cv.groupby('Hotel'):
            group = group.sort_values('Fold')
            starts.append(f"{hotel}: {int(group.TrainRows.iloc[0]):,} → {int(group.TrainRows.iloc[-1]):,} eligible bookings")
        h1 = cv.loc[cv.Hotel.eq('H1')]
        paragraphs.append('Training expands across folds (' + '; '.join(starts) + '). '
            f"Unavailable labels exclude {int(cv.TrainLabelsUnavailable.min()):,}–{int(cv.TrainLabelsUnavailable.max()):,} candidates per window. "
            f"H1 validation sizes vary from {int(h1.ValidationRows.min()):,} to {int(h1.ValidationRows.max()):,} bookings; rank candidates by the equal-weight mean of the validation windows.")
        paragraphs.append('All later experiments keep these memberships. A known limitation is outcome-dependent selection: cancellations become observable earlier than check-outs. Its effect is examined in Section 5.2; a revised eligibility policy is deferred to Notebook 02.')
    elif section == '5.1':
        summary = hotels(flow, 'baseline', 'cv_summary.csv')
        metrics = hotels(flow, 'baseline', 'fold_metrics.csv')
        paragraphs.append(markdown_table(summary[['Hotel', 'Variant'] + METRICS]))
        aligned = metrics.loc[metrics.Variant.eq('LR21')].merge(
            metrics.loc[metrics.Variant.eq('Dummy')], on=['Hotel', 'Fold'], suffixes=('_lr', '_dummy'))
        auc_wins = int(aligned.ROC_AUC_lr.gt(aligned.ROC_AUC_dummy).sum())
        loss_worse = int(aligned.LogLoss_lr.gt(aligned.LogLoss_dummy).sum())
        row = one(aligned, Hotel='H1', Fold=1)
        paragraphs.append(f"LR ranks bookings better than Dummy in {auc_wins}/{len(aligned)} windows, but has worse log loss in {loss_worse}/{len(aligned)}. "
            f"In H1 fold 1, log loss is {row.LogLoss_lr:.4f} versus {row.LogLoss_dummy:.4f}. The ranking signal does not translate into useful probability levels.")
        paragraphs.append('**Decision:** diagnose the probability failure before trying another model family.')
    elif section == '5.2':
        rates = read(flow, 'diagnosis', 'probability_rates.csv')
        rates = rates.loc[rates.Variant.eq('LR21')]
        lines = []
        for hotel in flow.config['hotels']:
            row = one(rates, Hotel=hotel, Fold=1)
            lines.append(f"{hotel} fold 1 predicts {row.MeanProbability:.1%} against {row.ValidationCancelRate:.1%} observed")
        row = one(rates, Hotel='H2', Fold=2)
        paragraphs.append('; '.join(lines) + f". H2 fold 2 reverses direction ({row.MeanProbability:.1%} predicted versus {row.ValidationCancelRate:.1%} observed), so the error is not a fixed offset.")
        cohorts = read(flow, 'diagnosis', 'arrival_cohorts.csv')
        recent = pd.Timestamp(flow.config['split']['first_fit_date']).strftime('%Y-%m')
        paragraphs.append(markdown_table(cohorts.loc[cohorts.ArrivalMonth.eq(recent),
            ['Hotel', 'ArrivalMonth', 'EligibleRows', 'EligiblePositives', 'UnavailableRows']]))
        paragraphs.append('The newest arrival cohort contains only early observed outcomes; the eligible bookings shown above are all cancellations/no-shows. This is evidence of the eligibility limitation, not evidence about the eventual unavailable labels.')
        contribution = read(flow, 'diagnosis', 'feature_contributions.csv')
        for hotel, group in contribution.groupby('Hotel'):
            top = group.nlargest(3, 'MeanContributionShift')
            paragraphs.append(f"{hotel}'s largest upward log-odds shifts are " + ', '.join(
                f'{r.SourceFeature} ({r.MeanContributionShift:.3f})' for r in top.itertuples()) + '.')
        ranges = read(flow, 'diagnosis', 'calendar_ranges.csv')
        year = ranges.loc[ranges.Feature.eq('ArrivalDateYear')]
        if year.UnseenValidationValues.eq(0).all():
            paragraphs.append('Validation arrival years are already present in train; the evidence is a shift in calendar contributions, not an unseen-year claim.')
        paragraphs.append('**Decision:** test whether calendar terms learned a misleading pattern from the selected recent cohort. Remove Year, then Year/Month/Week, while keeping all other settings fixed. This is a hypothesis, not a causal conclusion.')
    elif section == '5.3':
        summary = hotels(flow, 'calendar_comparison', 'cv_summary.csv')
        metrics = hotels(flow, 'calendar_comparison', 'fold_metrics.csv')
        paragraphs.append(markdown_table(summary.pivot(index='Hotel', columns='Variant', values='LogLoss').reset_index()))
        h1 = metrics.loc[metrics.Hotel.eq('H1') & metrics.Fold.eq(1)].set_index('Variant')
        paragraphs.append(f"Removing Year reduces H1 fold-1 log loss from {h1.loc['LR21', 'LogLoss']:.4f} to {h1.loc['LR20', 'LogLoss']:.4f}, versus Dummy's {h1.loc['Dummy', 'LogLoss']:.4f}.")
        for hotel in flow.config['hotels']:
            rows = metrics.loc[metrics.Hotel.eq(hotel) & metrics.Variant.eq('LR18')]
            wins = ', '.join(str(v) for v in rows.loc[rows.GainVsDummy.gt(common.TOLERANCE), 'Fold'])
            paragraphs.append(f'{hotel}: LR18 beats Dummy in folds {wins}.')
        paragraphs.append('**Decision:** use the calendar-reduced feature set for all subsequent families, with LR18 as the reference. H2 still needs a more flexible model.')
    elif section == '6':
        summary = hotels(flow, 'comparison', 'cv_summary.csv')
        metrics = hotels(flow, 'comparison', 'fold_metrics.csv')
        paragraphs.append(markdown_table(summary.pivot(index='Hotel', columns='Variant', values='LogLoss').reset_index()))
        for hotel in flow.config['hotels']:
            row = one(summary, Hotel=hotel, Variant='RF18')
            paragraphs.append(f"{hotel}: RF18 beats Dummy in {int(row.FoldsBeatingDummy)}/{flow.config['split']['n_folds']} folds; its mean log loss is {row.LogLoss:.4f}.")
        xgb = one(metrics, Hotel='H1', Fold=4, Variant='XGB18')
        rf = metrics.loc[metrics.Variant.eq('RF18')]
        paragraphs.append(f"XGB18 H1 fold 4 predicts {xgb.MeanProbability:.1%} versus {xgb.ValidationCancelRate:.1%} observed. RF18 overpredicts in {int(rf.ProbabilityBias.gt(0).sum())}/{len(rf)} windows.")
        paragraphs.append('**Decision:** useful ranking still does not ensure useful probability levels, especially in H2. Examine overfitting, cancellation-rate drift, probability bias and feature drift before tuning.')
    else:
        return later_observation(flow, section)
    return '\n\n'.join(paragraphs) + '\n'


def write_observation(flow, section):
    text = observation(flow, section)
    common.write_json(flow.output_root / 'observations' / f'{section}.json',
                      {'run_id': flow.run_id, 'markdown': text})


def later_observation(flow, section):
    paragraphs = ['**Observation & Decision**']
    if section == '7':
        losses = read(flow, 'diagnostics', 'train_validation.csv')
        improved = int(losses.TrainLogLoss.lt(losses.DummyTrainLogLoss).sum())
        example = one(losses, Hotel='H1', Fold=4, Variant='XGB18')
        paragraphs.append(f"**Overfitting.** Models improve on Dummy on train in {improved}/{len(losses)} fits, but much of the tree models' advantage disappears on validation. XGB18 H1 fold 4 moves from {example.TrainLogLoss:.3f} train log loss to {example.LogLoss:.3f} on validation. Complexity needs control.")
        rates = read(flow, 'diagnostics', 'cancellation_rates.csv')
        rate_table = rates[['Hotel', 'Fold']].copy()
        rate_table['Train → validation'] = [f'{r.TrainCancelRate:.1%} → {r.ValidationCancelRate:.1%}' for r in rates.itertuples()]
        paragraphs.append('**Cancellation-rate drift.**\n\n' + markdown_table(rate_table))
        paragraphs.append('H1 changes direction between the early and later windows; H2 has a large first-window shift. A historical probability level does not match every later cohort.')
        bins = read(flow, 'diagnostics', 'reliability.csv')
        high = bins.loc[bins.Hotel.eq('H1') & bins.Fold.eq(4) & bins.Variant.eq('XGB18')].sort_values('ScoreGroup').iloc[-1]
        paragraphs.append(f"**Probability bias.** XGB18 H1 fold 4's highest-score group averages {high.MeanProbability:.1%} predicted risk but {high.ActualRate:.1%} observed cancellations. The reliability curves support trying temporal calibration.")
        groups = read(flow, 'diagnostics', 'numeric_groups.csv')
        low = groups.loc[groups.Feature.eq('EURUSD') & groups.Window.eq('Fold 4') & groups.ValueGroup.eq(1)]
        paragraphs.append('**Feature drift.** In the last validation window, the share in the lowest initial-train EURUSD group is '
            + ', '.join(f'{r.Hotel} {r.Share:.1%}' for r in low.itertuples())
            + '. LeadTime distributions and their within-group cancellation rates also change.')
        lead = groups.loc[groups.Hotel.eq('H2') & groups.Feature.eq('LeadTime')]
        initial = lead.loc[lead.Window.eq('Initial train')].set_index('ValueGroup')
        last = lead.loc[lead.Window.eq('Fold 4')].set_index('ValueGroup')
        paragraphs.append(f"H2's shortest/longest LeadTime groups change from {initial.loc[1, 'CancelRate']:.1%}/{initial.loc[4, 'CancelRate']:.1%} cancellations in initial train to {last.loc[1, 'CancelRate']:.1%}/{last.loc[4, 'CancelRate']:.1%} in the last window, reversing their risk order.")
        transient = read(flow, 'diagnostics', 'transient.csv').loc[lambda t: t.Hotel.eq('H2')]
        rises = int(transient.ValidationShare.gt(transient.TrainShare).sum())
        paragraphs.append(f"H2 Transient share is higher in validation than train in {rises}/{len(transient)} windows. These are descriptive changes, not causal effects.")
        paragraphs.append('**Decision:** tune complexity first, then ablate EURUSD and LeadTime, then calibrate the selected RF. Cancellation-rate drift remains open for Notebook 02.')
    elif section == '8':
        winners = read(flow, 'tuning_summary', 'winners.csv')
        paragraphs.append(markdown_table(winners[['Hotel', 'Variant', 'Parameters', 'LogLoss', 'InitialLogLoss', 'FoldsBeatingDummy']]))
        rf = one(winners, Hotel='H2', ModelKey='rf')
        paragraphs.append(f"H1's winning RF retains its initial complexity. H2's shallower RF improves mean log loss from {rf.InitialLogLoss:.4f} to {rf.LogLoss:.4f}, against Dummy's {rf.DummyLogLoss:.4f}, while its mean train/validation gap changes from {rf.InitialGap:.3f} to {rf.MeanGap:.3f}.")
        grid = read(flow, 'tuning/H2', 'cv_summary.csv')
        best = grid.loc[grid.ModelKey.eq('rf')].nsmallest(2, 'LogLoss')
        paragraphs.append('The two leading H2 RF configurations are nearly tied: ' + '; '.join(
            f'{r.GridParameters}: {r.LogLoss:.6f}' for r in best.itertuples()) + '. The selection rule chooses the lower mean; this small gap is not treated as meaningful evidence.')
        paragraphs.append('**Decision:** carry only the separately tuned RF for each hotel into feature ablation. XGB benefits from tuning, but remains behind RF; the H2 LR and XGB means remain behind Dummy.')
    elif section == '9':
        summary = read(flow, 'feature_summary', 'cv_summary.csv')
        metrics = hotels(flow, 'features', 'fold_metrics.csv')
        paragraphs.append(markdown_table(summary.pivot(index='Hotel', columns='Variant', values='LogLoss').reset_index()))
        for hotel in flow.config['hotels']:
            f0, f1 = one(summary, Hotel=hotel, Variant='F0'), one(summary, Hotel=hotel, Variant='F1')
            paragraphs.append(f"{hotel}: F1 improves on F0 in {int(f1.FoldsBeatingF0)}/{flow.config['split']['n_folds']} windows; mean absolute probability bias falls from {100 * f0.MeanAbsoluteWindowBias:.1f} to {100 * f1.MeanAbsoluteWindowBias:.1f} percentage points.")
        h2 = one(metrics, Hotel='H2', Variant='F1', Fold=1)
        paragraphs.append(f"H1's gain is concentrated in the last window; H2 improves consistently. Removing LeadTime alone worsens the mean, so it retains useful signal despite drift. H2 F1 fold 1 still overpredicts by {100 * h2.ProbabilityBias:.1f} percentage points.")
        paragraphs.append('**Decision:** select F1 (RF17). EURUSD acts as a time proxy and shifts toward the low end of its training distribution; the log-loss gain supports removing it. RF17 reuses hyperparameters selected on the preceding feature set. Try calibration to address the remaining probability bias.')
    elif section == '10':
        summary = read(flow, 'calibration', 'cv_summary.csv')
        metrics = read(flow, 'calibration', 'fold_metrics.csv')
        paragraphs.append(markdown_table(summary[['Hotel', 'Method', 'LogLoss', 'BrierScore', 'FoldsBeatingRaw', 'ExactZeroRows', 'CanceledAtZero']]))
        h2 = metrics.loc[metrics.Hotel.eq('H2') & metrics.Fold.eq(2)]
        actual = h2.ValidationCancelRate.iloc[0]
        paragraphs.append(f"Raw has the best mean log loss and Brier score for both hotels on the same later validation folds. In H2 fold 2, actual cancellations are {actual:.1%}; "
            + ', '.join(f'{r.Method} predicts {r.MeanProbability:.1%}' for r in h2.itertuples())
            + '. The calibrators learn from an earlier period and adjust toward a probability level that is no longer representative.')
        zero = summary.loc[summary.Method.eq('isotonic')]
        paragraphs.append('Isotonic assigns zero probability to ' + '; '.join(
            f'{int(r.ExactZeroRows):,} bookings in {r.Hotel}, including {int(r.CanceledAtZero):,} cancellations'
            for r in zero.itertuples()) + ', causing large log-loss penalties.')
        paragraphs.append('**Decision:** retain raw probabilities. A mapping learned from earlier OOF periods does not resolve the changing probability bias; this remains open for Notebook 02.')
    elif section == '11':
        summary = hotels(flow, 'frozen_comparison', 'cv_summary.csv')
        settings = read(flow, 'tuning_summary', 'winners.csv').loc[lambda t: t.ModelKey.eq('rf')].set_index('Hotel').Parameters
        summary['Parameters'] = summary.Hotel.map(settings)
        summary.loc[summary.Variant.eq('Dummy'), 'Parameters'] = '—'
        summary['Variant'] = summary.Variant.replace({'F1': 'RF17 raw'})
        paragraphs.append(markdown_table(summary[['Hotel', 'Variant', 'Parameters'] + METRICS]))
        f1 = summary.loc[summary.Variant.eq('RF17 raw')]
        paragraphs.append('RF17 beats Dummy in ' + '; '.join(
            f'{int(r.FoldsBeatingDummy)}/{flow.config["split"]["n_folds"]} windows for {r.Hotel}'
            for r in f1.itertuples()) + ', but the mean gains remain modest, especially in H2.')
        groups = read(flow, 'frozen', 'hard_groups.csv')
        largest = groups.sort_values('CVExcessContribution', ascending=False).iloc[0]
        paragraphs.append(f"The largest supported excess-loss contribution is {largest.Hotel} fold {int(largest.Fold)}, {largest.Dimension}={largest.Group}: {largest.CancelRate:.1%} observed versus {largest.MeanProbability:.1%} predicted. These segment results are diagnostic, not a new selection exercise.")
        rehearsal = read(flow, 'frozen', 'rehearsal_check.csv')
        paragraphs.append(f"The last-validation-fold refit rehearsal reproduces CV metrics with maximum absolute difference {rehearsal.MaxAbsDifference.max():.2e}.")
        paragraphs.append('**Decision:** freeze RF17 raw as the development baseline for Notebook 02, using the selected per-hotel settings shown in Section 8. No held-out outcomes or predictions enter this decision.')
        paragraphs.append('**Known limitations:** eligibility depends on observed outcome; cancellation-rate and feature drift remain unresolved; daily aggregate cancellation error and the information ceiling have not been measured. Repeated choices on the same validation folds make CV somewhat optimistic. Independent held-out evaluation belongs to Notebook 03.')
    else:
        raise ValueError(section)
    return '\n\n'.join(paragraphs) + '\n'


def show_parity(flow):
    print('Development replay only. Each maximum is taken over matching folds/bookings; tolerance = 1e-4.')
    print('The displayed means use the named metric; the maximum covers every metric in that check. Detailed values: outputs/baseline01/parity/comparison.csv.')
    display(parity_style(flow))


def parity_style(flow):
    details = read(flow, 'parity', 'comparison.csv')
    rows = []
    for (stage, name), group in details.groupby(['Stage', 'Check'], sort=False):
        selected = group.loc[group.Metric.eq('LogLoss')]
        selected = (selected if not selected.empty else group).iloc[0]
        numeric = selected.Metric not in ('Membership', 'Selection')
        rows.append({'Stage': stage, 'Check': name, 'DisplayedMetric': selected.Metric,
                     'RerunMean': selected.RerunMean if numeric else float('nan'),
                     'HistoricalMean': selected.HistoricalMean if numeric else float('nan'),
                     'MetricsCompared': len(group), 'MaxAbsDifference': group.MaxAbsDifference.max(),
                     'Pass': bool(group.Pass.all())})
    table = pd.DataFrame(rows)
    return table.style.format({'RerunMean': '{:.8f}', 'HistoricalMean': '{:.8f}',
                                'MaxAbsDifference': '{:.2e}'}, na_rep='—').hide(axis='index')
