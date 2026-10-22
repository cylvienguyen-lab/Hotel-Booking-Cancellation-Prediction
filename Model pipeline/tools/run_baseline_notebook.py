"""Execute Notebook 01 and refresh its markdown observations from this run's outputs.

Run inside the existing Jupyter container: python tools/run_baseline_notebook.py
"""
import json
import os
import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / 'model_train_main.ipynb'
OUTPUTS = ROOT / 'outputs' / 'baseline01'


def refresh_observations(notebook):
    manifest = OUTPUTS / 'run_manifest.json'
    if not manifest.exists():
        return
    run_id = json.loads(manifest.read_text(encoding='utf-8'))['run_id']
    for cell in notebook.cells:
        section = cell.metadata.get('baseline_observation')
        if section:
            path = OUTPUTS / 'observations' / f'{section}.json'
            if path.exists():
                document = json.loads(path.read_text(encoding='utf-8'))
                if document['run_id'] == run_id:
                    cell.source = document['markdown']


def main():
    os.chdir(ROOT)
    parser = argparse.ArgumentParser()
    parser.add_argument('--refresh-observations', action='store_true',
                        help='Refresh English markdown from a completed run; fit nothing and read outputs only')
    args = parser.parse_args()
    if args.refresh_observations:
        manifest = json.loads((OUTPUTS / 'run_manifest.json').read_text(encoding='utf-8'))
        if 'frozen' not in manifest['completed']:
            raise RuntimeError('A completed baseline run is required before refreshing observations')
        sys.path.insert(0, str(ROOT / 'src'))
        import baseline_report
        flow = SimpleNamespace(output_root=OUTPUTS, run_id=manifest['run_id'],
                               config=json.loads((OUTPUTS / 'run_config.json').read_text(encoding='utf-8')))
        checks = baseline_report.read(flow, 'parity', 'comparison.csv')
        if not (checks.Pass.all() and checks.MaxAbsDifference.max() <= 1e-4
                and checks.RunID.eq(flow.run_id).all()):
            raise RuntimeError('A passing comparison for the same run is required; stop and inspect the failure')
        notebook = nbformat.read(NOTEBOOK, as_version=4)
        for cell in notebook.cells:
            section = cell.metadata.get('baseline_observation')
            if section:
                baseline_report.write_observation(flow, section)
            if cell.metadata.get('baseline_section') == 'parity':
                cell.outputs = [nbformat.v4.new_output('display_data', data={
                    'text/html': baseline_report.parity_style(flow).to_html(),
                    'text/plain': 'Historical replay comparison; max differences shown in scientific notation.'})]
        refresh_observations(notebook)
        nbformat.write(notebook, NOTEBOOK)
        print('Refreshed observations from completed outputs; no models or Gold data were accessed.')
        return
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            cell.outputs = []
            cell.execution_count = None
            cell.metadata.pop('execution', None)
        elif cell.metadata.get('baseline_observation'):
            cell.source = '**Observation & Decision**\n\nPending the current run.'

    def completed(**kwargs):
        index = kwargs['cell_index']
        cell = kwargs['cell']
        section = cell.metadata.get('baseline_section', str(index))
        refresh_observations(notebook)
        nbformat.write(notebook, NOTEBOOK)
        failed = any(o.get('output_type') == 'error' for o in cell.get('outputs', []))
        print(f'{"Failed" if failed else "Completed"} section {section} (cell {index})', flush=True)
        for output in cell.get('outputs', []):
            if output.get('output_type') == 'stream':
                lines = output.text.strip().splitlines()
                if lines and ('parity' in lines[-1] or 'checks passed' in lines[-1]):
                    print(lines[-1], flush=True)

    client = NotebookClient(notebook, timeout=3600, kernel_name='python3',
                            resources={'metadata': {'path': str(ROOT)}},
                            on_cell_executed=completed)
    try:
        client.execute()
    finally:
        refresh_observations(notebook)
        nbformat.write(notebook, NOTEBOOK)
    print('Notebook 01 complete: development experiments and fold-4 rehearsal only.', flush=True)


if __name__ == '__main__':
    main()
