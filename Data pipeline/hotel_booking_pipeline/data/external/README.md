# Approved external sources

Keep original downloads unchanged. `source_manifest.json` records the source URLs, retrieval times and SHA-256 hashes. Only these three sources enter the active pipeline.

| Folder | Source | Use |
| --- | --- | --- |
| `holidays/` | Nager.Date Portugal, 2015–2017 | Reviewed national calendar: 9 / 13 / 13 holidays. Four suspended dates removed for 2015. |
| `exchange_rates/` | ECB D.USD.EUR.SP00.A; 939 observations | USD per EUR; latest observation strictly before CutoffDate. |
| `google_trends/` | Fixed worldwide hotel + flight search comparison | H1: Algarve hotels / flights to Faro. H2: Lisbon hotels / flights to Lisbon. |

Holiday correction evidence: [Law 23/2012](https://diariodarepublica.pt/dr/detalhe/lei/23-2012-178501) and [Law 8/2016](https://diariodarepublica.pt/dr/detalhe/lei/8-2016-74007209). Saved HTML is a page shell; the rendered law text was checked online. The implemented correction is in `utils/data_processing_silver_features.py` relative to the project root. See [README.ipynb](../../README.ipynb) for the processing rules and execution order.

Trends retains 138 full weeks per hotel from 140 raw weeks. Use the latest full week with week end + 7 days <= CutoffDate. The seven-day buffer is an assumption. The retrospective 0–100 export is not an original historical vintage; differently scaled exports cannot be combined directly.

All required downloads are supplied; rebuilding Bronze-to-Gold works offline. Use the manifest URLs and hashes to verify or retrieve sources. Do not silently replace the fixed snapshots with current downloads. Google Trends remains the fixed user-downloaded export.
