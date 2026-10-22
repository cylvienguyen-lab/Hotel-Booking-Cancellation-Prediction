# Data pipeline working rules

Read `README.ipynb` and `datamart/gold/feature_contract.json` before changing the pipeline.

- Keep the delivery scope at Bronze → Silver → Gold, with data-quality checks and the two reference EDA notebooks.
- Follow Assignment 1's module structure, simple `process_...()` functions and short English comments. No `main.py` or extra orchestration layer is required.
- Preserve raw sources, original column names, supplied Gold and the Assignment 1 Docker files unless the user requests a change.
- Keep H1/H2 outputs separate and one row per source booking. Preserve identical-looking records and stable `Booking_ID` values.
- Apply the agreed end-of-day T-14 filter and positive guests/nights scope. Retained booking details are assumed unchanged before the cutoff.
- Keep the agreed feature, metadata and label groups in the feature contract. Final status/date remain in the separate label store for audit.
- Keep `EstimatedRoomCount` as an assumed reporting estimate: H1 uses 2 people per room and H2 uses 4.
- Preserve the approved holiday, EUR/USD and fixed Google Trends sources, their join rules and documented limitations.
- `EDA/` contains exactly one Alex notebook and one Helen notebook. Preserve the originals; use current processing rules and Gold for the implemented data scope.
- `README.ipynb` is the execution entry point and includes compact output checks. Keep the guide and execution order together; do not add separate documentation or execution folders.
- Run processing only when requested or required by a processing change; do not rerun for documentation-only edits.
- Excel files are static human references. Refresh them separately after a data rebuild.
- `diagrams/` contains only the latest data pipeline PNG. Do not resume online layout edits without a user request.
