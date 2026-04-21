# Patch v2

- Added `Jinja2>=3.1` to `requirements.txt`.
- Made LaTeX table export resilient: if optional pandas/Jinja2 LaTeX path fails, the run still completes and writes a basic `.tex` table.
- No changes to F1/F2 research contracts. This is a code/export-layer fix only.
