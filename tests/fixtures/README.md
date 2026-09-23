# Synthetic UI fixtures

Every value in this directory is synthetic. These files exercise presentation,
clock conversion, uncertainty bands, revisions, metrics, and empty-state handling;
they are not real weather, model predictions, or evidence of forecast accuracy.

Regenerate the committed output deterministically:

```powershell
python tests/fixtures/make_fixtures.py
```

Use fixtures only through the public API by explicitly setting the output root:

```powershell
$env:WINDAGENT_OUTPUTS_DIR = "tests/fixtures/outputs"
$env:WINDAGENT_OFFLINE = "1"
streamlit run app/streamlit_app.py
```

Unset `WINDAGENT_OUTPUTS_DIR` to return to normal outputs. Do not copy fixtures
into the real output directory. The generator uses only Python's standard library
and writes only under this directory, never into production outputs.

The two February issues each contain 48 hours, two versions, three entities and
four weather models. The submission is deliberately a 48-hour subset of February.
Validation has three separate synthetic October–December issues. Validation
metrics and run KPIs are calculated from those synthetic rows. The leave-one-
turbine-out values are illustrative fixture calculations, not trained-model results.

All timestamps use UTC internally and fixed UTC+6 / UTC+5 display clocks. `run.json`
and `metrics.json` contain `synthetic_fixture: true`; each analysis starts with the
same marker in an HTML comment and explicitly labels its content.
