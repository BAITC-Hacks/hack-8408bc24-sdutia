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

The two February issues each contain a 48-hour window, two versions, three entities
and four weather models. Version 1 has 48 rows per entity. Version 2 is computed
six hours later and has only the 42 remaining hours (lead 6–47); it never revises
past targets. Every forecast row includes `version_as_of_utc`. Lead hours and
product names stay relative to the original issue time.

Weather rows include `version` and supply values at both H and H+1 for each target
hour H: 49 timestamps per model in version 1 and 43 in version 2, including the
endpoint at original issue +48 hours. Each run therefore has 270 forecast rows and
368 weather rows. Revision MAE is calculated only over matching target timestamps.
Other run KPIs describe the latest version's remaining 42 hours.

The submission is deliberately a 48-hour subset of February.
Validation has three separate synthetic October–December issues. Validation
metrics and run KPIs are calculated from those synthetic rows. The leave-one-
turbine-out values are illustrative fixture calculations, not trained-model results.

All timestamps use UTC internally and fixed UTC+6 / UTC+5 display clocks. `run.json`
and `metrics.json` contain `synthetic_fixture: true`; each analysis starts with the
same marker in an HTML comment and explicitly labels its content.
