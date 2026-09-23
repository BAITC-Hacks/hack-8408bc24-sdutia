# Black-box CLI QA

These tests launch the real `python -m windagent ...` in child processes and inspect exit codes, console messages and output files. They do not import or replace application functions. The public expectations come from [CONTRACTS.md §6](../../docs/CONTRACTS.md#6-cli-python--m-windagent-command).

From the repository root, with the dependencies in `requirements.txt` installed:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/qa -q -ra --basetemp=.venv/qa-test-temp
```

Every child sets `WINDAGENT_OFFLINE=1`, disables LLM providers and clears API keys. A Python audit hook also blocks socket connection and DNS attempts, logging only event names in the test's temporary directory. This allows the suite to detect commands that ignore offline mode without contacting external services. This hook changes only the network boundary, not application calculations or data.

Each test writes to its own temporary output directory. The configured data, model and site files come from this checkout. The tests do not regenerate committed results, train models, call `report`, or run a clean-clone/full-period verification. The short backtest intentionally covers only a small portion of the period. The evaluation input is synthetic with a known mathematical error and is not evidence of real forecast accuracy.

## Known defects

Reproduced contract failures are tracked in [docs/requests/codex-qa.md](../../docs/requests/codex-qa.md). Tests for those defects use `xfail(strict=True)` with the corresponding defect ID. These are reported as expected failures, not passes. A fix that unexpectedly passes requires removing its marker, so a resolved defect cannot remain silently marked.

To see every known defect as an ordinary failing assertion:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/qa -q --runxfail --basetemp=.venv/qa-strict-temp
```

The supplied `--basetemp` directories are disposable and pytest clears them on reuse. Use these dedicated paths rather than a directory containing personal files.

Normal input and missing-data errors check the documented single-line `ERROR: ...` format and absence of a traceback. A separate test checks explicit `--debug`. External-service failure code `4` is not independently simulated: this suite operates offline and blocks network access. It checks the offline missing-data path instead. Expensive training and full walk-forward validation are outside this suite.
