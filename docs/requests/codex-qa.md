# QA requests to the core engineer

Tested on 2026-09-23 in the QA checkout, Python 3.12.14 with the pinned project requirements. Core baseline: `5d4b25c`; presentation-only commit: `b4bcfc7`. Core source files were not changed by QA.

Every subprocess uses `WINDAGENT_OFFLINE=1`, `LLM_PROVIDER=none`, empty API keys and isolated output directories. A socket audit hook prevents external communication and records any attempted connection or DNS lookup. The guard-induced exception is not itself a product defect; the attempted network access is the evidence for the offline findings.

The regression cases below are marked `xfail(strict=True)` until the core owner resolves them. Run with `--runxfail` to see the original failing expectations. All reproduction commands below assume the QA checkout and its installed `.venv`:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/qa -q -ra --basetemp=.venv/qa-test-temp
.\.venv\Scripts\python.exe -m pytest tests/qa -q --runxfail --basetemp=.venv/qa-strict-temp
```

The temporary directories are dedicated disposable test output. No real API endpoint needs to be contacted to reproduce these findings.

## Verification record

- Error-contract suite: **20 passed, 13 expected failures** across the six defect categories below. The original strict run reproduced these failures before markers were added.
- Workflow suite: **6 passed**. Covered offline help/info, saved forecast numerical parity, disclosed LLM fallback, a short rolling backtest, a custom forecast horizon and evaluation using synthetic actuals with a known error (including file paths with spaces).
- Expected failures do not establish correctness of the affected commands. Their markers use `raises=AssertionError`; harness failures such as timeouts remain ordinary failures.
- Not repeated: clean-clone verification, AI-judge review, full retraining or full-period validation. The restart handoff assigns those to the core engineer or says to skip them.

## CLI-ERROR-FORMAT — parser errors bypass the documented message format

- **Severity:** minor.
- **What:** argument parsing for `python -m windagent`, unknown command, missing option value, nonnumeric horizon and unknown policy. Contract: `docs/CONTRACTS.md` §6.
- **Expected:** exit `2`, one stderr line beginning `ERROR: `, no traceback unless `--debug`.
- **Actual:** exit `2`, but argparse emits multiple usage lines followed by lowercase `windagent: error: ...`. Handled domain errors do follow the contract.
- **Reproduce:** `.\.venv\Scripts\python.exe -m pytest tests/qa/test_cli_errors.py -q --runxfail -k parser_input_errors --basetemp=.venv/qa-parser-temp`.
- **Example underlying command:** `python -m windagent forecast --issue "2026-02-05 00:00" --horizon abc`.
- **Core owner resolution:** pending.

## CLI-BACKTEST-DATE — malformed dates return an unexpected-error code

- **Severity:** major.
- **What:** `backtest --start` and `--end` date validation.
- **Expected:** invalid dates return exit `2` with a useful input-error message.
- **Actual:** malformed start/end dates and an impossible calendar date return exit `1`, for example `ERROR: unexpected DateParseError: Unknown datetime string format, unable to parse: not-a-date`.
- **Reproduce:** `.\.venv\Scripts\python.exe -m pytest tests/qa/test_cli_errors.py -q --runxfail -k backtest_invalid_dates --basetemp=.venv/qa-dates-temp`.
- **Example underlying command:** `python -m windagent backtest --start not-a-date --end 2026-02-05 --offline --policy rules`.
- **Core owner resolution:** pending.

## CLI-ISSUE-NOW — conflicting forecast modes are accepted

- **Severity:** major.
- **What:** `forecast --issue ... --now` despite the exclusive alternatives in the command contract.
- **Expected:** reject the combination before loading data or contacting services, exit `2`, explain the conflicting flags.
- **Actual:** the command chooses live mode and reaches a network lookup instead of reporting the conflict. The test guard blocks it. This also exposes CLI-OFFLINE-NOW below.
- **Reproduce safely:** `.\.venv\Scripts\python.exe -m pytest tests/qa/test_cli_errors.py -q --runxfail -k issue_and_now --basetemp=.venv/qa-conflict-temp`.
- **Underlying arguments:** `forecast --issue "2026-02-05 00:00" --now --offline --policy rules`.
- **Core owner resolution:** pending.

## CLI-OFFLINE-NOW — live forecasting ignores offline mode

- **Severity:** major.
- **What:** `forecast --now --offline --policy rules` with `WINDAGENT_OFFLINE=1`.
- **Expected:** no network attempts. Use usable cached data or return a documented data-unavailable error.
- **Actual:** the subprocess attempts `socket.getaddrinfo`. The audit guard blocks it before any external request. With the guard, the command ends in an unexpected-error response; that response is an effect of the harness, not the finding.
- **Reproduce safely:** `.\.venv\Scripts\python.exe -m pytest tests/qa/test_cli_errors.py -q --runxfail -k now_offline --basetemp=.venv/qa-now-temp`.
- **Impact:** an operator cannot rely on the documented offline flag for this command.
- **Core owner resolution:** pending.

## CLI-MISSING-DATA — absent SCADA or weather cache causes an unexpected failure

- **Severity:** major.
- **What:** historical forecast with either raw SCADA absent and cached weather present, or raw SCADA present and the weather cache absent. Both cases use an isolated `WINDAGENT_DATA_DIR`.
- **Expected:** exit `3`, a clear missing-SCADA or missing-weather-cache message, no traceback. Offline mode should not imply an external service was contacted.
- **Actual:** exit `1`, `ERROR: unexpected ValueError: No objects to concatenate`.
- **Reproduce:** `.\.venv\Scripts\python.exe -m pytest tests/qa/test_cli_errors.py -q --runxfail -k "missing_scada or missing_weather" --basetemp=.venv/qa-data-temp`.
- **Underlying arguments:** `forecast --issue "2026-02-05 00:00" --offline --policy rules`, with the partially populated temporary data directory supplied by the test. Neither reproduction attempts a network connection.
- **Core owner resolution:** pending.

## CLI-OUTPUTS-DIR — documented backtest option is missing

- **Severity:** major.
- **What:** `backtest --outputs-dir DIR`, explicitly listed in `docs/CONTRACTS.md` §6.
- **Expected:** run the backtest successfully and place its output only in the requested directory.
- **Actual:** exit `2`, argparse rejects `--outputs-dir` as unrecognized. The `WINDAGENT_OUTPUTS_DIR` environment override does work in the passing workflow tests.
- **Reproduce:** `.\.venv\Scripts\python.exe -m pytest tests/qa/test_cli_errors.py -q --runxfail -k outputs_dir_flag --basetemp=.venv/qa-output-dir-temp`.
- **Underlying arguments:** `backtest --start 2026-02-05 --end 2026-02-05 --policy rules --offline --outputs-dir <temporary-directory>`.
- **Core owner resolution:** pending.
