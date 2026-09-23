<!-- synthetic_fixture: true; UI testing only, not forecast evidence. -->

### Synthetic fixture / Синтетический пример

These deterministic values exercise the UI. They do not represent measured weather, real model forecasts, validation skill, or a completed agent run.

- Version 1 covers all 48 hours for farm, t1 and t2.
- Version 2 is issued six hours later and covers only the 42 remaining hours.
- Version as-of timestamps are explicit; lead hours remain relative to the original issue.
- Four synthetic weather models have versioned cached inputs at H and H+1.
- Normalized power has no MW conversion because rated capacity is unset.
