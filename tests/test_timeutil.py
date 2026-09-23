import pandas as pd
import pytest

from windagent import timeutil
from windagent.errors import InputError


def test_known_scada_row_maps_to_utc():
    # Data clock is a fixed UTC+6: 2024-03-01 00:00 on the logger is 2024-02-29 18:00 UTC.
    utc = timeutil.parse_clock_time("2024-03-01 00:00", 6)
    assert timeutil.iso_z(utc) == "2024-02-29T18:00:00Z"


def test_round_trip_data_clock_utc_official():
    utc = timeutil.parse_clock_time("2026-02-10 00:00", 6)
    assert timeutil.clock_str(utc, 6) == "2026-02-10 00:00"
    assert timeutil.clock_str(utc, 5) == "2026-02-09 23:00"  # official Kazakhstan time
    assert timeutil.clock_str(utc, 0) == "2026-02-09 18:00"


def test_issue_id_round_trip_has_no_colons():
    utc = timeutil.parse_clock_time("2026-01-31 00:00", 6)
    iid = timeutil.issue_id(utc, 6)
    assert iid == "2026-01-31_0000" and ":" not in iid
    assert timeutil.issue_time_from_id(iid, 6) == utc


def test_naive_timestamps_are_rejected():
    with pytest.raises(ValueError):
        timeutil.to_utc(pd.Timestamp("2026-02-01 00:00"))


@pytest.mark.parametrize("bad", ["", "31.01.2026", "2026-13-01 00:00", "2026-02-30", "tomorrow"])
def test_bad_user_times_raise_input_error(bad):
    with pytest.raises(InputError):
        timeutil.parse_clock_time(bad, 6)
