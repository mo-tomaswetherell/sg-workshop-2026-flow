import re
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import xarray as xr

# YYYY-MM-DD with optional -HH:MM. Allows any four digit year, but checks
# month/date/hour/minute ranges. Uses day<=31 (irrespective of month)
_DATE_PATTERN = re.compile(
    r"^\d{4}-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])(-([01]\d|2[0-3]):[0-5]\d)?$"
)

# strftime format
_DAY_FORMAT = "%Y-%m-%d"
_MINUTE_FORMAT = "%Y-%m-%d-%H:%M"
_DAY_LEN = len("YYYY-MM-DD")

@dataclass(frozen=True)
class DateRange:
    """Represents a discrete temporal window for dataset filtering

    Attributes:
        start (str): The opening bound of the time window.
            Must match format: 'YYYY-MM-DD' or 'YYYY-MM-DD-HH:MM'.
        end (str): The closing bound of the time window (inclusive).
            Must match format: 'YYYY-MM-DD' or 'YYYY-MM-DD-HH:MM'.
    """
    start: str
    end: str

    def __post_init__(self) -> None:
        """Validate string formats and chronological order upon instantiation."""
        # Validate structural format
        for name, value in (("start", self.start), ("end", self.end)):
            if not isinstance(value, str):
                raise ValueError(
                    f"Invalid {name} date format: '{value}'. "
                    "Expected a string."
                )
            if not _DATE_PATTERN.match(value):
                raise ValueError(
                    f"Invalid {name} date format: '{value}'. "
                    "Expected 'YYYY-MM-DD' or 'YYYY-MM-DD-HH:MM'."
                )

        # Prevent mixing granularities (e.g., comparing a day to a specific hour)
        if len(self.start) != len(self.end):
            raise ValueError(
                f"Mismatched date granularities: '{self.start}' and '{self.end}'. "
                "Both bounds must use the exact same format string."
            )
        # Validate chronological order using string comparison.
        # Because the date format goes from largest unit to smallest (Year -> Month -> Day -> Hour),
        # standard string sorting (lexical comparison) naturally matches chronological sorting, and is calendar agnostic. 
        if self.start > self.end:
            raise ValueError(
                f"Chronological error: Start date '{self.start}' "
                f"cannot be after end date '{self.end}'."
            )

    @property
    def strftime_format(self) -> str:
        """strftime format matching this range's granularity."""
        return _DAY_FORMAT if len(self.start) == _DAY_LEN else _MINUTE_FORMAT


def _build_time_mask(time: xr.DataArray, ranges: Sequence[DateRange]) -> np.ndarray:
    """Boolean mask over `time`. True where time falls in any of the ranges (inclusive)."""
    mask = np.zeros(time.size, dtype=bool)
    labels: dict[str, np.ndarray] = {}
    for r in ranges:
        if r.strftime_format not in labels:
            labels[r.strftime_format] = time.dt.strftime(r.strftime_format).values.astype(str)
        s = labels[r.strftime_format]
        mask |= (s >= r.start) & (s <= r.end)
    return mask

def select_date_ranges(ds: xr.Dataset, ranges: Sequence[tuple[str, str]]) -> xr.Dataset:
    """Returns `ds` restircted to timesteps inside any of the [start, end] ranges.

    Raises:
        ValueError: If a range is incorrectly specified or no timesteps fall within the ranges.
    """
    date_ranges = [DateRange(start, end) for start, end in ranges]
    mask = _build_time_mask(ds["time"], date_ranges)
    if not mask.any():
        raise ValueError(f"No timesteps fall within the date ranges: {list(ranges)}")
    return ds.isel(time=np.flatnonzero(mask))
