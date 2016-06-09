import numpy as np
import pandas as pd

from toolz import groupby, merge

from .base import PipelineLoader
from .frame import DataFrameLoader
from zipline.pipeline.common import (
    EVENT_DATE_FIELD_NAME,
    SID_FIELD_NAME,
    TS_FIELD_NAME,
)
from zipline.pipeline.loaders.utils import (
    next_event_indexer,
    previous_event_indexer,
)

WRONG_COLS_ERROR = "Expected columns {expected_columns} for but " \
                   "got columns {resulting_columns}."

WRONG_SINGLE_COL_DATA_FORMAT_ERROR = ("Data for sid {sid} is expected to have "
                                      "1 column and to be in a DataFrame, "
                                      "Series, or DatetimeIndex.")

WRONG_MANY_COL_DATA_FORMAT_ERROR = ("Data for sid {sid} is expected to have "
                                    "more than 1 column and to be in a "
                                    "DataFrame.")

SERIES_NO_DTINDEX_ERROR = ("Got Series for sid {sid}, but index was not "
                           "DatetimeIndex.")

DTINDEX_NOT_INFER_TS_ERROR = ("Got DatetimeIndex for sid {sid}.\n"
                              "Pass `infer_timestamps=True` to use the first "
                              "date in `all_dates` as implicit timestamp.")

DF_NO_TS_NOT_INFER_TS_ERROR = ("Got DataFrame without a '{"
                               "timestamp_column_name}' column for sid {sid}."
                               "\nPass `infer_timestamps=True` to use the "
                               "first date in `all_dates` as implicit "
                               "timestamp.")


class EventsLoader(PipelineLoader):
    """
    Base class for PipelineLoaders that supports loading the next and previous
    value of an event field.

    Does not currently support adjustments.

    Parameters
    ----------
    events : pd.DataFrame
        A DatetimeIndexed DataFrame representing events (e.g. share buybacks or
        earnings announcements) associated with particular companies.

        ``events`` must contain at least three columns::
            sid : int64
                The asset id associated with each event.

            event_date : datetime64[ns]
                The date on which the event occurred.

            timestamp : datetime64[ns]
                The date on which we learned about the event. Must be greater
                than or equal to the frame index value.

    all_dates : pd.DatetimeIndex
        Index of dates for which we can serve queries.

    dataset : DataSet
        The DataSet object for which this loader loads data.
    """
    def __init__(self, events, next_value_columns, previous_value_columns):
        # We always work with entries from ``events`` directly as numpy arrays,
        # so we coerce from a frame here.
        self.events = {
            name: np.asarray(series)
            for name, series in events.sort(EVENT_DATE_FIELD_NAME).iteritems()
        }

        # Columns to load with self.load_next_events.
        self.next_value_columns = next_value_columns

        # Columns to load with self.load_previous_events.
        self.previous_value_columns = previous_value_columns

    def split_next_and_previous_event_columns(self, requested_columns):
        """
        Split requested columns into columns that should load the next known
        value and columns that should load the previous known value.
        """
        def next_or_previous(c):
            if c in self.next_value_columns:
                return 'next'
            elif c in self.previous_value_columns:
                return 'previous'
            raise ValueError(
                "{c} not found in next_value_columns "
                "or previous_value_columns".format(c=c)
            )
        groups = groupby(next_or_previous, requested_columns)
        return groups['next'], groups['previous']

    def next_event_indexer(self, dates, sids):
        return next_event_indexer(
            dates,
            sids,
            self.events[EVENT_DATE_FIELD_NAME],
            self.events[TS_FIELD_NAME],
            self.events[SID_FIELD_NAME],
        )

    def previous_event_indexer(self, dates, sids):
        return previous_event_indexer(
            dates,
            sids,
            self.events[EVENT_DATE_FIELD_NAME],
            self.events[TS_FIELD_NAME],
            self.events[SID_FIELD_NAME],
        )

    def load_next_events(self, columns, dates, sids, mask):
        if not columns:
            return {}

        return self._load_events(
            name_map=self.next_value_columns,
            indexer=self.next_event_indexer(dates, sids),
            columns=columns,
            dates=dates,
            sids=sids,
            mask=mask,
        )

    def load_previous_events(self, columns, dates, sids, mask):
        if not columns:
            return {}

        return self._load_events(
            name_map=self.previous_value_columns,
            indexer=self.previous_event_indexer(dates, sids),
            columns=columns,
            dates=dates,
            sids=sids,
            mask=mask,
        )

    def _load_events(self, name_map, indexer, columns, dates, sids, mask):
        def to_frame(array):
            return pd.DataFrame(array, index=dates, columns=sids)

        out = {}
        for c in columns:
            raw = self.events[name_map[c]][indexer]
            # indexer will be -1 for locations where we don't have a known
            # value.
            raw[indexer < 0] = c.missing_value

            # Delegate the actual array formatting logic to a DataFrameLoader.
            loader = DataFrameLoader(c, to_frame(raw), adjustments=None)
            out[c] = loader.load_adjusted_array([c], dates, sids, mask)[c]
        return out

    def load_adjusted_array(self, columns, dates, sids, mask):
        n, p = self.split_next_and_previous_event_columns(columns)
        next_col_results = self.load_next_events(n, dates, sids, mask)
        prev_col_results = self.load_previous_events(p, dates, sids, mask)

        return merge(next_col_results, prev_col_results)
