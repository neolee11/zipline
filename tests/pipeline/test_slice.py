"""
Tests for slicing pipeline terms.
"""
from numpy import arange
from pandas import (
    DataFrame,
    date_range,
    Int64Index,
    Timestamp,
)

from zipline.errors import UnsupportedPipelineColumn
from zipline.pipeline import CustomFactor, Pipeline
from zipline.pipeline.data import USEquityPricing
from zipline.pipeline.engine import SimplePipelineEngine
from zipline.pipeline.factors import Returns
from zipline.pipeline.loaders.frame import DataFrameLoader
from zipline.testing import check_arrays
from zipline.testing.fixtures import WithTradingEnvironment, ZiplineTestCase

from .test_engine import (
    AssetID,
    AssetIDPlusDay,
    OpenPrice,
)


class SliceTestCase(WithTradingEnvironment, ZiplineTestCase):
    sids = ASSET_FINDER_EQUITY_SIDS = Int64Index([1, 2, 3])
    START_DATE = Timestamp('2015-01-31', tz='UTC')
    END_DATE = Timestamp('2015-03-01', tz='UTC')

    @classmethod
    def init_class_fixtures(cls):
        super(SliceTestCase, cls).init_class_fixtures()

        day = cls.env.trading_day
        sids = cls.sids

        cls.dates = dates = date_range(
            '2015-02-01', '2015-02-28', freq=day, tz='UTC',
        )

        cls.raw_data = DataFrame(
            data=arange(len(dates) * len(sids), dtype=float).reshape(
                len(dates), len(sids),
            ),
            index=dates,
            columns=cls.asset_finder.retrieve_all(sids),
        )

        close_loader = DataFrameLoader(USEquityPricing.close, cls.raw_data)

        cls.engine = SimplePipelineEngine(
            {USEquityPricing.close: close_loader}.__getitem__,
            cls.dates,
            cls.asset_finder,
        )

    def test_slice(self):
        my_asset_column = 0
        start_date_index = 5
        end_date_index = 9

        assets = self.asset_finder.retrieve_all(self.sids)
        my_asset = assets[my_asset_column]
        my_asset_only = (AssetID().eq(my_asset_column + 1))

        returns = Returns(window_length=2)
        returns_slice = returns[my_asset]

        class UsesSlicedInput(CustomFactor):
            window_length = 3
            inputs = [returns, returns_slice]

            def compute(self, today, assets, out, returns, returns_slice):
                # Make sure that our slice is the correct shape (i.e. has only
                # one column) and that it has the same values as the original
                # returns factor from which it is derived.
                assert returns_slice.shape == (self.window_length, 1)
                check_arrays(returns_slice[:, 0], returns[:, my_asset_column])

        columns = {
            'uses_sliced_input': UsesSlicedInput(),
            'uses_sliced_input_masked': UsesSlicedInput(mask=my_asset_only),
        }
        pipe = Pipeline(columns=columns)

        # Assertions about the expected slice data are made in the `compute`
        # function of our custom factor above.
        self.engine.run_pipeline(
            pipe, self.dates[start_date_index], self.dates[end_date_index],
        )

        # Test that slices cannot be added as a pipeline column.
        with self.assertRaises(UnsupportedPipelineColumn):
            pipe = Pipeline(columns={'open_slice': OpenPrice()[my_asset]})

        pipe = Pipeline(columns={})
        with self.assertRaises(UnsupportedPipelineColumn):
            pipe.add(OpenPrice()[my_asset], 'open_slice')

    def test_single_column_output(self):
        start_date_index = 5
        end_date_index = 9
        alternating_mask = (AssetIDPlusDay() % 2).eq(0)

        class SingleColumnOutput(CustomFactor):
            window_length = 3
            inputs = [USEquityPricing.close]
            ndim = 1

            def compute(self, today, assets, out, close):
                # Because we specified ndim as 1, `out` should be a singleton
                # array but `close` should be a regular sized input.
                assert out.shape == (1,)
                assert close.shape == (3, 1)
                out[:] = close.mean()

        class Demean(CustomFactor):
            window_length = 1
            inputs = [USEquityPricing.close, SingleColumnOutput()]

            def compute(self, today, assets, out, close, avg_close):
                # Make sure that `avg_close` has the correct shape. That is, it
                # should always have one column regardless of any mask passed
                # to `Demean`.
                assert avg_close.shape == (1, 1)

        columns = {
            'demean': Demean(),
            'masked_demean': Demean(mask=alternating_mask),
        }
        pipe = Pipeline(columns=columns)

        # Assertions about the expected slice data are made in the `compute`
        # function of our custom factors above.
        self.engine.run_pipeline(
            pipe, self.dates[start_date_index], self.dates[end_date_index],
        )
