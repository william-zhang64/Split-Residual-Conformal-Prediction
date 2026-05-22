from utils import *
import numpy as np
import pandas as pd
import statsmodels.tsa.api as sm
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.kernel_ridge import KernelRidge
from statsmodels.tsa.tsatools import lagmat
from statsmodels.tsa.deterministic import DeterministicProcess
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

class IndicatorARIMA:
    def __init__(self, indicators, order, trend, test_date, data, transform=None, horizon=6):
        """
        Parameters:
        indicators (str): Column name of the indicator to model.
        order (tuple): ARIMA order (p, d, q).
        trend (str): Trend parameter for ARIMA.
        test_date (str or pd.Timestamp): The test (forecasting) start date.
        data (tuple): Tuple containing (indicator_df, manheim_df).
        transform (str, optional): Type of transformation ('differencing', 'log-differencing'). Default is None.
        horizon (int, optional): Number of months to forecast. Default is 6.
        """
        transforms = [None, 'differencing', 'log-differencing']
        assert transform in transforms, f'transform must be one of {transforms}'
        assert horizon >= 1

        self.indicator_df, self.manheim_df = data
        X_raw = self.indicator_df[[indicators]].dropna()
        self.test_date_val = X_raw.loc[X_raw.index == pd.to_datetime(test_date)]
        test_series = X_raw.loc[X_raw.index >= pd.to_datetime(test_date) - pd.DateOffset(months=horizon)]
        self.test_date_series = test_series.loc[test_series.index < pd.to_datetime(test_date)]
        
        X_raw = X_raw.loc[X_raw.index <= pd.to_datetime(test_date) - pd.DateOffset(months=horizon)]
        X_raw.index.freq = 'MS'
        self.X_raw = pd.DataFrame(X_raw)
        self.order = order
        self.trend = trend
        self.test_date = test_date
        self.transform = transform
        self.X_transformed = self.X_raw
        self.X_init = self.X_raw.loc[pd.to_datetime(test_date) - pd.DateOffset(months=horizon)]
        self.transformed = False
        self.fitted = False

    def apply_transform(self):
        if self.transform == 'differencing':
            self.X_transformed = self.X_raw.diff(1)
        elif self.transform == 'log-differencing':
            self.X_transformed = np.log(self.X_raw + 1e-6).diff(1)
        self.transformed = True

    def fit(self):
        if self.transformed == False:
            self.apply_transform()
        self.model = ARIMA(self.X_transformed.dropna(), order=self.order, trend=self.trend, freq='MS')
        self.results = self.model.fit()
        self.fitted = True

    def forecast_series(self, horizon):
        if self.fitted == False:
            self.fit()
        X_ = pd.DataFrame(self.results.forecast(horizon))
        X_.columns = [self.X_transformed.columns[0]]
        X_.index = pd.date_range(pd.to_datetime(self.test_date) - pd.DateOffset(months=horizon), pd.to_datetime(self.test_date), inclusive='right', freq='MS')
        if self.transform == 'differencing':
            X_ = self.X_init + X_.cumsum()
        elif self.transform == 'log-differencing':
            X_ = self.X_init * np.exp(X_.cumsum())
        return X_

    def get_forecast(self, horizon):
        X_ = self.forecast_series(horizon)
        X_ = X_.iloc[-1:]
        return X_

    def get_perfect_forecast(self):
        return self.test_date_val
    
    def perfect_forecast_series(self):
        return self.test_date_series

class IndicatorVAR:
    def __init__(self, indicators, lags, trend, test_date, data, output_indicators=None, transform=None, horizon=6):
        """
        Parameters:
        indicators (list): Column names of the indicators to include as inputs.
        lags (int): Number of time-lagged indicator values to consider.
        trend (str): Trend parameter for ARIMA.
        test_date (str or pd.Timestamp): The test (forecasting) start date.
        data (tuple): Tuple containing (indicator_df, manheim_df).
        output_indicators (list): Column names of the indicators to output during forecasting. Default is None.
        transform (str, optional): Type of transformation ('differencing', 'log-differencing'). Default is None.
        horizon (int, optional): Number of months to forecast. Default is 6.
        """
        self.indicator_df, self.manheim_df = data
        X_raw = self.indicator_df[indicators].dropna()
        self.test_date_val = X_raw.loc[X_raw.index == pd.to_datetime(test_date)]
        test_series = X_raw.loc[X_raw.index >= pd.to_datetime(test_date) - pd.DateOffset(months=horizon)]
        self.test_date_series = test_series.loc[test_series.index < pd.to_datetime(test_date)]
        X_raw = X_raw.loc[X_raw.index <= pd.to_datetime(test_date) - pd.DateOffset(months=horizon)]
        X_raw.index.freq = 'MS'
        self.X_raw = pd.DataFrame(X_raw)
        self.lags = lags
        self.trend = trend
        self.transform = transform
        self.X_transformed = self.X_raw
        self.X_init = self.X_raw.loc[pd.to_datetime(test_date) - pd.DateOffset(months=horizon)]
        self.transformed = False
        self.fitted = False
        self.test_date = test_date

        if output_indicators == None:
            self.output_indicators = indicators
        else:
            self.output_indicators = output_indicators
            assert set(self.output_indicators).issubset(set(indicators))

    def apply_transform(self):
        if self.transform == 'differencing':
            self.X_transformed = self.X_raw.diff(1)
        elif self.transform == 'log-differencing':
            self.X_transformed = np.log(self.X_raw).diff(1)
        self.transformed = True

    def fit(self):
        if self.transformed == False:
            self.apply_transform()
        self.model = VAR(endog=self.X_transformed.dropna(), freq='MS')
        self.results = self.model.fit(maxlags=self.lags, trend=self.trend)
        self.fitted = True

    def forecast_series(self, horizon):
        if self.fitted == False:
            self.fit()
        X_ = pd.DataFrame(self.results.forecast(y=self.X_transformed.values[-self.lags:], steps=horizon))
        X_.columns = self.X_transformed.columns
        X_.index = pd.date_range(pd.to_datetime(self.test_date) - pd.DateOffset(months=horizon), pd.to_datetime(self.test_date), inclusive='right', freq='MS')
        if self.transform == 'differencing':
            X_ = self.X_init + X_.cumsum()
        elif self.transform == 'log-differencing':
            X_ = self.X_init * np.exp(X_.cumsum())
        return X_[self.output_indicators]

    def get_forecast(self, horizon):
        X_ = self.forecast_series(horizon)
        X_ = X_.iloc[-1:]
        return X_
    
    def get_perfect_forecast(self):
        return self.test_date_val
    def perfect_forecast_series(self):
        return self.test_date_series

class CombinedUpstream:
    def __init__(self, models, test_date, data):
        self.models = models
        self.test_date = test_date
        self.data = data

    def forecast_series(self, horizon):
        preds = []
        for model in self.models:
            model_class = model['model_class']
            model_kwargs = model['kwargs']
            model = model_class(indicators=model['indicators'], test_date=self.test_date, data=self.data, **model_kwargs)
            preds.append(model.forecast_series(horizon))
        preds = pd.DataFrame(pd.concat(preds, axis=1))
        return preds

    def get_forecast(self, horizon):
        preds = []
        for model in self.models:
            model_class = model['model_class']
            model_kwargs = model['kwargs']
            model = model_class(indicators=model['indicators'], test_date=self.test_date, data=self.data, **model_kwargs)
            preds.append(model.get_forecast(horizon))
        preds = pd.DataFrame(pd.concat(preds, axis=1))
        return preds
    def get_perfect_forecast(self):
        preds = []
        for model in self.models:
            model_class = model['model_class']
            model_kwargs = model['kwargs']
            model = model_class(indicators=model['indicators'], test_date=self.test_date, data=self.data,  **model_kwargs)
            preds.append(model.get_perfect_forecast())
        preds = pd.DataFrame(pd.concat(preds, axis=1))
        return preds
    
    def perfect_forecast_series(self):
        preds = []
        for model in self.models:
            model_class = model['model_class']
            model_kwargs = model['kwargs']
            model = model_class(indicators=model['indicators'], test_date=self.test_date, data=self.data, **model_kwargs)
            preds.append(model.perfect_forecast_series())
        preds = pd.DataFrame(pd.concat(preds, axis=1))
        return preds

class TwoStageModel:
    def __init__(self, indicators, upstream, downstream, test_date, data, downstream_kwargs={}):
        self.indicators = indicators
        self.upstream = upstream
        self.downstream = downstream
        self.test_date = test_date
        self.downstream_kwargs = downstream_kwargs
        self.indicator_df, self.manheim_df = data

    def forecast(self, horizon):
        X_downstream = self.indicator_df[self.indicators].loc[self.manheim_df.index]
        y_downstream = self.manheim_df['manheim']
        X_downstream = X_downstream.loc[X_downstream.index <= pd.to_datetime(self.test_date) - pd.DateOffset(months=horizon)]
        y_downstream = y_downstream.loc[y_downstream.index <= pd.to_datetime(self.test_date) - pd.DateOffset(months=horizon)]
        if self.downstream == 'OLS':
            upstream_preds = self.upstream.get_forecast(horizon)
            downstream = LinearRegression().fit(X_downstream, y_downstream, sample_weight=exponential_decay_weights(len(y_downstream)))
            pred = downstream.predict(upstream_preds)
            return pred
        elif self.downstream == 'ARIMA':
            upstream_preds = self.upstream.forecast_series(horizon)
            model = ARIMA(endog=y_downstream, exog=X_downstream, **self.downstream_kwargs)
            downstream = model.fit()
            return downstream.forecast(steps=horizon, exog=upstream_preds.iloc[-horizon:]).iloc[-1]
    def perfect_upstream_forecast(self, horizon):
        X_downstream = self.indicator_df[self.indicators].loc[self.manheim_df.index]
        y_downstream = self.manheim_df['manheim']
        X_downstream = X_downstream.loc[X_downstream.index <= pd.to_datetime(self.test_date) - pd.DateOffset(months=horizon)]
        y_downstream = y_downstream.loc[y_downstream.index <= pd.to_datetime(self.test_date) - pd.DateOffset(months=horizon)]
        upstream_preds = self.upstream.get_perfect_forecast()
        upstream_series = self.upstream.perfect_forecast_series()
        if self.downstream == 'OLS':
            downstream = LinearRegression().fit(X_downstream, y_downstream, sample_weight=exponential_decay_weights(len(y_downstream)))
            pred = downstream.predict(upstream_preds)
            return pred
        elif self.downstream == 'ARIMA':
            model = ARIMA(endog=y_downstream, exog=X_downstream, **self.downstream_kwargs)
            downstream = model.fit()
            return downstream.forecast(steps=horizon, exog=upstream_series).iloc[-1]
        
