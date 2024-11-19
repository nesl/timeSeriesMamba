import torch
import torch.nn as nn
from statsmodels.tsa.arima.model import ARIMA
import numpy as np

class Model(nn.Module):
    """
    ARIMA-based model for long-term forecasting in PyTorch.
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.p = configs.arima_p
        self.d = configs.arima_d
        self.q = configs.arima_q
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.channels = configs.enc_in  # Number of input features

    def _fit_arima(self, series):
        """
        Fit ARIMA model to a given time series.
        :param series: Time series data (numpy array)
        :return: ARIMA model instance
        """
        model = ARIMA(series, order=(self.p, self.d, self.q))
        fitted_model = model.fit()
        return fitted_model

    def forecast(self, x):
        """
        Fits ARIMA models for each channel and predicts the next steps.
        :param x: Input tensor of shape (batch_size, seq_len, channels)
        :return: Predicted tensor of shape (batch_size, pred_len, channels)
        """
        x_np = x.detach().cpu().numpy()  # Convert to numpy for ARIMA
        batch_size, seq_len, channels = x_np.shape
        preds = np.zeros((batch_size, self.pred_len, channels))

        for b in range(batch_size):
            for ch in range(channels):
                series = x_np[b, :, ch]
                model = self._fit_arima(series)
                preds[b, :, ch] = model.forecast(steps=self.pred_len)

        return torch.tensor(preds, dtype=x.dtype, device=x.device)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        """
        Forward method for long-term forecasting.
        :param x_enc: Input tensor of shape (batch_size, seq_len, channels)
        :return: Forecasted tensor of shape (batch_size, pred_len, channels)
        """
        return self.forecast(x_enc)
