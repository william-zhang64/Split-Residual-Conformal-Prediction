# Split-Residual-Conformal-Prediction
Implementation details for the paper: Decomposition-Based Modular Conformal Prediction for Two-Stage Modeling

We provide 4 notebooks to reproduce the experiments. 

Non-adaptive method experiments are in Non Adaptive.ipynb

Then in Synthetic Adaptive.ipynb, we have our adaptive results on synthetic data with asymmetric shifts,
in Two Stage Manheim.ipynb, we have the adaptive results on the automobile dataset, and in Two Stage Stocks.ipynb, we have adaptive results on stocks data.

We include the Indicators folder which provides auxiliary data for the automobile dataset, and DtACI-main (directly from Gibbs & Candès 2024)
to run that baseline. We also provide the stocks data we used, in stocks.zip.

The adaptive_conformal.py file contains the implementation of our adaptive method as well as the adaptive baselines.
two_stage.py and utils.py hold the implementation for a twostage time-series forecasting model.
