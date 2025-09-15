# Fit-the-ARMA-model-for-any-data-set

### DEVELOPED BY: MOENISH BAALAN G
### REGISTER NO: 212223220057

# AIM:
To implement ARMA model in python.

# ALGORITHM:
1. Import necessary libraries.

2. Set up matplotlib settings for figure size.

3. Define an ARMA(1,1) process with coefficients ar1 and ma1, and generate a sample of 1000 data points using the ArmaProcess class. Plot the generated time series and set the title and x- axis limits.

4. Display the autocorrelation and partial autocorrelation plots for the ARMA(1,1) process using plot_acf and plot_pacf.

5. Define an ARMA(2,2) process with coefficients ar2 and ma2, and generate a sample of 10000 data points using the ArmaProcess class. Plot the generated time series and set the title and x- axis limits.

6. Display the autocorrelation and partial autocorrelation plots for the ARMA(2,2) process using plot_acf and plot_pacf.

# PROGRAM:

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv("results.csv")
data['date'] = pd.to_datetime(data['date'])

yearly_scores = data.groupby(data['date'].dt.year)['home_score'].mean().reset_index()
yearly_scores.rename(columns={'date': 'year', 'home_score': 'avg_home_score'}, inplace=True)

X = yearly_scores['avg_home_score'].dropna().values
N = 1000

plt.figure(figsize=(12, 6))
plt.plot(yearly_scores['year'], X, marker='o')
plt.title('Yearly Average Home Scores')
plt.xlabel("Year")
plt.ylabel("Avg Home Score")
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plot_acf(X, lags=len(X)//4, ax=plt.gca())
plt.title('Original Data ACF')

plt.subplot(2, 1, 2)
plot_pacf(X, lags=len(X)//4, ax=plt.gca())
plt.title('Original Data PACF')
plt.tight_layout()
plt.show()

arma11_model = ARIMA(X, order=(1, 0, 1)).fit()
phi1_arma11 = arma11_model.arparams[0]
theta1_arma11 = arma11_model.maparams[0]

ar1 = np.array([1, -phi1_arma11])
ma1 = np.array([1, theta1_arma11])
ARMA_1 = ArmaProcess(ar1, ma1).generate_sample(nsample=N)

plt.figure(figsize=(12, 6))
plt.plot(ARMA_1)
plt.title('Simulated ARMA(1,1) Process')
plt.xlim([0, 500])
plt.show()

plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plot_acf(ARMA_1, lags=40, ax=plt.gca())
plt.title("ACF of Simulated ARMA(1,1)")

plt.subplot(2, 1, 2)
plot_pacf(ARMA_1, lags=40, ax=plt.gca())
plt.title("PACF of Simulated ARMA(1,1)")
plt.tight_layout()
plt.show()

arma22_model = ARIMA(X, order=(2, 0, 2)).fit()
phi1_arma22, phi2_arma22 = arma22_model.arparams
theta1_arma22, theta2_arma22 = arma22_model.maparams

ar2 = np.array([1, -phi1_arma22, -phi2_arma22])
ma2 = np.array([1, theta1_arma22, theta2_arma22])
ARMA_2 = ArmaProcess(ar2, ma2).generate_sample(nsample=N*10)

plt.figure(figsize=(12, 6))
plt.plot(ARMA_2)
plt.title('Simulated ARMA(2,2) Process')
plt.xlim([0, 500])
plt.show()

plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plot_acf(ARMA_2, lags=40, ax=plt.gca())
plt.title("ACF of Simulated ARMA(2,2)")

plt.subplot(2, 1, 2)
plot_pacf(ARMA_2, lags=40, ax=plt.gca())
plt.title("PACF of Simulated ARMA(2,2)")
plt.tight_layout()
plt.show()

```
# OUTPUT:

## Original data:

<img width="1284" height="675" alt="image" src="https://github.com/user-attachments/assets/ad5c6523-12fd-4827-bb66-5ea0fc9350d2" />

## Autocorrelation:

<img width="1385" height="356" alt="image" src="https://github.com/user-attachments/assets/bfc2ca7a-d53a-48ba-b96e-2945c096f1a6" />

## Partial Autocorrelation:

<img width="1379" height="339" alt="image" src="https://github.com/user-attachments/assets/80aa8da8-45e4-4c4d-994e-7c5b28f1d277" />

## SIMULATED ARMA(1,1) PROCESS:

<img width="1375" height="344" alt="image" src="https://github.com/user-attachments/assets/7098acb3-50f9-4fcd-a7de-4791c4231d68" />

## Autocorrelation:

<img width="1368" height="331" alt="image" src="https://github.com/user-attachments/assets/27228145-c7f3-4031-a801-4e9f17f7e257" />


## Partial Autocorrelation:

<img width="1244" height="658" alt="image" src="https://github.com/user-attachments/assets/98519f03-a5d9-4cd0-b043-c86e522bb6c6" />


## SIMULATED ARMA(2,2) PROCESS:

<img width="1283" height="661" alt="image" src="https://github.com/user-attachments/assets/548d826d-2a2c-4305-84c1-b6d3a8cc6def" />

## Autocorrelation:

<img width="1411" height="342" alt="image" src="https://github.com/user-attachments/assets/8c0dbc01-7cfd-4637-ad8f-b56f0423940e" />

## Partial Autocorrelation:

<img width="1373" height="335" alt="image" src="https://github.com/user-attachments/assets/57930a9b-008b-4ed8-995d-c456febeab33" />

RESULT:
Thus, a python program is created to fir ARMA Model successfully.
