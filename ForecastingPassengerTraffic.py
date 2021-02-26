import pandas as pd
import sys
import numpy as np
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.arima_model import ARMA

n = int(input())
data = []
for i in range(n):
    data.append(int(input().strip().split()[1]))

model = AR(data)
modelfitted = model.fit()
yhat  = modelfitted.predict()[:12]

f = []
for i in yhat:
    f.append(str(int(i)))

result = "\n".join(f)

print(result)