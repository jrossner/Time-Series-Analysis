#for data format
import numpy as np
#for fitting arima model
from statsmodels.tsa.arima.model import ARIMA
#for checking acf and pacf plots
from statsmodels.tsa.graphics.tsaplots import plot_pacf, plot_acf
#for plotting (visual analysis)
import matplotlib.pyplot as plt

data = np.array('''set time series data, only 1 dimension''')

#check out raw data
f = plt.figure()
f.set_width(15)
f.set_height(3)
plt.plot(data)
plt.show()

#set train and test set
lotal_obs = len(data)
trn_prop = 0.9 #percent of data set to use for training
trn_length = round(trn_prop*total_obs, 0)
train_set = data[0:trn_length]
test_set = data[trn_length:]

#check acf plot for possible q values
plot_acf(train_set); #use semicolon to prevent duplicate plots

#check pacf plot for possible p values
plot_pacf(train_set); #use semicolon to prevent duplicate plots

#to compare AICs of multiple models:
models = []
def fit_arima(train,p,q):
  '''
  This function fits an ARIMA(p,0,q) model, finds the AIC, and 
  appends a dictionary of the p,q, and AIC value for the model
  to the models list for later comparison
  '''
  
  model = ARIMA(train, order = (p,0,q))
  model_fit = model.fit()
  aic = model_fit.aic
  models.append({'p': p, 'q': q, 'AIC': aic})
  #ending function with print of summary to use for further parameter analysis
  print(model_fit.summary())
  

#once best AIC has been found
best_model = ARIMA(train_set, order = (best_p, 0, best_q))
best_fit = best_model.fit()

#check out model
print(f"The MSE of this model is: {best_fit.mse}")
plt.plot(best_fit.resid)
plt.show()

#check fitted values against real values from training set
fit_vals = best_fit.fittedvalues
f = plt.figure()
f.set_figwidth(15)
f.set_figheight(3)
plt.plot(train_set, color = "blue")
plt.plot(fit_vals, color = "orange", linestyle = "dotted")
plt.show()

#Make forecast and check against test_set
predicted_vals = best_fit.forecast(len(test_set))
f = plt.figure()
f.set_figwidth(15)
f.set_figheight(3)
plt.plot(test_set, color = "blue")
plt.plot(predicted_vals, color = "orange", linestyle = "dotted")
plt.show()