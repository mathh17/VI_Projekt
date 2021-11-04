"""
In this script the algorithm Adaboost will be implemented. 
TBC

"""
#%%
#Load all the packages
import sklearn
import os
import requests
import numpy as np
import pandas as pd
import seaborn as sns
import Holidays_calc as hc
from datetime import datetime, timedelta
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn import ensemble
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor

#%%
# read the files from the datafolder containing data fra DK2
# changing the path to the datafolder
path = r'C:\Users\MTG\OneDrive - Energinet.dk\Skrivebord\VI_projekt\Scripts\data\stations_data_dk2'
os.chdir(path)

temp_conc_data = pd.DataFrame(columns=['time'])
radi_conc_data = pd.DataFrame(columns=['time'])

# goes through all the files one by one adding them all together to create a Dataframe with one column for each station
for file in os.listdir():
    df = pd.read_pickle(file)
    file_name = os.path.basename(file)
    if 'temp_mean_past1h' in df.columns:
        temp_conc_data = pd.merge(temp_conc_data,df[['time','temp_mean_past1h']],on='time',how='outer', suffixes=(['old','_{}'.format(file_name)]))
    if 'radia_glob_past1h' in df.columns:
        radi_conc_data = pd.merge(radi_conc_data,df[['time','radia_glob_past1h']],on='time',how='outer', suffixes=(['old','_{}'.format(file_name)]))

#%%
# takes all the columns and calculates the mean for each row. which gives us a mean value for all stations at the given time.
temp_conc_data['mean'] = temp_conc_data.iloc[:,1:12].sum(axis=1) / 11
radi_conc_data['mean'] = radi_conc_data.iloc[:,1:7].sum(axis=1) / 6
dk2_mean = pd.DataFrame()
dk2_mean['time'] = temp_conc_data['time']
dk2_mean['temp_mean_past1h'] = temp_conc_data['mean']
dk2_mean['radia_glob_past1h'] = radi_conc_data['mean']
dk2_mean.head()

#%%
# Read Enernginet Pickle Data
# Change back 
old_path = r'C:\Users\MTG\OneDrive - Energinet.dk\Skrivebord\VI_projekt\Scripts'
os.chdir(old_path)
df_DK1_2010_2015 = pd.read_pickle("data/dk1_data_2010_2015.pkl")
df_DK2_2010_2015 = pd.read_pickle("data/dk2_data_2010_2015.pkl")
df_DK1_2015_2020 = pd.read_pickle("data/dk1_data_2015_2020.pkl")
df_DK2_2015_2020 = pd.read_pickle("data/dk2_data_2015_2020.pkl")
df_DK1 = pd.concat([df_DK1_2010_2015,df_DK1_2015_2020], ignore_index=True)
df_DK2 = pd.concat([df_DK2_2010_2015,df_DK2_2015_2020], ignore_index=True)

#%%
#Merge data into one DF, on the hour of observations
dk2_mean['time'] = pd.to_datetime(dk2_mean['time'],format='%Y-%m-%dT%H:%M:%S', utc=True)
df_DK2['HourUTC'] = pd.to_datetime(df_DK2['HourUTC'],format='%Y-%m-%dT%H:%M:%S', utc=True)
df_DK2 = df_DK2.rename(columns={'HourUTC':'time'})
conc_data = pd.merge(dk2_mean, df_DK2, on='time', how='outer')
conc_data.dropna(inplace=True)

#%%
holidays = []
for i in range(len(conc_data)):
    is_holiday = hc.get_date_type(conc_data.loc[i,'time'])
    holidays.append(is_holiday)
conc_data['is_holiday'] = holidays

#%%
#Dividing train test and validation.
values = conc_data['Con']
pred_data = conc_data[['time','temp_mean_past1h','radia_glob_past1h','is_holiday']]
pred_data['time'] = pred_data['time'].dt.hour
#%%
#Dividing dataset into test, train and validation
X_train, X_test, y_train, y_test = train_test_split(pred_data, values, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
print(X_train.shape, X_test.shape, X_val.shape, y_train.shape, y_test.shape, y_val.shape)
#%%
# Declaring hyperparameters, 
# initializing the model, 
# fitting it to the data,
# predicting on the validation data.
loss = ['squared_error']#,'absolute_error','huber']
learning_rate = [0.1]#,0.1,0.2,1]
n_estimators = [500]#,50,100,200]
criterion = ['friedman_mse']#,'mse']

results = []
# The for loops goes through the different hyperparameters
for losses in loss:
    for lrs in learning_rate:
        for estimators in n_estimators:
            for crits in criterion:
                gbt = ensemble.GradientBoostingRegressor(loss=losses,
                                                        learning_rate=lrs,
                                                        n_estimators=estimators,
                                                        criterion=crits)
                gbt.fit(X_train,y_train)
                y_val_hat = gbt.predict(X_val)
                acc = gbt.score(X_val,y_val)
# the results from the different hyperparameters are then stored so we can get the best one
                results.append([acc,losses,lrs,estimators,crits])
results_df = pd.DataFrame(results)
results_df.columns=['accuracy',"loss functions","learning_rate","n_estimators","criterions"]
#prints the results from best to worst in regards to accuracy, listting the hyperparameters for the result
results_df = results_df.sort_values('accuracy', ascending=False)
results_df
# %%
#Calculates the MSE if you just use the data from 24hours before at prediction.
naive_y_val = np.roll(y_val.to_numpy(),24)
mse_naive_y_val = np.sum((y_val-naive_y_val)**2)/len(y_val)
mse_naive_y_val
#%%
#Calculates the MSE for the predicted values from the validation set.
np.sum((y_val-y_val_hat)**2)/len(y_val)

#%%

y_test_hat = gbt.predict(X_test)
test_acc = gbt.score(X_test,y_test)
# %%
#pair plot of DMI data and the consmption. 
pair_data = X_val
pair_data['con'] = y_val
pairplot = sns.pairplot(pair_data)
plt.show()
# %%
# Plots the deviance in the prediction and on the training data. To visualize how the model learns and behaves.
best_est = results_df.iloc[0,3]
test_score = np.zeros((best_est,), dtype=np.float64)
for i, y_pred in enumerate(gbt.staged_predict(X_test)):
    test_score[i] = gbt.loss_(y_test, y_pred)

fig = plt.figure(figsize=(6, 6))
plt.subplot(1, 1, 1)
plt.title('Deviance')
plt.plot(np.arange(best_est) + 1, gbt.train_score_, 'b-',
         label='Training Set Deviance')
plt.plot(np.arange(best_est) + 1, test_score, 'r-',
         label='Test Set Deviance')
plt.legend(loc='upper right')
plt.xlabel('Boosting Iterations')
plt.ylabel('Deviance')
fig.tight_layout()
plt.show()
# %%
# Plots the predicted values with the exact values to compare how the model predicts.
y_plot = pd.DataFrame()
y_plot['exact_values'] = y_val
y_plot['predicted_values'] = y_val_hat
y_plot = y_plot.sort_index()
fig = plt.figure(figsize=(6, 6))
plt.subplot(1, 1, 1)
plt.title('Predicts vs Exact values')
plt.plot(range(y_plot.index), y_plot['exact_values'], 'r-',
         label='Exact values')
plt.plot(range(y_plot.index), y_plot['predicted_values'], 'b-',
         label='Precited Values')
plt.legend(loc='upper right')
#plt.xlabel('Boosting Iterations')
plt.ylabel('Consumption')
fig.tight_layout()
plt.show()
# %%
