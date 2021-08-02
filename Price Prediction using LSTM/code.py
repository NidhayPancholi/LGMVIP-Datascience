# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error,accuracy_score,confusion_matrix,classification_report

sbi=pd.read_csv("/kaggle/input/stock-bse/Sbi.csv",usecols=[0,1,2,3,4,12,11,6])
chalet=pd.read_csv("/kaggle/input/stock-bse/Chalet hotels.csv",usecols=[0,1,2,3,4,12,11,6])
chalet.head()

def make_pattern(df,n,days):
    X=[]
    X_volume=[]
    y=[]
    volume=df.iloc[:]['No.of Shares']
    volume=volume[::-1]
    prices=df.iloc[:]['Close Price']
    prices=prices[::-1]
    for x in range(len(prices)):
        try:
            X.append(prices[x:x+n])
            X_volume.append(volume[x:x+n])
        except KeyError:
            continue
        try:
            y.append(prices[len(prices)-x-n-days])
        except KeyError:
            X.pop()
            X_volume.pop()
            continue
    X=np.array(X)
    X_volume=np.array(X_volume)
    y=np.array(y)
    X_train=np.concatenate((X,X_volume),1)
    return X_train,y
  
def train_model(train,labels):
    forest=RandomForestRegressor(max_depth=6,n_estimators=500)
    forest.fit(train,labels)
    return forest
def buy_signal(x,y,day_data):
    if x[day_data-1]<y:
        return 1
    return 0
def plot_price(pred,true,day_data,valid=False):
    true_x=[x for x in range(len(true))]
    if not valid:
        pred_x=[x+day_data for x in range(len(pred))]
    else :
        pred_x=[x for x in range(len(pred))]
    fig=plt.figure(figsize=(20,8))
    sns.lineplot(x=true_x,y=true,label='Closing')
    sns.lineplot(x=pred_x,y=pred,label='Predicted')
def train_test(X,y,percent):
    test_size=int(np.ceil(X.shape[0]*(1-percent)))
    return X[:test_size],X[test_size:],y[:test_size],y[test_size:]

def create_classifiying_labels(X,y,day_data):
    buys=[]
    for x in range(len(X)):
        buys.append(buy_signal(X[x],y[x],day_data))
    return buys
  
  
  def true_false_buy(future_price,whole_timeseries,false_buy,true_buy,days_in_future):
    fig=plt.figure(figsize=(15,6))
    sns.lineplot(x=[x for x in range(len(future_price))],y=whole_timeseries[len(whole_timeseries)-len(future_price):],label='Price')
    sns.scatterplot(x=false_buy.index-days_in_future,y=false_buy['Today price'],label='False Buy',color='red')
    sns.scatterplot(x=true_buy.index-days_in_future,y=true_buy['Today price'],label='True Buy',color='green')

def buy_sell_prediction(future_price,whole_timeseries,prediction_df,days_in_future):
    fig=plt.figure(figsize=(15,6))
    sns.lineplot(x=[x for x in range(len(future_price))],y=whole_timeseries[len(whole_timeseries)-len(future_price):],label='Price')
    sns.scatterplot(x=prediction_df.index-days_in_future,y=prediction_df['Today price'],hue=prediction_df['buy_pred'])
    
def calculate_gain(predicted_df):
    gain=0
    loss=0
    buys=predicted_df[predicted_df['buy_pred']==1]
    for x in range(buys.shape[0]):
        if buys.iloc[x][1]-buys.iloc[x][0]>=0:
            gain+= buys.iloc[x][1]-buys.iloc[x][0]
        else :
            loss += buys.iloc[x][0]-buys.iloc[x][1]
    return gain,loss,buys.shape[0]


def potential_gain(predicted_df):
    gain=0
    buys=predicted_df[predicted_df['Buy_true']==1]
    for x in range(buys.shape[0]):
        gain+= buys.iloc[x][1]-buys.iloc[x][0]
    return gain,buys.shape[0]
  
day_data=10
days_in_future=10
X,y=make_pattern(sbi,day_data,days_in_future)
X_train,X_valid,y_train,y_valid=train_test(X,y,0.25)
forest=train_model(X_train,y_train)

train_preds=forest.predict(X_train)
preds=forest.predict(X_valid)

plot_price(train_preds,y_train,day_data,True)

plot_price(preds,y_valid,day_data,True)

temp=np.array(X_valid)
today_price=temp[:,day_data-1]
d=pd.DataFrame({'Today price':today_price,'Future price':y_valid,'Predicted price':preds})

d['buy_pred']=(d['Predicted price']-d['Today price']>0).astype(int)
d['Buy_true']=(d['Future price']-d['Today price']>0).astype(int)

calculate_gain(d)
potential_gain(d)
