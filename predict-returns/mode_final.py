import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import datetime

from sklearn.cross_validation import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import xgboost as xgb
from sklearn.neighbors import KNeighborsClassifier

train = pd.read_csv("train.csv")
test=pd.read_csv("test.csv")
train["return"].iloc[1665]=train["return"].mean()
# train["return"].iloc[7096]=train["return"].mean()
# train["return"].iloc[1731]=train["return"].mean()
# train["return"].iloc[1795]=train["return"].mean()
y_train=train["return"]
train=train.drop(["return"],axis=1)
combine=train.append(test,ignore_index=True)

start_creation=[]
creation_sell=[]
start_sell=[]

for i in range(len(combine)):

	temp_start1=combine["start_date"].iloc[i]
	temp_start1=str(temp_start1)
	temp_start2=datetime.datetime(int(temp_start1[:4]),int(temp_start1[4:6]),int(temp_start1[6:]))

	temp_creation1=combine["creation_date"].iloc[i]
	temp_creation1=str(temp_creation1)
	temp_creation2=datetime.datetime(int(temp_creation1[:4]),int(temp_creation1[4:6]),int(temp_creation1[6:]))

	temp_end1=combine["sell_date"].iloc[i]
	temp_end1=str(temp_end1)
	temp_end2=datetime.datetime(int(temp_end1[:4]),int(temp_end1[4:6]),int(temp_end1[6:]))

	start_creation.append((temp_creation2-temp_start2).days)
	creation_sell.append((temp_end2-temp_creation2).days)
	start_sell.append((temp_end2-temp_start2).days)


# combine["start_creation"]=start_creation
# combine["creation_sell"]=creation_sell
# combine["start_sell"]=start_sell

# data preparation
combine["sold"] = combine["sold"].fillna(combine["sold"].mean())
combine["bought"] = combine["bought"].fillna(combine["bought"].mean())
combine["libor_rate"] = combine["libor_rate"].fillna(combine["libor_rate"].mean())

combine["pf_category"] = combine["pf_category"].astype('category')
combine["pf_category"].cat.categories = (0,1,2,3,4)

combine["type"] = combine["type"].astype('category')
combine["type"].cat.categories = (0,1,2,3,4,5,6,7)

combine["country_code"] = combine["country_code"].astype('category')
combine["country_code"].cat.categories = (0,1,2,3,4)

combine["currency"] = combine["currency"].astype('category')
combine["currency"].cat.categories = (0,1,2,3,4)

combine["office_id"] = combine["office_id"].astype('category')
combine["office_id"].cat.categories=(0,1)

combine["sold"]=combine["sold"]/combine["sold"].max()
combine["bought"]=combine["bought"]/combine["bought"].max()



consumetime=[]
for i in range(len(combine)):
# for i in range(1429,1434):
	temp=(combine["sold"].iloc[i] - combine["bought"].iloc[i])
	if(start_sell[i]):
		temp=temp/start_sell[i]
	consumetime.append(temp)

for i in range(len(consumetime)):
	consumetime = consumetime/max(consumetime)


createtime=[]
for i in range(len(combine)):
# for i in range(1429,1434):
	temp=(combine["sold"].iloc[i] - combine["bought"].iloc[i])
	if(start_creation[i]):
		temp=temp/start_creation[i]
	createtime.append(temp)


for i in range(len(createtime)):
	createtime = createtime/max(createtime)


soldtime=[]
for i in range(len(combine)):
# for i in range(1429,1434):
	temp=(combine["sold"].iloc[i] - combine["bought"].iloc[i])
	if(creation_sell[i]):
		temp=temp/creation_sell[i]
	soldtime.append(temp)

for i in range(len(soldtime)):
	soldtime = soldtime/max(soldtime)


combine["consumetime"]=consumetime
combine["createtime"]=createtime
combine["soldtime"]=soldtime


columns = ["country_code","soldtime","createtime","hedge_value",\
			"libor_rate","euribor_rate","currency","pf_category","office_id"]

combine["hedge_value"]=combine["hedge_value"].astype('category')
combine["hedge_value"].cat.categories=(0,1)


htrain=combine.loc[combine["hedge_value"].isnull()==False,columns]
htest=combine.loc[combine["hedge_value"].isnull()==True,columns]


htarget=htrain["hedge_value"]
hpredict=pd.DataFrame(htest["hedge_value"],index=htest.index)
htrain=htrain.drop("hedge_value",axis=1)
htest=htest.drop("hedge_value",axis=1)

clf=KNeighborsClassifier()
clf.fit(htrain,htarget)
prediction=clf.predict(htest)
hpredict["hedge_value"]=prediction

combine.loc[hpredict.index,"hedge_value"]=hpredict["hedge_value"]


columns = ["country_code","soldtime","createtime","hedge_value",\
			"libor_rate","euribor_rate"]

x_train=combine.loc[:len(train)-1,columns]
x_test=combine.loc[len(train):,columns]

# x_train=np.array(x_train)
# # y_train=np.array(y_train)
# # x_test=np.array(x_test)

clf=RandomForestRegressor(bootstrap=True,criterion='mse',max_depth=10)
eq=clf.fit(x_train,y_train)
print(eq.score(x_train,y_train))
prediction=eq.predict(x_test)

# imp=eq.feature_importances_
# for i in range(len(columns)):
# 	print("{} : {}".format(columns[i],imp[i]))

prediction=np.array(prediction)
prediction=prediction.round(6)
sub=pd.read_csv("sample_submission.csv")
sub["return"]=prediction
sub.to_csv("sub2.csv",index=False)

