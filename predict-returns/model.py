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
combine=pd.read_csv("combine.csv")
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
y_train=train["return"]
x_test=combine.loc[len(train):,columns]

# x_train=np.array(x_train)
# # y_train=np.array(y_train)
# # x_test=np.array(x_test)

clf=RandomForestRegressor(bootstrap=True,criterion='mse',max_depth=10)
eq=clf.fit(x_train,y_train)
print(eq.score(x_train,y_train))
prediction=eq.predict(x_test)

imp=eq.feature_importances_
for i in range(len(columns)):
	print("{} : {}".format(columns[i],imp[i]))

# prediction=np.array(prediction)
# prediction=prediction.round(6)
# sub=pd.read_csv("sample_submission.csv")
# sub["return"]=prediction
# sub.to_csv("sub2.csv",index=False)

