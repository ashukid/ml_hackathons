import pandas as pd
import numpy as np
import datetime

train = pd.read_csv("train.csv")
target=train["return"]
test=pd.read_csv("test.csv")

train["return"].iloc[1665]=train["return"].mean()
# train["return"].iloc[7096]=train["return"].mean()
# train["return"].iloc[1731]=train["return"].mean()
# train["return"].iloc[1795]=train["return"].mean()

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






combine.to_csv("combine.csv",index=False)


