# x,xx,y,yy = train_test_split(x_train,target,test_size=0.1,random_state=100)

# clf=RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None)
# eq=clf.fit(x,y)
# print(eq.score(xx,yy))


# imp=eq.feature_importances_
# for i in range(len(columns)):
# 	print("{} : {}".format(columns[i],imp[i]))



# kf=KFold(n_splits=10)
# accuracy=[]
# predictions=[]
# for traini,testi in kf.split(x_train):
# 	train_data_x,train_data_y=x_train[traini],y_train[traini]
# 	test_data_x,test_data_y=x_train[testi],y_train[testi]
# 	dtrain=xgb.DMatrix(train_data_x,train_data_y)
# 	dvalid=xgb.DMatrix(test_data_x,test_data_y)
# 	dtest=xgb.DMatrix(x_test)

# 	watchlist=[(dtrain,'train'),(dvalid,'valid')]
# 	gbm=xgb.train(params,dtrain,1000,evals=watchlist,early_stopping_rounds=50,verbose_eval=1)
# 	predict=gbm.predict(dtest)
# 	score=gbm.predict(dvalid)
# 	accuracy.append(r2_score(test_data_y,score))
# 	predictions.append(predict)

# print(np.array(accuracy).mean())


# predictions=np.array(predictions)
# prediction=[]
# for i in range(len(predictions[0])):
# 	prediction.append(predictions[:,i].mean())


params={
    "objective":"reg:linear",     
    "learning_rate":0.1,
    "subsample":0.8,
    "colsample_bytree": 0.8,
#     'eval_metric':'auc',
    "max_depth":10,
    'silent':1,
    'nthread':3
}
