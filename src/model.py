import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

data_dir = '../data/'

# change range to [0,1] 
def do_normalizing(train_X,valid_X,test_X):
	train_X = train_X/np.max(train_X,axis=0)[None:]
	valid_X = valid_X/np.max(valid_X,axis=0)[None:]
	test_X  = test_X/np.max(test_X,axis=0)[None:]
	
	return train_X,valid_X,test_X

# load the data
def load_data(train,valid,test,norm):
	train_data = pd.read_csv(train,sep='	').values
	valid_data = pd.read_csv(valid,sep='	').values
	test_data  = pd.read_csv(test, sep='	').values
	train_X, train_y = train_data[:,1:], train_data[:,0] 
	valid_X, valid_y = valid_data[:,1:], valid_data[:,0]
	test_X,  test_y  =  test_data[:,1:],  test_data[:,0]
	if norm:
		train_X,valid_X,test_X = do_normalizing(train_X,valid_X,test_X)
	
	return train_X,train_y,valid_X,valid_y,test_X,test_y

# linear regression with l2 regularization
def linearReg(X,y,l2):
	# model = linear_model.LinearRegression()
	model = linear_model.Ridge(alpha = l2)
	model.fit(X,y)
	
	return model

# function to return perfomance measures
def testModel(model,X,y_true):
	y_pred = model.predict(X)
	rmse   = np.sqrt(mean_squared_error(y_true, y_pred))
	r2_val = r2_score(y_true, y_pred)
	
	return y_pred,rmse,r2_val

train = data_dir + 'train.csv'
valid = data_dir + 'valid.csv'
test  = data_dir + 'test.csv'
print ('Loading the data...')
train_X,train_y,valid_X,valid_y,test_X,test_y = load_data(train,valid,test,True)
print ('Train_data details: ',train_X.shape, train_y.shape)
print ('Valid_data details: ',valid_X.shape, valid_y.shape)
print ('Test_data  details: ',test_X.shape , test_y.shape)
print ('Reading the data Done!')

print('\nTraining a LinearRegression Model..')
model = linearReg(train_X,train_y,0)
print('Training Done!')

print('Without regularization:')
y_pred,rmse,r2_val = testModel(model,train_X,train_y)
print('r2_score on train_data : {}'.format(round(r2_val,3)))
y_pred,rmse,r2_val = testModel(model,valid_X,valid_y)
print('r2_score on valid_data : {}'.format(round(r2_val,3)))
y_pred,rmse,r2_val = testModel(model,test_X,test_y)
print('r2_score on test_data  : {}'.format(round(r2_val,3)))


print('\nWith regularization:')
print('Searching for optimal alpha(l2)...')
best_l2 = 0				# start with no regularization as optimal
best_model   = model 	# start with model with out regularization is optimal
best_valid_r2 = r2_val	# start with r2_score for valid with out regularization is best
l2_vals = [1,10,100,200,300,400,500,600,700,800,900,1000] # very high values of alpha to avoid overfitting...
for l2 in l2_vals:										  # since we have very low training data
	model = linearReg(train_X,train_y,l2)
	# print('\nWith regularization of alpha: {}'.format(l2))
	# y_pred,rmse,r2_val = testModel(model,train_X,train_y)
	# print('r2_score on train_data : {}'.format(round(r2_val,3)))
	y_pred,rmse,r2_val = testModel(model,valid_X,valid_y)
	# print('r2_score on valid_data : {}'.format(round(r2_val,3)))
	if best_valid_r2 < r2_val:
		best_l2 = l2
		best_model = model
		best_valid_r2 = r2_val

print('\nSearching for optimal alpha(l2) Done!')
print('Optimal alpha(l2): {}'.format(best_l2))

print('\nWith regularization of alpha: {}'.format(best_l2))
y_pred,rmse,r2_val = testModel(best_model,train_X,train_y)
print('r2_score on train_data : {}'.format(round(r2_val,3)))
y_pred,rmse,r2_val = testModel(best_model,valid_X,valid_y)
print('r2_score on valid_data : {}'.format(round(r2_val,3)))
y_pred,rmse,r2_val = testModel(best_model,test_X,test_y)
print('r2_score on test_data  : {}'.format(round(r2_val,3)))

# print('RMSE     : {}'.format(round(rmse,3)))
# print('r2_score : {}'.format(round(r2_val,3)))

# print('RMSE     : {}'.format(round(rmse,3)))
# print('r2_score : {}'.format(round(r2_val,3)))

# print('RMSE     : {}'.format(round(rmse,3)))
# print('r2_score : {}'.format(round(r2_val,3)))

# from sklearn.ensemble import RandomForestRegressor
# regr = RandomForestRegressor(random_state=42)
# regr.fit(train_data.iloc[:,0:features],train_data['Runs'])
# test_pred = regr.predict(test_data.iloc[:,0:features])

# import matplotlib.pyplot as plt
# plt.plot(test_data["Runs"],test_pred)