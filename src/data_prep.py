import pandas as pd
import numpy as np

numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
def get_noNumcols(data):
	all_cols   = list(data.columns)
	tmp_df     = data.select_dtypes(include=numerics)
	num_cols   = tmp_df.columns
	nonum_cols = list(set(all_cols) - set(num_cols))
	
	return nonum_cols

data_dir  = '../data/'
tar_col   = 'per_capita_exp_total_py'
data      = pd.read_excel(data_dir + 'DATA.xlsx', sheet_name='Data')
print('The shape of data: {}'.format(data.shape))

######################### DATA_PREPROCESSING #######################################
missing_cutoff  = 0.9
majority_cutoff = 0.9

data.replace(to_replace='N/A', value=np.nan, inplace=True)			# replace the string 'N/A' with nan

del_cols    = []													# list to columns to be deleted
data_cols   = list(data.columns)									# current columns in the data
nonum_cols1 = get_noNumcols(data)									# get non-numeric columns

# The above function returns ['aco42', 'aco19', 'aco30', 'aco_state', 'aco_name', 'aco28', 'aco20', 
# 'aco40', 'aco41', 'dm_comp', 'aco27', 'aco_num', 'aco18'], but one can see that only
# ['aco_num','aco_name','aco_state'] are category(non-numeric) cols after reading the varaible description
nonum_cols2 = ['aco_num','aco_name','aco_state']

change_type_cols = list(set(nonum_cols1)-set(nonum_cols2))			# change the columns for object type to float64
for col in change_type_cols:										# as they type should have been that
	data[col] = pd.to_numeric(data[col],errors='coerce')

missing_per = data.isnull().sum(axis=0)/float(data.shape[0])		# check the fraction of missing values for each column

if missing_per[data_cols.index(tar_col)] > 0:						# if missing values there in target column..
	pred_data = data.loc[data[tar_col].isnull()]					# .. keep those rows aside
	data      = data[np.isfinite(data[tar_col])]					# delete them from the data

for idx,col in enumerate(data_cols):
	if missing_per[idx] > 0 and col != tar_col:
		if  missing_per[idx] >= missing_cutoff:						# delete column if (>100*cutoff%) rows values are missing
			del_cols.append(col)
		else:														# replace them with mode for non-numeric cols
			if col in nonum_cols2:
				data[col].fillna(data[col].mode()[0],inplace=True)
			else:													# replace them with mean for numerics cols
				data[col].fillna(data[col].mean(),inplace=True)

## Delete columns if majority(>100*cutoff%) rows have same value
## model won't learn anything from a feature if there isn't much change
for col in data_cols:
	per = data[col].value_counts(normalize=True)
	if max(per) >= majority_cutoff:
		del_cols.append(col)

del_cols.extend(nonum_cols2)										# delete category variables -- to many levels
del_cols = list(set(del_cols))
print ('Columns to be deleted: ',del_cols)

keep_cols = list(set(data_cols)-set(del_cols))						# columns to be kept in the given data
keep_cols.insert(0,keep_cols.pop(keep_cols.index(tar_col)))			# move target variable to the first position(column) 
data = data.ix[:,keep_cols]											# data with only requried columns

data.to_csv(data_dir+'cleaned_cat.csv',sep='	',encoding='utf-8',index=False) # save the data

# One-time call function -- since we don't want the split of data to change
# Split the data into train(80%), valid(10%), test(10%)
# length of valid = ratio*length of data

data = data.sample(frac=1).reset_index(drop=True)
train_data = data.iloc[:1400, :]
valid_data = data.iloc[1400:1500, :]
test_data  = data.iloc[1500:, :] 
train_data.to_csv(data_dir+'train.csv',sep='	',encoding='utf-8',index=False)
valid_data.to_csv(data_dir+'valid.csv',sep='	',encoding='utf-8',index=False)
test_data.to_csv(  data_dir+'test.csv',sep='	',encoding='utf-8',index=False)
pred_data.to_csv(  data_dir+'pred.csv',sep='	',encoding='utf-8',index=False)

print('The shape of total_data: {}'.format(data.shape))
print('The shape of train_data: {}'.format(train_data.shape))
print('The shape of valid_data: {}'.format(valid_data.shape))
print('The shape of test_data : {}'.format(test_data.shape))
# print('The shape of pred_data : {}'.format(pred_data.shape))


# Dealing with categorical columns --  cluster them based on the frequency
# for col in nonum_cols2:
# 	if col != 'aco_state':