# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 10:53:54 2020

@author: d01730
"""


from quant_rating_library import *


# I - Data Importation

df_data = import_data('../../../1_Data/data_financial_statements.csv')


# II - Data Cleaning

replace_zero_by_nan(df_data)

## 1- Clean dates

LIST_COMPANY=df_data.index.get_level_values(0).unique().to_list()

for ticker in LIST_COMPANY:
    fill_forward(df_data.loc[(ticker,),])

keep_end_of_month_dates(df_data)


#Remove space character inside the ratings
df_data = remove_space(df_data,'RTG_SP_LT_LC_ISSUER_CREDIT')
df_data = remove_space(df_data,'RTG_MOODY_LONG_TERM')
df_data = remove_space(df_data,'RTG_FITCH_LT_ISSUER_DEFAULT')



#Remove outlook in the ratings 
df_data['RTG_SP_LT_LC_ISSUER_CREDIT']=remove_outlook(df_data,'RTG_SP_LT_LC_ISSUER_CREDIT')
df_data['RTG_FITCH_LT_ISSUER_DEFAULT']= remove_outlook(df_data,'RTG_FITCH_LT_ISSUER_DEFAULT')
df_data['RTG_MOODY_LONG_TERM']=remove_outlook(df_data,'RTG_MOODY_LONG_TERM')



#Importing rating tables
table_rating_sp = pd.read_csv('../../../2_Code/first_analysis/code_sp.csv',sep=';',index_col=0)
table_rating_fitch = pd.read_csv('../../../2_Code/first_analysis/code_fitch.csv',sep=';', index_col=0)
table_rating_moodys = pd.read_csv('../../../2_Code/first_analysis/code_moodys.csv',sep=';', index_col=0)







# IV - Preprocessing  

## 1- Construction des ratio

# Feature engineering
add_ratio(df_data)
df_data.replace([np.inf, -np.inf], np.nan, inplace = True)



#Data scalling 
#df_data=scaler_data(df_data)

#Data for learning
df_data['SP_ENCODED_RATING'] = encode_rating(df_data,'RTG_SP_LT_LC_ISSUER_CREDIT','S&P',table_rating_sp,table_rating_moodys,table_rating_fitch)
df_data['FITCH_ENCODED_RATING'] = encode_rating(df_data,'RTG_FITCH_LT_ISSUER_DEFAULT','Fitch',table_rating_sp,table_rating_moodys,table_rating_fitch)
df_data['MOODYS_ENCODED_RATING'] = encode_rating(df_data,'RTG_MOODY_LONG_TERM','Moodys',table_rating_sp,table_rating_moodys,table_rating_fitch)
df_data['RATING_ENCODED'] = rating_consensus(df_data,'SP_ENCODED_RATING','FITCH_ENCODED_RATING','MOODYS_ENCODED_RATING')

df_data.dropna(subset=['RATING_ENCODED'], inplace=True)

df_data.reset_index(inplace=True)

## V - Baseline Model

## 1- Splitting data

numerical_features = list(df_data.columns[df_data.dtypes==np.float64]) 
for elem in ['RATING_ENCODED','MOODYS_ENCODED_RATING','FITCH_ENCODED_RATING','SP_ENCODED_RATING']:
    numerical_features.remove(elem)


#TimeSplit test = date >= 31/12/2018 and train = date < 2018
X_train = df_data[df_data.date<datetime(2019,12,31)][numerical_features]
y_train =df_data[df_data.date<datetime(2019,12,31)].RATING_ENCODED
X_test = df_data[df_data.date>=datetime(2019,12,31)][numerical_features]
y_test = df_data[df_data.date>=datetime(2019,12,31)].RATING_ENCODED


## 2) Preprocessing

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(missing_values=np.nan, strategy='median'))])
    
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent', missing_values=np.nan))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features)])
        #('cat', categorical_transformer, encoded_categorical_features)])


## 3) Base Model

#,max_depth=30, n_estimators=1000, n_jobs=-1
rf = RandomForestRegressor(max_features=40,max_depth=30,n_estimators=1000,n_jobs=-1)

base_model_rf = Pipeline(steps=[('preprocessor', preprocessor),
                       ('regressor', rf)
                     ])
base_model_rf.fit(X_train,y_train)



## 4) Quality metrics of base model

print(get_quality_scores_var(base_model_rf,'Random_Forest_Regression', X_train, X_test,y_train, y_test))

print(get_quality_scores(base_model_rf,'Random_Forest_Regression', X_train, X_test,y_train, y_test))

fig=plot_confusion_matrix(base_model_rf, X_test, y_test)

## 5) Extract results

X = df_data[numerical_features]
y_predict = np.clip(np.round(base_model_rf.predict(X)),0,25)
y = np.array(df_data.RATING_ENCODED)
rating_predict = []
rating=[]
for i in range(len(y_predict)):
    rating_predict.append(f_decode_rating(y_predict[i],'S&P',table_rating_sp,table_rating_moodys,table_rating_fitch))
    rating.append(f_decode_rating(y[i],'S&P',table_rating_sp,table_rating_moodys,table_rating_fitch))
    
df_data['RATING_PREDICTED_CODE']=y_predict
df_data['RATING']=np.array(rating)
df_data['NEW_RATING_PREDICTED']=np.array(rating_predict)

df_data.to_csv('results.csv')
