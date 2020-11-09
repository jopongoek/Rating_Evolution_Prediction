# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 10:53:54 2020

@author: d01730
"""


from quant_rating_final_library import *


# I - Data Importation

df_data = import_data('data_financial_statements.csv')

# II - Data Cleaning

replace_zero_by_nan(df_data)

## 1- Clean dates

#Remplissage des lignes vides par l'historique disponible
LIST_COMPANY=df_data.index.get_level_values(0).unique().to_list()

range_date = list(pd.date_range(datetime(1999,12,31),datetime(2020,3,31),freq='Q'))


new_index = add_end_of_month_dates(df_data, range_date)

df_data=df_data.reindex(new_index).sort_index()


for ticker in LIST_COMPANY:
    fill_forward(df_data.loc[(ticker,),:])
    
    
keep_end_of_month_dates(df_data)

keep_trimester_month_dates(df_data)

#Remove space character inside the ratings
df_data = remove_space(df_data,'RTG_SP_LT_LC_ISSUER_CREDIT')
df_data = remove_space(df_data,'RTG_MOODY_LONG_TERM')
df_data = remove_space(df_data,'RTG_FITCH_LT_ISSUER_DEFAULT')



#Remove outlook in the ratings 
df_data['RTG_SP_LT_LC_ISSUER_CREDIT']=remove_outlook(df_data,'RTG_SP_LT_LC_ISSUER_CREDIT')
df_data['RTG_FITCH_LT_ISSUER_DEFAULT']= remove_outlook(df_data,'RTG_FITCH_LT_ISSUER_DEFAULT')
df_data['RTG_MOODY_LONG_TERM']=remove_outlook(df_data,'RTG_MOODY_LONG_TERM')



#Importing rating tables
table_rating_sp = pd.read_csv('code_sp.csv',sep=';',index_col=0)
table_rating_fitch = pd.read_csv('code_fitch.csv',sep=';', index_col=0)
table_rating_moodys = pd.read_csv('code_moodys.csv',sep=';', index_col=0)
table_rating_bloomberg = pd.read_csv('code_bloomberg.csv',sep=';', index_col=0)



# III - Preprocessing & Features Engineering

## 1- Construction des ratio

# Data scalling 
df_data=scaler_data(df_data)

# Feature engineering
add_ratio(df_data)
df_data.replace([np.inf, -np.inf], np.nan, inplace = True)

df_data['SP_ENCODED_RATING'] = encode_rating(df_data,'RTG_SP_LT_LC_ISSUER_CREDIT','S&P',table_rating_sp,table_rating_moodys,table_rating_fitch,table_rating_bloomberg)
df_data['FITCH_ENCODED_RATING'] = encode_rating(df_data,'RTG_FITCH_LT_ISSUER_DEFAULT','Fitch',table_rating_sp,table_rating_moodys,table_rating_fitch,table_rating_bloomberg)
df_data['MOODYS_ENCODED_RATING'] = encode_rating(df_data,'RTG_MOODY_LONG_TERM','Moodys',table_rating_sp,table_rating_moodys,table_rating_fitch,table_rating_bloomberg)
df_data['RATING_ENCODED'] = rating_rule(df_data.SP_ENCODED_RATING,df_data.FITCH_ENCODED_RATING,df_data.MOODYS_ENCODED_RATING)
df_data['BLOOMBERG_ENCODED_RATING'] = encode_rating(df_data,'RSK_BB_ISSUER_DEFAULT','Bloomberg',table_rating_sp,table_rating_moodys,table_rating_fitch,table_rating_bloomberg)


# CREATION Data for learning
data_for_learning = get_data_for_learning(df_data)

data_for_learning['NUMBER_OF_NAN']= data_for_learning.isnull().sum(axis=1)

data_for_learning.dropna(subset=['Evolution_Rating'], inplace=True)

#data_for_learning.dropna(subset=['DELTA_BLOOMBERG_ENCODED_RATING'], inplace=True)

numerical_features = list(data_for_learning.columns) 
for elem in ['Evolution_Rating','RATING_ENCODED','BLOOMBERG_ENCODED_RATING','DELTA_RATING_ENCODED','DELTA_BLOOMBERG_ENCODED_RATING']:
    numerical_features.remove(elem)
        
#categorical_features = ['RATING_ENCODED']


#TimeSplit test = date >= 31/12/2018 and train = date < 2018
X_train = data_for_learning[data_for_learning.index.get_level_values(1)<=datetime(2016,12,31)][numerical_features]#+categorical_features]
y_train =data_for_learning[data_for_learning.index.get_level_values(1)<=datetime(2016,12,31)].Evolution_Rating.astype('int')
X_test = data_for_learning[data_for_learning.index.get_level_values(1)>datetime(2016,12,31)][numerical_features]#+categorical_features]
y_test = data_for_learning[data_for_learning.index.get_level_values(1)>datetime(2016,12,31)].Evolution_Rating.astype('int')



## 2- Preprocessing

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(missing_values=np.nan, strategy='median'))])
    
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent', missing_values=np.nan))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features)])
        #('cat', categorical_transformer, categorical_features)])


## IV - Models & Results

# 1) MODEL 1: RandomForestClassifier

rf = RandomForestClassifier(max_depth=5, n_estimators=100, max_features=30, n_jobs=-1,class_weight='balanced')

base_model_rf = Pipeline(steps=[('preprocessor', preprocessor),
                       ('classifier', rf)
                     ])
base_model_rf.fit(X_train,y_train)


# 2) MODEL 2: GradientBoostingClassifier

#gb = GradientBoostingClassifier(max_depth=30, n_estimators=100, max_features=30)
#
#base_model_gb = Pipeline(steps=[('preprocessor', preprocessor),
#                       ('classifier', gb)
#                     ])
#base_model_gb.fit(X_train,y_train)


# 3) TimeSplitCrossValidation

params_grid = {'classifier__n_estimators': [10,100,1000],             
               'classifier__max_features': [3,10,30],  
               'classifier__max_depth'   : [3,5,20],                     
              }

# TIME SPLIT CROSS VALIDATION & SCORING

tscv = TimeSeriesSplit()   


# GRID SEARCH CV
def new_recall_scorer(y_true, y_predict):
    recall = sk_metrics.recall_score(y_true, y_predict,average=None)[0]
    return recall


new_scorer = make_scorer(new_recall_scorer,greater_is_better=True)

rf_optimized = GridSearchCV(base_model_rf, params_grid, cv=tscv, verbose = 1, scoring = new_scorer, n_jobs=-1)  

rf_optimized.fit(X_train,y_train)


## 4) Quality metrics of base model

print(get_quality_scores(base_model_rf,'Random_Forest_Classifier', X_train, X_test,y_train, y_test))
print(get_quality_scores_details(base_model_rf,'Random_Forest_Classifier', X_test,y_test))


print(get_quality_scores(base_model_gb,'Gradient_Boosting_Classifier', X_train, X_test,y_train, y_test))
print(get_quality_scores_details(base_model_gb,'Gradient_Boosting_Classifier', X_test,y_test))


# BLOOMBERG Model quality metrics

y_bloom_predict = np.array(data_for_learning.DELTA_BLOOMBERG_ENCODED_RATING).astype(int)
y = np.array(data_for_learning.Evolution_Rating).astype('int32')
print(get_quality_scores_bloom(y, y_bloom_predict))

## 5) Extract results

X = data_for_learning[numerical_features+categorical_features]
y_predict = np.clip(base_model_rf.predict(X),-1,1)
y = np.array(data_for_learning.Evolution_Rating).astype('int32')
   
data_for_learning['Evolution_PREDICTED']=y_predict

data_for_learning.to_csv('results.csv')

## 6) Feature Importance

feature_names = X_train.columns

tree_feature_importances = (
    base_model_rf.named_steps['classifier'].feature_importances_)

sorted_idx = tree_feature_importances.argsort()[len(tree_feature_importances)-20:]

y_ticks = np.arange(1, 21)
fig, ax = plt.subplots()
ax.barh(y_ticks, tree_feature_importances[sorted_idx])
ax.set_yticklabels(feature_names[sorted_idx])
ax.set_yticks(y_ticks)
ax.set_xlabel('Importance')
fig.tight_layout()
plt.show()
