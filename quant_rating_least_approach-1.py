# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 18:36:53 2020

@author: d01730
"""

from quant_rating_least_approach_lib import *

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

#Faire correspondre le rating à la bonne date d'observations des données
df_data['RIGHT_RATING']=get_right_rating(df_data)

#Importing rating tables
table_rating_sp = pd.read_csv('code_sp.csv',sep=';',index_col=0)
table_rating_fitch = pd.read_csv('code_fitch.csv',sep=';', index_col=0)
table_rating_moodys = pd.read_csv('code_moodys.csv',sep=';', index_col=0)
table_rating_bloomberg = pd.read_csv('code_bloomberg.csv',sep=';', index_col=0)


# III - Preprocessing & Features Engineering

## 1- Construction des ratios

# Data scalling
 
df_data=scaler_data(df_data)

# Feature engineering

add_ratio(df_data)

df_data.replace([np.inf, -np.inf], np.nan, inplace = True)

df_data['RATING_ENCODED'] = encode_rating(df_data,'RIGHT_RATING','S&P',table_rating_sp,table_rating_moodys,table_rating_fitch,table_rating_bloomberg)

df_data['BLOOMBERG_ENCODED_RATING'] = encode_rating_bloom(df_data,'BB_1YR_DEFAULT_PROB',table_rating_bloomberg)

data_for_learning = get_data_for_learning(df_data)

data_for_learning['NUMBER_OF_NAN']= data_for_learning.isnull().sum(axis=1)

data_for_learning.dropna(subset=['Evolution_Rating'], inplace=True)
data_for_learning.dropna(subset=['DELTA_BLOOMBERG_ENCODED_RATING'], inplace=True)

numerical_features = list(data_for_learning.columns)

for elem in ['Evolution_Rating','RATING_ENCODED','BLOOMBERG_ENCODED_RATING']:
    
    numerical_features.remove(elem)
        
categorical_features = ['RATING_ENCODED']

#TimeSplit test = date >= 31/12/2016 and train = date < 2016
X_train = data_for_learning[data_for_learning.index.get_level_values(1)<=datetime(2016,12,31)][numerical_features+categorical_features]
y_train =data_for_learning[data_for_learning.index.get_level_values(1)<=datetime(2016,12,31)].Evolution_Rating.astype('int')
X_test = data_for_learning[data_for_learning.index.get_level_values(1)>datetime(2016,12,31)][numerical_features+categorical_features]
y_test = data_for_learning[data_for_learning.index.get_level_values(1)>datetime(2016,12,31)].Evolution_Rating.astype('int')

## 2- Preprocessing

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(missing_values=np.nan, strategy='median'))])
    
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent', missing_values=np.nan))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)])


## IV - Models & Results

# 1) MODEL 1: RandomForestClassifier

rf = RandomForestClassifier(max_depth=5, n_estimators=1000, max_features=30, n_jobs=-1,class_weight='balanced')

base_model_rf = Pipeline(steps=[('preprocessor', preprocessor),
                       ('classifier', rf)
                     ])
base_model_rf.fit(X_train,y_train)

print(get_quality_scores(base_model_rf,'Random_Forest_Classifier', X_train, X_test,y_train, y_test))
print(get_quality_scores_details(base_model_rf,'Random_Forest_Classifier', X_test,y_test))


# GridSearchCV : F1 score optimization

params_grid = {'classifier__n_estimators': [10,100,1000],             
               'classifier__max_features': [3,10,30],  
               'classifier__max_depth'   : [3,5,20]                     
              }

tscv = TimeSeriesSplit() 

def new_f1_scorer(y_true, y_predict):
    recall = sk_metrics.recall_score(y_true, y_predict, average=None)[0]
    precision = sk_metrics.precision_score(y_true, y_predict, average=None)[0]
    return 2 * precision*recall/precision+recall


new_scorer = make_scorer(new_f1_scorer,greater_is_better=True)

rf_optimized = GridSearchCV(base_model_rf, params_grid, cv=tscv, verbose = 1, scoring = new_scorer, n_jobs=-1)  

rf_optimized.fit(X_train,y_train)

print(get_quality_scores(rf_optimized,'Random_Forest_Classifier', X_train, X_test,y_train, y_test))
print(get_quality_scores_details(rf_optimized,'Random_Forest_Classifier', X_test,y_test))

## V - Resultats

# 1) Best Features : Permutation Importance

feature_names = X_train.columns

result = permutation_importance(base_model_rf, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)

#plot boxplot


sorted_idx = result.importances_mean.argsort()[len(result.importances_mean)-10:]
  
fig, ax = plt.subplots()

ax.boxplot(result.importances[sorted_idx].T,vert=False, labels=X_test.columns[sorted_idx])

ax.set_title("Permutation Importances (test set)")

fig.tight_layout()

plt.show()

#plot barh

sorted_idx = result.importances_mean.argsort()[len(result.importances_mean)-20:]

fig, ax = plt.subplots()

y_ticks = np.arange(1, 21)

ax.barh(y_ticks, result.importances_mean[sorted_idx])

ax.set_yticklabels(feature_names[sorted_idx])

ax.set_yticks(y_ticks)

ax.set_xlabel('Importance')

fig.tight_layout()

plt.show()

# 2) Comparaison au prédicteur de bloomberg

y_bloom_pred = list(np.clip(data_for_learning.DELTA_BLOOMBERG_ENCODED_RATING,-1,1).astype(int))
y_true = list(data_for_learning.Evolution_Rating.astype(int))



