# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 10:53:54 2020

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

df_data.dropna(subset=['RATING_ENCODED'], inplace=True)


## V - Baseline Model

## 1- Splitting data

numerical_features = list(df_data.columns[df_data.dtypes==np.float64]) 
for elem in ['RATING_ENCODED','BB_1YR_DEFAULT_PROB','ALTMAN_Z_SCORE']:
    numerical_features.remove(elem)


#TimeSplit test = date >= 31/12/2018 and train = date < 2018
X_train = df_data[df_data.index.get_level_values(1)<datetime(2016,12,31)][numerical_features]
y_train =df_data[df_data.index.get_level_values(1)<datetime(2016,12,31)].RATING_ENCODED
X_test = df_data[df_data.index.get_level_values(1)>=datetime(2016,12,31)][numerical_features]
y_test = df_data[df_data.index.get_level_values(1)>=datetime(2016,12,31)].RATING_ENCODED


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

rf = RandomForestClassifier(max_features=10, max_depth=20, n_estimators=1000, n_jobs=-1)

base_model_rf = Pipeline(steps=[('preprocessor', preprocessor),
                       ('classifier', rf)
                     ])
base_model_rf.fit(X_train,y_train)

print(get_quality_scores(base_model_rf,'Random_Forest_Classifier', X_train, X_test,y_train, y_test))


# GridSearchCV : F1 score optimization

params_grid = {'classifier__n_estimators': [10,100,1000],             
               'classifier__max_features': [10,20,30],  
               'classifier__max_depth'   : [5,10,20]                     
              }

tscv = TimeSeriesSplit() 

def new_f1_scorer(y_true, y_predict):
    recall = sk_metrics.recall_score(y_true, y_predict, average='macro')
    precision = sk_metrics.precision_score(y_true, y_predict, average='macro')
    return 2 * precision*recall/precision+recall


new_scorer = make_scorer(new_f1_scorer,greater_is_better=True)

rf_optimized = GridSearchCV(base_model_rf, params_grid, cv=tscv, verbose = 1, scoring = new_scorer, n_jobs=-1)  

rf_optimized.fit(X_train,y_train)

print(get_quality_scores(rf_optimized,'Random_Forest_Classifier', X_train, X_test,y_train, y_test))

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
