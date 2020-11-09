# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 18:38:30 2020

@author: d01730
"""


import numpy as np
import math
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib as mp
import seaborn as sns
import matplotlib.backends.backend_pdf as mbbp
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
import sklearn.metrics as sk_metrics
from sklearn.metrics import confusion_matrix, make_scorer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingClassifier
import multiprocessing as mpp
import timeit as ti
from sklearn.inspection import permutation_importance


def import_data(fichier_excel):
    df_data                                    = pd.read_csv(fichier_excel,sep=',', parse_dates= True)
    df_data['date']                            = pd.to_datetime(df_data['date'],dayfirst=True)
    df_data['RTG_FITCH_LT_ISSUER_DFLT_RTG_DT'] = pd.to_datetime(df_data['RTG_FITCH_LT_ISSUER_DFLT_RTG_DT'], dayfirst=True)
    df_data['RTG_FITCH_OUTLOOK_DT']            = pd.to_datetime(df_data['RTG_FITCH_OUTLOOK_DT'],dayfirst=True)
    df_data['RTG_MOODY_LONG_TERM_DATE']        = pd.to_datetime(df_data['RTG_MOODY_LONG_TERM_DATE'],dayfirst=True)
    df_data['RTG_SP_OUTLOOK_DT']               = pd.to_datetime(df_data['RTG_SP_OUTLOOK_DT'],dayfirst=True)
    df_data['RTG_SP_LT_LC_ISS_CRED_RTG_DT']    = pd.to_datetime(df_data['RTG_SP_LT_LC_ISS_CRED_RTG_DT'],dayfirst=True)
    df_data.set_index(['ticker','date'], inplace = True)
    return df_data

def replace_zero_by_nan(df):
    """fonction qui remplace les '0' par NaN sauf pour les champs avec des pourcentages
    """
    champs_pourcentage= ['EARN_YLD','BB_1YR_DEFAULT_PROB','EQY_DVD_YLD_IND','WACC','WACC_RETURN_ON_INV_CAPITAL']
    for x in df.columns:
        if x not in champs_pourcentage:
            df[x].replace(0,np.nan,inplace=True)
    return df

def replace_infinite_by_nan(df):
    """
        fonction qui remplace les données infinies par NaN
    """
    for x in df.columns:
        df[x].replace(np.inf,np.nan, inplace=True)
    return df

def fill_forward(df):
    """ fills missing data with last available data
    """
    df.fillna(method = 'ffill', inplace = True)

    

def add_end_of_month_dates(df, list_dates_fin_mois):
    """

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.
    list_dates_fin_mois : TYPE
        DESCRIPTION.

    Returns
    -------
    list_index : TYPE
        DESCRIPTION.

    """
    list_index = list(df.index)
    for ticker in df.index.get_level_values(0).unique().to_list():
        for date in list_dates_fin_mois:
            if((ticker,date)not in list_index):
                list_index.append((ticker,date))
    return list_index
                
    

def keep_end_of_month_dates(df):
    """ remove not end of month
    """
    list_dates = df.index.get_level_values(1).unique().to_list()
    #end_month_dates = [d for d in list_dates if d.is_month_end]
    df.drop(index=[d for d in list_dates if(d.month not in [3,6,9,12])], level=1,inplace=True)


def keep_trimester_month_dates(df):
    """
    Enleve toutes les dates non trimestrielles ex : 31/01/2006
    Parameters
    ----------
    df : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    list_dates = df.index.get_level_values(1).unique().to_list()
    
    df.drop(index=[d for d in list_dates if not(d.is_month_end)], level=1,inplace=True)

 
def remove_space(df,feature):
    """The function remove spaces in values of the feature in parameter
    """
    df[feature] = df[feature].str.replace(" ","")
    return df

def remove_outlook(df, rating):
    """
    Enlève les outlook des ratings ex: A*+ devient A

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.
    rating : TYPE
        DESCRIPTION.

    Returns
    -------
    rating_new_format : TYPE
        DESCRIPTION.

    """
    rating_old_format = list(df[rating])
    rating_new_format = []
    tmp_rating = ""
    for rating in rating_old_format:
        tmp_rating = rating
        if("*+" in str(rating)):
            tmp_rating = rating.replace("*+","")
        elif("*-" in str(rating)):
            tmp_rating = rating.replace("*-","")
        elif("*" in str(rating)):
            tmp_rating = rating.replace("*","")
        elif("WD" in str(rating) or "RD" in str(rating) or 'SD' in str(rating)):
            tmp_rating = "D"
        elif("(P)" in str(rating)):
            tmp_rating = rating.replace("(P)","")
        elif("NR" in str(rating)):
            tmp_rating = np.nan
        rating_new_format.append(tmp_rating)
    return rating_new_format



def get_pointer_change_rating(df, rating_agency='S&P', last_date = datetime(2020,4,1)):
    """returns the list of tuple (dates of rating changement,quaterly_date,rating) 
    """
    rating_tmp = str(np.nan)
    
    switcher_format = {
        'S&P': (list(df.RTG_SP_LT_LC_ISSUER_CREDIT),list(df.RTG_SP_LT_LC_ISS_CRED_RTG_DT)),
        'Moodys': (list(df.RTG_MOODY_LONG_TERM),list(df.RTG_MOODY_LONG_TERM_DATE)),
        'Fitch': (list(df.RTG_FITCH_LT_ISSUER_DEFAULT),list(df.RTG_FITCH_LT_ISSUER_DFLT_RTG_DT))
    }
    
    list_rating,list_date_rating = switcher_format.get(rating_agency)
    list_date_change = []
    list_rating_change = []
    
    for i, rating in enumerate(list_rating):
        if(str(list_date_rating[i])=='NaT'):
            i=i+1
        else:
            if(rating_tmp!=str(rating) and str(rating)!=str(np.nan)): 
                rating_tmp = rating
                list_date_change.append(list_date_rating[i])
                list_rating_change.append(rating)
    list_date_change.append(last_date)       
    return list_date_change, list_rating_change


def construct_tuple(list_date_change,list_rating_change):
    
    list_tuple_chg_rating = []
    
    for i in range(len(list_date_change)-1):
        
        list_tuple_chg_rating.append((list_date_change[i],list_date_change[i+1],list_rating_change[i]))
    
    return list_tuple_chg_rating
        
        

def get_right_rating(df):
    
    list_ticker = sorted(list(set(df.index.get_level_values(0))))
    right_rating = []
    
    for ticker in list_ticker:
        
        fs_ticker = df[df.index.get_level_values(0)==ticker]
        
        list_date_change, list_rating_change = get_pointer_change_rating(fs_ticker)
        
        list_tuple_chg_rating =construct_tuple(list_date_change, list_rating_change)
        
        list_date = list(fs_ticker.index.get_level_values(1))
        
        if len(list_tuple_chg_rating)==0:
                
                for i in range(fs_ticker.shape[0]):
                    
                    right_rating.append(np.nan)
                
        else:
        
            for date in list_date:
                
                if date < list_tuple_chg_rating[0][0]:
                    
                    right_rating.append(np.nan)
                
                else:
                    
                    for element in list_tuple_chg_rating:
                        
                        if element[0]<=date<element[1]:
                            
                            right_rating.append(element[2])
                            
                            break
                    
    return right_rating


def f_encode_rating(rating,agence_notation,table_rating_sp,table_rating_moodys,table_rating_fitch, table_rating_bloomberg):
    switcher_rating = {
        'S&P': table_rating_sp,
        'Moodys': table_rating_moodys,
        'Fitch': table_rating_fitch,
        'Bloomberg': table_rating_bloomberg
    }
    if(pd.isnull(rating)):
        return(np.nan)
    else:
        table_rating=switcher_rating.get(agence_notation,"nothing")
        return float(table_rating.loc[rating].code)

def f_decode_rating(number,agence_notation,table_rating_sp,table_rating_moodys,table_rating_fitch, table_rating_bloomberg):
    switcher_rating = {
        'S&P': table_rating_sp,
        'Moodys': table_rating_moodys,
        'Fitch': table_rating_fitch,
        'Bloomberg': table_rating_bloomberg
    }
    table_rating = switcher_rating.get(agence_notation, "nothing")
    return(table_rating[table_rating.code==int(np.round(number))].index[0])


def default_probability_to_rating(default_probability, table_rating):
    list_of_intervals = [(borne_inf,borne_sup) for borne_inf, borne_sup in zip(table_rating.borne_inf,table_rating.borne_sup)]
    for i,interval in enumerate(list_of_intervals):
        if(interval[0]<default_probability<=interval[1]):
            return table_rating.code[i]
    return np.nan
    


def add_ratio(df):   
    df['RATIO_BEST_GROSS_MARGIN']= (df.SALES_REV_TURN - df.ARD_COST_OF_GOODS_SOLD)/df.SALES_REV_TURN
    df['RATIO_BEST_PX_SALES']= df.CUR_MKT_CAP/df.SALES_REV_TURN
    df['RATIO_BOOK_VAL_PER_SH']= df.TOT_COMMON_EQY/df.BS_SH_OUT
    df['RATIO_COM_EQY_TO_TOT_ASSET']= df.TOT_COMMON_EQY/df.BS_TOT_ASSET
    df['RATIO_CURRENT_EV_TO_12M_SALES']= df.CUR_MKT_CAP - (df.BS_ST_BORROW+df.BS_LT_BORROW) + df.BS_CASH_NEAR_CASH_ITEM/df.SALES_REV_TURN
    df['RATIO_CURRENT_EV_TO_T12M_EBIT']= df.CUR_MKT_CAP - (df.BS_ST_BORROW+df.BS_LT_BORROW) + df.BS_CASH_NEAR_CASH_ITEM/df.EBIT
    df['RATIO_CURRENT_EV_TO_T12M_EBITDA']= df.CUR_MKT_CAP - (df.BS_ST_BORROW+df.BS_LT_BORROW) + df.BS_CASH_NEAR_CASH_ITEM/df.EBITDA
    df['RATIO_OPER_INC_PER_SH']= df.IS_OPER_INC/df.BS_SH_OUT
    df['RATIO_OPER_MARGIN']= df.IS_OPER_INC/df.SALES_REV_TURN
    df['RATIO_PE']= (df.CUR_MKT_CAP/df.TOTAL_EQUITY)/df.IS_EPS
    df['RATIO_PX_TO_BOOK']= df.PX_LAST/df.BOOK_VAL_PER_SH
    df['RATIO_PX_TO_CASH_FLOW']= df.CUR_MKT_CAP/df.CF_CASH_FROM_OPER
    df['RATIO_PX_TO_FREE_CASH_FLOW']= df.CUR_MKT_CAP/df.CF_CASH_FROM_OPER
    df['RATIO_PX_TO_SALES']= df.CUR_MKT_CAP/df.SALES_REV_TURN
    df['RATIO_PX_TO_TANG_BV_PER_SH']= df.CUR_MKT_CAP/df.BS_NET_FIX_ASSET
    df['RATIO_RETURN_COM_EQY']= df.NET_INCOME/df.TOT_COMMON_EQY
    df['RATIO_RETURN_ON_ASSET']= df.NET_INCOME/df.BS_TOT_ASSET
    df['RATIO_REVENUE_PER_SH']= df.SALES_REV_TURN/df.IS_AVG_NUM_SH_FOR_EPS
    df['RATIO_ROC_WACC']= df.WACC_NOPAT/df.WACC_RETURN_ON_INV_CAPITAL
    df['RATIO_TOT_DEBT_TO_TOT_EQY']= (df.BS_ST_BORROW+df.BS_LT_BORROW)/df.TOTAL_EQUITY
    df['RATIO_CFO_TO_SALES']= df.CF_CASH_FROM_OPER/df.SALES_REV_TURN
    df['RATIO_FREE_CASH_FLOW_MARGIN']= df.CF_CASH_FROM_OPER/df.SALES_REV_TURN
    df['RATIO_FREE_CASH_FLOW_YIELD']= df.CF_CASH_FROM_OPER/df.CUR_MKT_CAP
    df['RATIO_WORKING_CAPITAL_TO_TOTAL_ASSETS']= df.WORKING_CAPITAL/df.BS_TOT_ASSET
    df['RATIO_TOTAL_LIABILITES_TO_TOTAL_ASSETS']= df.BS_TOTAL_LIABILITIES/df.BS_TOT_ASSET
    df['RATIO_NET_PROFIT_TO_TOTAL_ASSETS']= df.NET_INCOME/df.BS_TOT_ASSET
    df['RATIO_EQUITY_TO_TOTAL_ASSETS']= df.TOTAL_EQUITY/df.BS_TOT_ASSET
    df['RATIO_NET_PROFIT_TO_EQUITY']= df.NET_INCOME/df.TOTAL_EQUITY
    df['RATIO_QUICK']= (df.BS_CASH_NEAR_CASH_ITEM+df.BS_MKT_SEC_OTHER_ST_INVEST+df.BS_ACCT_NOTE_RCV)/df.BS_CUR_LIAB
    df['RATIO_NET_PROFIT_MARGIN']= df.NET_INCOME/df.SALES_REV_TURN
    df['RATIO_WORKING_CAPITAL_TO_SALES']= df.WORKING_CAPITAL/df.SALES_REV_TURN
    df['RATIO_GROSS_PROFITABILITIY']= df.ARD_COST_OF_GOODS_SOLD/df.SALES_REV_TURN
    df['RATIO_CURRENT_ASSETS_TO_TOTAL_ASSETS']= df.BS_CUR_ASSET_REPORT/df.BS_TOT_ASSET
    df['RATIO_EBIT_TO_TOTAL_ASSETS']= df.EBIT/df.BS_TOT_ASSET
    df['RATIO_EBIT_TO_SALES']= df.EBIT/df.SALES_REV_TURN
    df['RATIO_CASH']= df.CASH_AND_MARKETABLE_SECURITIES/df.BS_CUR_LIAB
    df['RATIO_SALES_TO_LONG_TERM_ASSETS']= df.SALES_REV_TURN/df.BS_NET_FIX_ASSET
    df['RATIO_LONG_TERM_DEBT_TO_EQUITY']= df.BS_LT_BORROW/df.TOTAL_EQUITY
    df['RATIO_CASH_TO_TOT_ASSET']= df.BS_CASH_NEAR_CASH_ITEM/df.BS_TOT_ASSET
    df['RATIO_CUR']= df.BS_CUR_ASSET_REPORT/df.BS_CUR_LIAB
    df['RATIO_EBITDA_TO_NET_INTEREST']= df.EBITDA/df.IS_INT_EXPENSE+df.CAPITAL_EXPEND+df.IS_INT_INC
    df['RATIO_FCF_TO_TOTAL_DEBT']= df.CF_CASH_FROM_OPER/df.BS_ST_BORROW + df.BS_LT_BORROW
    df['RATIO_LT_DEBT_TO_TOT_ASSET']= df.BS_LT_BORROW/df.BS_TOT_ASSET
    df['RATIO_SALES_TO_TOT_ASSET']= df.SALES_REV_TURN/df.BS_TOT_ASSET
    df['RATIO_TOT_DEBT_TO_EBITDA']= df.BS_ST_BORROW + df.BS_LT_BORROW/df.EBITDA
    df['RATIO_TOT_DEBT_TO_TOT_ASSET']= df.BS_ST_BORROW + df.BS_LT_BORROW/df.BS_TOT_ASSET
    df['RATIO_NET_DEBT_TO_EBITDA']= (df.BS_ST_BORROW + df.BS_LT_BORROW)-df.BS_CASH_NEAR_CASH_ITEM/df.EBITDA
    df['RATIO_NET_DEBT_TO_TOTAL_ASSET']= (df.BS_ST_BORROW + df.BS_LT_BORROW)-df.BS_CASH_NEAR_CASH_ITEM/df.BS_TOT_ASSET
    df['RATIO_NET_DEBT_TO_TOTAL_EQUITY']= (df.BS_ST_BORROW+df.BS_LT_BORROW)-df.BS_CASH_NEAR_CASH_ITEM/df.TOTAL_EQUITY
    return df

def scaler_data(df):
    df.reset_index(inplace=True)
    list_dates= list(set(df.date))
    numerical_features = list(df.columns[df.dtypes==np.float64])
    for date in list_dates:
        for feature in numerical_features:
            med = np.nanmedian(df[df.date==date][feature])
            stand_dev = np.nanstd(df[df.date==date][feature])
            df.loc[df.date==date,feature]=(df.loc[df.date==date,feature]-med)/stand_dev
    df.set_index(['ticker','date'], inplace = True)
    return df



def get_past_yearly_date(date):
    """return the date one year from input date
    """
    y=date.year-1
    m=date.month
    d=date.day
    return datetime(y,m,d)

        
        
    
def get_ahead_yearly_date(date):
    """return the date one year ahead input date
    """
    y=date.year+1
    m=date.month
    d=date.day
    return datetime(y,m,d)

def get_past_quaterly_date(date,first_quaterly=datetime(1999,12,31), last_quaterly=datetime(2019,12,31)):
    """return the quaterly date just before the rating date
    """
    nearest_date = ""
    range_date   = pd.date_range(first_quaterly,last_quaterly,freq='Q')
    nearest_date = min([i for i in range_date if i < date], key=lambda x: abs(x - date))
    return nearest_date


def get_future_quaterly_date(date,first_quaterly=datetime(1999,12,31), last_quaterly=datetime(2019,12,31)):
    """return the quaterly date just after the rating date
    """
    nearest_date = ""
    range_date   = pd.date_range(first_quaterly,last_quaterly,freq='Q')
    nearest_date = min([i for i in range_date if i > date], key=lambda x: abs(x - date))
    return nearest_date


def get_pointer(df, start_date=datetime(2000,12,31), end_date=datetime(2019,3,31)):
    """returns the list of tuple (date,date-1y,date+1y) 
    """
    list_dates = df.index.get_level_values(0).unique().to_list()
    not_valid_dates = []
    for x in list_dates:
        if(x < start_date or x > end_date):
            not_valid_dates.append(x)
    
    for ele in not_valid_dates:
        list_dates.remove(ele)
        
    list_pointer_date=[]
    for date in list_dates:
            list_pointer_date.append((date,get_past_yearly_date(date),get_ahead_yearly_date(date)))   
    return list_pointer_date


def get_data_for_learning(df):
    """ converts raw data into target format data for learning
    
        format convention
        ------------------
        columns: d2, features (d2), features (d2)-features(d1), d2-d1, rating(d1), rating(d2)
        with:
          - d1 :
          - d2 :
          - features(d2)
    """
    numerical_features = list(df.columns[df.dtypes==np.float64]) 
    
    for elem in ['MOODYS_ENCODED_RATING','FITCH_ENCODED_RATING','SP_ENCODED_RATING','ALTMAN_Z_SCORE']:
         
        numerical_features.remove(elem)
    
    delta_features = ['DELTA_'+feature for feature in numerical_features]
    
    #Création d'un dataframe avec nb de ligne au début
    
    list_dates        = df.index.get_level_values(1).unique().to_list()
    
    data_for_learning = pd.DataFrame(index=df.index, columns= ['Evolution_Rating']+numerical_features+delta_features)

    for date in list_dates[4:-4]:
        date_past             = get_past_yearly_date(date)
        date_ahead            = get_ahead_yearly_date(date)
        F_date                = df.loc[(slice(None),date),numerical_features]
        F_date_past           = df.loc[(slice(None),date_past),numerical_features]
        rating_encoded        = df.loc[(slice(None),date),'RATING_ENCODED']
        rating_encoded_ahead  = df.loc[(slice(None),date_ahead),'RATING_ENCODED']
        delta_F               = F_date.values-F_date_past.values

        data_for_learning.loc[(slice(None),date), numerical_features]   = F_date
        data_for_learning.loc[(slice(None),date), delta_features]       = delta_F
        data_for_learning.loc[(slice(None),date),'Evolution_Rating']    = np.clip(rating_encoded_ahead.values-rating_encoded.values,-1,1)

    return data_for_learning



def plot_confusion_matrix(estimator, X, y,title='',class_names=None, figsize = (10,7), fontsize=14):
    pred_class=np.clip(estimator.predict(X),-1,1)
    confusion_mat= confusion_matrix(y, pred_class)
    if class_names is None:
        class_names=range(confusion_mat.shape[0])
    df_cm = pd.DataFrame(confusion_mat, index=class_names, columns=class_names)
    fig = plt.figure(figsize=figsize)
    plt.title(title)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return fig


def get_quality_scores(model,model_name, X_train, X_test, y_train, y_test):
    """return a dataframe of the scoring of the model
    """
    res         = pd.DataFrame(np.zeros((10,1)))
    res.columns = [model_name]
    res.index   = ['Train Accuracy','Train Average Recall', 'Train Average Precision', 'Train Average F1 Score','Train Mean Absolute Error', 'Test Accuracy','Test Average Recall', 'Test Average Precision', 'Test Average F1 Score','Test Absolute Mean Error']
    #train
    X_train_predict = np.clip(model.predict(X_train),0,25)   
    res.iloc[0,0] = '{:.2%}'.format(sk_metrics.accuracy_score(X_train_predict, y_train, normalize=True))
    res.iloc[1,0] = '{:.2%}'.format(sk_metrics.recall_score(y_train, X_train_predict, average='macro'))
    res.iloc[2,0] = '{:.2%}'.format(sk_metrics.precision_score(y_train, X_train_predict, average='macro'))
    res.iloc[3,0] = '{:.2%}'.format(sk_metrics.f1_score(y_train, X_train_predict, average='macro'))
    res.iloc[4,0] = sk_metrics.mean_absolute_error(y_train, X_train_predict)
    #test
    X_test_predict = np.clip(model.predict(X_test),0,25)
    res.iloc[5,0] = '{:.2%}'.format(sk_metrics.accuracy_score(X_test_predict, y_test, normalize=True))
    res.iloc[6,0] = '{:.2%}'.format(sk_metrics.recall_score(y_test, X_test_predict,average='macro'))
    res.iloc[7,0] = '{:.2%}'.format(sk_metrics.precision_score(y_test, X_test_predict, average='macro'))
    res.iloc[8,0] = '{:.2%}'.format(sk_metrics.f1_score(y_test, X_test_predict, average='macro'))
    res.iloc[9,0] = sk_metrics.mean_absolute_error(y_test, X_test_predict)
    return res


def get_quality_scores_bloom(y, y_bloom_predict):
    """return a dataframe of the scoring of the model
    """
    res         = pd.DataFrame(np.zeros((5,1)))
    res.columns = ['Bloomberg Model']
    res.index   = ['Accuracy','Average Recall', 'Average Precision', 'Average F1 Score','Mean Absolute Error']


    res.iloc[0,0] = '{:.2%}'.format(sk_metrics.accuracy_score(y_bloom_predict,y, normalize=True))
    res.iloc[1,0] = '{:.2%}'.format(sk_metrics.recall_score(y, y_bloom_predict, average='macro'))
    res.iloc[2,0] = '{:.2%}'.format(sk_metrics.precision_score(y, y_bloom_predict, average='macro'))
    res.iloc[3,0] = '{:.2%}'.format(sk_metrics.f1_score(y, y_bloom_predict, average='macro'))
    res.iloc[4,0] = sk_metrics.mean_absolute_error(y, y_bloom_predict)
    return res

    

def rating_rule(list_rating_code1, list_rating_code2, list_rating_code3):
    
    consensus = [] 
    for i in range(len(list_rating_code1)):
        
        rating_code1 =list_rating_code1[i]
        rating_code2 = list_rating_code2[i]
        rating_code3 = list_rating_code3[i]
        
        list_tmp = []
        
        if(not math.isnan(rating_code1)):
            list_tmp.append(rating_code1)
            
        if(not math.isnan(rating_code2)):
            list_tmp.append(rating_code2)
        
        if(not math.isnan(rating_code3)):
            list_tmp.append(rating_code3)
        
        list_tmp = sorted(list_tmp)
        
        if(len(list_tmp)==0):
            consensus.append(np.nan)
        elif(len(list_tmp)==1):
            consensus.append(list_tmp[0])
        elif(len(list_tmp)==2):
            consensus.append(list_tmp[0])
        else:
            consensus.append(list_tmp[0])
            
    return consensus
    

    
    

def encode_rating(df,feature_rating,agence_notation,table_rating_sp,table_rating_moodys,table_rating_fitch,table_rating_bloomberg):
    rating_list = list(df[feature_rating])
    rating_code = []
    for rating in rating_list:
        rating_code.append(f_encode_rating(rating,agence_notation,table_rating_sp,table_rating_moodys,table_rating_fitch,table_rating_bloomberg))
    return rating_code
    

def rating_consensus(list_rating1, list_rating2, list_rating3):
    consensus = []
    
    for i in range(len(list_rating1)):
        consensus.append(rating_rule(list_rating1[i],list_rating2[i],list_rating3[i]))
    return consensus
    

def encode_rating_bloom(df,feature_rating,table_rating_bloomberg):
    defaul_prob_list = list(np.abs(df[feature_rating]))
    rating_code = []
    for default_probability in defaul_prob_list:
        rating_code.append(default_probability_to_rating(default_probability, table_rating_bloomberg))
    return rating_code
    

def get_quality_scores_details(model,model_name, X_test,y_test):
    """return a dataframe of the scoring of the model
    """
    res         = pd.DataFrame(np.zeros((9,1)))
    res.columns = [model_name]
    res.index   = ['Recall Downgrade', 'Recall Stable', 'Recall Upgrade', 'Precision Downgrade','Precision Stable','Precision Upgrade', 'F1 Downgrade', 'F1 Stable', 'F1 Upgrade']
    #train
    X_test_predict = np.clip(model.predict(X_test),-1,1)
    
    res.iloc[0,0] = '{:.2%}'.format(sk_metrics.recall_score(y_test, X_test_predict, average=None)[0])
    res.iloc[1,0] = '{:.2%}'.format(sk_metrics.recall_score(y_test, X_test_predict, average=None)[1])
    res.iloc[2,0] = '{:.2%}'.format(sk_metrics.recall_score(y_test, X_test_predict, average=None)[2])
    res.iloc[3,0] = '{:.2%}'.format(sk_metrics.precision_score(y_test, X_test_predict, average=None)[0])
    res.iloc[4,0] = '{:.2%}'.format(sk_metrics.precision_score(y_test, X_test_predict, average=None)[1])
    res.iloc[5,0] = '{:.2%}'.format(sk_metrics.precision_score(y_test, X_test_predict, average=None)[2])
    res.iloc[6,0] = '{:.2%}'.format(sk_metrics.f1_score(y_test, X_test_predict, average=None)[0])
    res.iloc[7,0] = '{:.2%}'.format(sk_metrics.f1_score(y_test, X_test_predict, average=None)[1])
    res.iloc[8,0] = '{:.2%}'.format(sk_metrics.f1_score(y_test, X_test_predict, average=None)[2])

    return res

def new_f1_scorer(y_true, y_predict):
    recall = sk_metrics.recall_score(y_true, y_predict,average='macro')
    precision = sk_metrics.precision_score(y_true, y_predict, average='macro')
    return 2 * (precision * recall) / (precision + recall)

