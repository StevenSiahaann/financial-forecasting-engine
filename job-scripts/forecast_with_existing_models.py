from .global_import import *
from .utils.all_utils import *
from .utils.prophet import *
from .utils.xgboost import *
from .utils.logging import *
logger=logging.getLogger()
MEASURE_COLUMNS=measure_dict
def load_all_combinations_model(combinations,model_type,type_data):
    logger.info(f"Loading All Combinations Models Trained using {model_type} on {type_data} data.")
    all_models_data_time_series={}
    all_models_data_xgboost={}
    for comb in combinations:
        logger.info(f"Processing load {model_type.split('_')[0]} (Time series models) for {comb[0]}-{comb[1]}-{comb[2]}.")
        model_time_series,scalers_num_time_series,scalers_cat_time_series,list_scaled_log_col_time_series,coefficients_df_sort_time_series=load_model(f'results/{type_data}/{model_type}/dump_model/{model_type.split('_')[0]}/{comb[0]}_{comb[1]}_{comb[2]}.pkl')
        model_data_time_series = {
            'model': model_time_series,
            'scalers_numerical_train': scalers_num_time_series,
            'scalers_categorical_train':scalers_cat_time_series,
            'scalers_log_col':list_scaled_log_col_time_series,
            'coefficients_df_sorted':coefficients_df_sort_time_series
        }
        all_models_data_time_series[f'{comb[0]}_{comb[1]}_{comb[2]}'] = model_data_time_series
        logger.info(f"Finished load {model_type.split('_')[0]} (Time series models) for {comb[0]}-{comb[1]}-{comb[2]}.")


        model_xgboost,scalers_num_xgboost,scalers_cat_xgboost,list_scaled_log_col_xgboost,coefficients_df_sort_xgboost=load_model(f'results/{type_data}/{model_type}/dump_model/{model_type.split('_')[1]}/{comb[0]}_{comb[1]}_{comb[2]}.pkl')
        logger.info(f"Processing load {model_type.split('_')[1]} (XGBoost models) for {comb[0]}-{comb[1]}-{comb[2]}.")

        model_data_xgboost = {
            'model': model_xgboost,
            'scalers_numerical_train': scalers_num_xgboost,
            'scalers_categorical_train':scalers_cat_xgboost,
            'scalers_log_col':list_scaled_log_col_xgboost,
            'coefficients_df_sorted':coefficients_df_sort_xgboost
        }
        logger.info(f"Processing load {model_type.split('_')[1]} (XGBoost models) for {comb[0]}-{comb[1]}-{comb[2]}.")
        all_models_data_xgboost[f'{comb[0]}_{comb[1]}_{comb[2]}'] = model_data_xgboost
    logger.info(f"Finished Loading of {len(all_models_data_time_series)} {model_type.split('_')[0]} models and {len(all_models_data_xgboost)} {model_type.split('_')[1]} on {type_data} data.")
    return all_models_data_time_series,all_models_data_xgboost
def predict_time_series(df,model_type,all_model_data_time_series):
    combinations = df[['kanwil', 'segmen', 'produk']].drop_duplicates()
    combinations = list(combinations.itertuples(index=False, name=None))
    all_results_time_series=pd.DataFrame()
    for comb in combinations:
        logger.warning(f"Processing predictions using {model_type.split('_')[0]} (Time series models) for {comb[0]}-{comb[1]}-{comb[2]}.")
        model_data_time_series = all_model_data_time_series.get(f'{comb[0]}_{comb[1]}_{comb[2]}')
        df_to_predict=df.loc[(df['kanwil']==comb[0])&(df['produk']==comb[2])&(df['segmen']==comb[1])]
        df_to_predict=df_to_predict.rename(columns={'amount':'y','periode':'ds'})
        if(model_type=='prophet_xgb'):
            df_to_predict['y']=df_to_predict['y']/1e9
            df_to_predict['corona'] = df_to_predict['ds'].apply(is_corona_period)
            if(comb[0]!= 'JAKARTA 1' and  comb[1]!='MIKRO' and comb[2]=='KUPEDES'):
                df_to_predict['outliers'] = df_to_predict['periode'].apply(special_case_kupedes)

            outliers_amount, lower_amount, upper_amount,mean_amount = detect_outliers_iqr(df_to_predict, 'y')

            pred=model_data_time_series['model'].predict(df_to_predict.drop(columns=['kanwil','y','segmen','produk']))
            pred['yhat'] = pred['yhat'].apply(lambda x: lower_amount if x < 0 else x)

            df_to_predict.reset_index(drop=True, inplace=True)
            pred.reset_index(drop=True, inplace=True)

            pred['y']=df_to_predict['y']
            pred.rename(columns={'y':'Amount','yhat':'Amount_Predict'}, inplace=True)
            pred['Kanwil']=comb[0]
            pred['Segmen']=comb[1]
            pred['Product']=comb[2]
            pred['Method']='PROPHET'
            all_results_time_series=pd.concat([all_results_time_series,pred[['ds','Kanwil','Segmen','Product','Method','Amount','Amount_Predict']]])
        # TODO : Put sarima_xgb code here
        elif(model_type=='sarima_xgb'):
            break
        logger.warning(f"Finished predictions process using {model_type.split('_')[0]} (Time series models) for {comb[0]}-{comb[1]}-{comb[2]}.")
    logger.warning(f"Finished all predictions process using {model_type.split('_')[0]} (Time series models)")
    return all_results_time_series
def predict_xgboost(all_models_data_xgboost,all_results_time_series,merged_df):
    merged_df_xgb = merged_df.rename(columns={'kanwil': 'Kanwil','segmen':'Segmen','produk':'Product','periode':'ds'})
    merged_df_xgb = pd.merge(all_results_time_series, merged_df_xgb, how='inner', on=['ds', 'Kanwil', 'Segmen', 'Product'])
    merged_df_xgb=merged_df_xgb.drop(columns=['amount'])    
    combinations = all_results_time_series[['Kanwil', 'Segmen', 'Product']].drop_duplicates()
    combinations = list(combinations.itertuples(index=False, name=None))
    all_xgb_results=pd.DataFrame()
    for comb in combinations:
        model_data_xgboost = all_models_data_xgboost.get(f'{comb[0]}_{comb[1]}_{comb[2]}')
        merged_df_xgb_test=merged_df_xgb.loc[(merged_df_xgb['Kanwil']==comb[0])&(merged_df_xgb['Product']==comb[2])&(merged_df_xgb['Segmen']==comb[1])]

        X_forecast = merged_df_xgb_test.drop(columns=['Kanwil','Amount','Amount_Predict','Segmen','Product','Method'])
        X_forecast['year'] = X_forecast['ds'].dt.year
        X_forecast['month'] = X_forecast['ds'].dt.month
        X_forecast = X_forecast.drop(columns=['ds'])


        X_forecast_scaled=process_data_numeric_with_existing_scalers_xgb(X_forecast,model_data_xgboost['scalers_numerical_train'],model_data_xgboost['scalers_log_col'])

        
        X_forecast_scaled_extract=X_forecast_scaled[model_data_xgboost['model'].get_booster().feature_names].reset_index(drop=True)

        y_pred = model_data_xgboost['model'].predict(X_forecast_scaled_extract)
        predictions_non_scaled = model_data_xgboost['scalers_numerical_train']['residu'].inverse_transform(y_pred.reshape(-1, 1)).flatten()
        merged_df_xgb_test['predictions_xgb']=predictions_non_scaled
        merged_df_xgb_test['Final_Predictions']=merged_df_xgb_test['Amount_Predict']+merged_df_xgb_test['predictions_xgb']
        all_xgb_results=pd.concat([all_xgb_results,merged_df_xgb_test])
        if(merged_df_xgb['Method'].unique()[0]=='PROPHET'):
            all_xgb_results['Method']='PROPHET_XGB'
        elif(merged_df_xgb['Method'].unique()[0]=='SARIMA'):
            all_xgb_results['Method']='SARIMA_XGB'
    return all_xgb_results
def forecast_with_existing_model(model_type,type_data,version_control,hierarchy_type):
    logger.info(f"Starting read financial variable and historical data of {type_data}, hierarchy_type: {hierarchy_type}, and version : {version_control}")
    cdsw_data_dummy,df_ext_var_final=read_data_from_pa(type_data,version_control,target_month_duration_forecasting_date=15)
    logger.info(f'''Finished read financial variable and historical data of {type_data}. Total Rows of historical {type_data}: {len(cdsw_data_dummy)}
                Total Rows of financial variable data: {len(df_ext_var_final)}''')
    logger.info(f"Starting Merging and Transformed Data into  1 dataframe.")
    merged_df=prepare_data(cdsw_data_dummy,df_ext_var_final)
    logger.info(f"Finished Merged and Transformed Data into  1 dataframe. Total Rows: {len(merged_df)}")

    if(hierarchy_type!='All RO'):
        logger.info(f"Filtering data into hierarchy choosed: {hierarchy_type}.")
        merged_df=merged_df.loc[merged_df['kanwil']==hierarchy_type]
        logger.info(f"Finished data into hierarchy choosed: {hierarchy_type}. Total rows {len(hierarchy_type)}")


    combinations = merged_df[['kanwil', 'segmen', 'produk']].drop_duplicates()
    combinations = list(combinations.itertuples(index=False, name=None))
    all_models_data_time_series,all_models_data_xgboost=load_all_combinations_model(combinations,model_type,type_data)
    all_results_time_series=predict_time_series(merged_df,model_type,all_models_data_time_series)
    all_xgb_results =predict_xgboost(all_models_data_xgboost,all_results_time_series,merged_df)
    return all_xgb_results
