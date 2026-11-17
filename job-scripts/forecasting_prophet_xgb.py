from .global_import import *
from .utils.all_utils import *
from .utils.prophet import *
from .utils.xgboost import *
from .utils.logging import *
logger=logging.getLogger()

MEASURE_COLUMNS=measure_dict
def forecast_and_train_all_prophet_xgb(model_type,type_data,version_control,hierarchy_type):
    logger.info(f"Starting read financial variable and historical data of {type_data}, hierarchy_type: {hierarchy_type}, and version : {version_control}")
    cdsw_data_dummy,df_ext_var_final=read_data_from_pa(type_data,version_control,target_month_duration_forecasting_date=15)
    logger.info(f'''Finished read financial variable and historical data of {type_data}. Total Rows of historical {type_data}: {len(cdsw_data_dummy)}
                Total Rows of financial variable data: {len(df_ext_var_final)}''')

    merged_df=prepare_data(cdsw_data_dummy,df_ext_var_final)
    logger.info(f"Finished Merged and Transformed Data into  1 dataframe. Total Rows: {len(merged_df)}")

    if(hierarchy_type!='All RO'):
        logger.info(f"Filtering data into hierarchy choosed: {hierarchy_type}.")
        merged_df=merged_df.loc[merged_df['kanwil']==hierarchy_type]
        logger.info(f"Finished data into hierarchy choosed: {hierarchy_type}. Total rows {len(hierarchy_type)}")

    all_data_prophet,all_models_data=train_prophet(merged_df,type_data)
    for model_key, model_data in all_models_data.items():
        dump_model(model_data,f'results/{type_data}/prophet_xgb/dump_model/prophet/{model_key}.pkl')

    #xgboost process
    all_xgb_results,all_models_data,all_feature_importances,all_corr_df=train_xgboost_using_time_series_results(all_data_prophet,merged_df,MEASURE_COLUMNS[type_data]["ro"],MEASURE_COLUMNS[type_data]["nas"],type_data)
    for model_key, model_data in all_models_data.items():
        dump_model(model_data,f'results/{type_data}/prophet_xgb/dump_model/xgb/{model_key}.pkl')

    #tm1 dump
    all_tm1_store_results,all_tm1_store_feature_importances,all_tm1_store_eval_results,completed_data=prepare_tm1_results(all_xgb_results,all_feature_importances,merged_df)
    all_tm1_store_results.to_excel(f'results/{type_data}/tm1_dump_local/all_final_results_{model_type}_tm1.xlsx')
    all_tm1_store_feature_importances.reset_index(drop=True).to_excel(f'results/{type_data}/tm1_dump_local/all_feature_importances_{model_type}_tm1.xlsx')
    all_tm1_store_eval_results.reset_index(drop=True).to_excel(f'results/{type_data}/tm1_dump_local/all_evaluation_results_{model_type}_tm1.xlsx')

    #Local dump
    completed_data.to_excel(f'results/{type_data}/{model_type}/final_results_{model_type}/final_results_{model_type}.xlsx')

    #Local dump
    for model_key, df_corr in all_corr_df.items():
        df_corr.reset_index(drop=True).to_excel(f'results/{type_data}/prophet_xgb/correlation_analysis/{model_key}.xlsx')

    #Plotting Result and save
    combinations_plot = all_xgb_results[['Kanwil', 'Segmen', 'Product']].drop_duplicates()
    combinations_plot = list(combinations_plot.itertuples(index=False, name=None))
    for comb in combinations_plot:
        df_plot=completed_data.loc[(completed_data['Kanwil']==comb[0])&(completed_data['Segmen']==comb[1])&(completed_data['Product']==comb[2])]
        plot_and_save(df_plot,'prophet_xgb',type_data,comb[0],comb[1],comb[2])