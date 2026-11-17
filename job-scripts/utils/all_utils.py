from ..global_import import *
from .all_utils import *

def plot_and_save(df,model_type,type,kanwil,segmen,produk):
    df_sorted = df.sort_values(by='ds')
    plt.figure(figsize=(10, 6))
    plt.plot(df_sorted['ds'], df_sorted['Amount'], label='Actual', marker='o')
    plt.plot(df_sorted['ds'], df_sorted['Final_Predictions'], label='Predicted', marker='x')
    plt.xlabel('Date')
    plt.ylabel('Values')
    plt.title(f'Actual vs Predicted for {kanwil} - {segmen} - {produk}')
    plt.legend()    
    plt.tight_layout()
    plot_filename = f"visualization/{type}/{model_type}/{kanwil}_{segmen}_{produk}.png"
    plt.savefig(plot_filename)
    print(f"Plot saved as {plot_filename}")
def load_env_file(filepath):
    with open(filepath) as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                key, value = line.strip().split('=', 1)
                os.environ[key] = value
def load_config_json(file_path_json):
    with open(file_path_json, 'r') as f:
        config = json.load(f)
    return config
def special_case_kupedes(ds):
    date = pd.to_datetime(ds)
    return (date.year == 2023)
def special_case_kur(ds):
    date = pd.to_datetime(ds)
    return (date.year == 2023)
def detect_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    Q2 = df[column].quantile(0.5)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    lower_bound = max(lower_bound, 0)
    mean=Q2
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers, lower_bound, upper_bound,mean
def replace_outliers_with_bound(df, column, lower_bound, upper_bound):
    df[column] = df[column].apply(lambda x: upper_bound if x > upper_bound else lower_bound if x < lower_bound else x)
    return df
def remove_outliers(df, column, lower_bound, upper_bound):
    df_filtered = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df_filtered
def is_precorona_period(ds):
    date = pd.to_datetime(ds)
    return (date.year < 2020)
def is_postcorona_period(ds):
    date = pd.to_datetime(ds)
    return (date.year > 2021)
def is_corona_period(ds):
    date = pd.to_datetime(ds)
    return (date.year == 2021 and date.year==2020)

def dump_model(model_data,file_path):
    with open (file_path,'wb') as f:
        pickle.dump(model_data,f)

def load_model(file_path):
    with open(file_path, 'rb') as f:
        loaded_data = pickle.load(f)
    model = loaded_data['model']
    scalers_num = loaded_data['scalers_numerical_train']
    scalers_cat = loaded_data['scalers_categorical_train']
    list_scaled_log_col=loaded_data['scalers_log_col']
    coefficients_df_sorted=loaded_data['coefficients_df_sorted']
    return model, scalers_num,scalers_cat,list_scaled_log_col,coefficients_df_sorted
def metrics(predicted, actual):
    RMSE  = round(np.sqrt(mean_squared_error(predicted, actual)),3)
    mae = mean_absolute_error(predicted, actual)
    mape = mean_absolute_percentage_error(predicted, actual)
    r2 = r2_score(predicted, actual)
    return {'RMSE':RMSE, 'MAE':mae,'MAPE':mape, 'R2':r2}

def Evaluate_Model(actual, predicted,type):
    forecast = actual[['ds','y']]
    forecast = forecast.set_index('ds')
    forecast.rename(columns={'y':'Actual'}, inplace=True)
    forecast['Predicted'] = predicted['yhat'].values
    if(type=='forecast'):
        return forecast
    elif(type=='test'):
        forecast_tmp=forecast.loc[(forecast.index>'2024-03-01')&(forecast.index<='2024-06-01')]
    elif(type=='train'):
        forecast_tmp=forecast.loc[forecast.index<='2024-03-01']
    KPI = metrics(forecast_tmp['Predicted'], forecast_tmp['Actual'])
    return forecast
def modify_measure(row):
    if (row['kanwil']=='NASIONAL'):
        return f"{row['Measure']}_nasional"
    return row['Measure']
def get_all_version_of_ext_var():
    return
def read_data_from_pa(type,version_control,valid_date_start_data='2019-01-01',target_month_duration_forecasting_date=15):
    # real case cdsw uncomment this
    """
    cdsw_data=pd.read_csv(f"{base_dir_cdsw}/{type}.csv",sep='|', index_col=False)
    df_ext_var = pd.DataFrame()
    try:
        tm1 = TM1Service(address=address, port=port, user=user, password=password, ssl=ssl)
        df_ext_var = tm1.cubes.cells.execute_view_dataframe(cube_name=cube_name, view_name=view_name) 
    except Exception as e:
        print(f"Error connecting to TM1: {e}")
        return df_ext_var 
    """
    #local testing
    if(type=='simpanan'):
        cdsw_data=pd.read_excel(f"data/{type}/{type} forecast (until des 2025)-v1.xlsx", index_col=False)
    elif(type=='pinjaman'):
        cdsw_data=pd.read_excel(f"data/{type}/{type} forecast (until des 2025).xlsx", index_col=False)
    cdsw_data_dummy=cdsw_data[['periode','kanwil','segmen','produk','amount']]

    df_ext_var=pd.read_csv(f"data/dummy_Financial Variable.csv",sep=',', index_col=False).reset_index(drop=True)
    df_ext_var=df_ext_var.loc[df_ext_var['Version']==version_control]

    #Handling different values or columns names
    df_ext_var['kanwil'] = df_ext_var['kanwil'].replace('KORPORASI', 'NASIONAL')
    df_ext_var['Measure'] = df_ext_var.apply(
        lambda row: pd.Series(modify_measure(row)), axis=1
    )
    # Make sure same format ds
    cdsw_data_dummy['periode'] = pd.to_datetime(cdsw_data_dummy['periode'])
    df_ext_var['periode'] = pd.to_datetime(df_ext_var['periode'])

    #handle jakarta 1 mikro move to jakarta 2 mikro from 2019 until 2021
    start_date = '2019-01-01'
    end_date = '2021-12-01'        
    jakarta1 = cdsw_data_dummy[(cdsw_data_dummy['kanwil'] == 'JAKARTA 1') & 
                            (cdsw_data_dummy['segmen'] == 'MIKRO') & 
                            (cdsw_data_dummy['periode'].between(start_date, end_date))]
    
    jakarta1_agg = jakarta1.groupby(['periode', 'produk'], as_index=False)['amount'].sum()        
    cdsw_data_dummy = cdsw_data_dummy.merge(jakarta1_agg, on=['periode', 'produk'], how='left', suffixes=('', '_jakarta1'))        
    cdsw_data_dummy.loc[(cdsw_data_dummy['kanwil'] == 'JAKARTA 2') & 
                        (cdsw_data_dummy['segmen'] == 'MIKRO'), 
                        'amount'] += cdsw_data_dummy['amount_jakarta1'].fillna(0)        
    cdsw_data_dummy.loc[(cdsw_data_dummy['kanwil'] == 'JAKARTA 1') & 
                        (cdsw_data_dummy['segmen'] == 'MIKRO') & 
                        (cdsw_data_dummy['periode'].between(start_date, end_date)), 
                        'amount'] = 0        
    cdsw_data_dummy.drop(columns=['amount_jakarta1'], inplace=True)

    valid_date_start_data = pd.to_datetime(valid_date_start_data)
    df_pivot = df_ext_var.reset_index(drop=True).pivot_table(index=['periode','kanwil'], 
                            columns='Measure',
                            values='Value')
    df_pivot=df_pivot.reset_index()
    cdsw_data_filtered = cdsw_data_dummy[cdsw_data_dummy['periode'] >= valid_date_start_data]
    df_ext_var_filtered = df_pivot[df_pivot['periode'] >= valid_date_start_data]
    cdsw_data_final = cdsw_data_filtered[cdsw_data_filtered['periode'] < datetime.now() + pd.DateOffset(months=target_month_duration_forecasting_date)]
    df_ext_var_final = df_ext_var_filtered[df_ext_var_filtered['periode'] < datetime.now() + pd.DateOffset(months=target_month_duration_forecasting_date)]

    return cdsw_data_final,df_ext_var_final
def prepare_data(actual_data,ext_var):
    nasional_df = ext_var[ext_var['kanwil'] == 'NASIONAL'].set_index('periode')
    for idx, row in ext_var[ext_var['kanwil'] != 'NASIONAL'].iterrows():
        periode = row['periode']
        for col in ext_var.filter(regex='.*nasional').columns:
            if pd.isna(row[col]):
                if periode in nasional_df.index:
                    ext_var.loc[idx, col] = nasional_df.loc[periode, col]
    df_ext_var_final_non_kck_banda = ext_var.loc[(ext_var['kanwil'] != 'K C K') & (ext_var['kanwil'] != 'BANDA ACEH')]
    merged_df=actual_data.merge(df_ext_var_final_non_kck_banda,on=['periode','kanwil'],how='inner')
    return merged_df
def iqr(x):
    IQR = np.diff(x.quantile([0.25,0.75]))[0]
    Q1 = x.quantile(0.25)
    Q3 = x.quantile(0.75)
    S = 1.5*IQR
    if IQR==0: 
        return x
    else:   
        x[x < Q1 - S] = Q1 - S
        x[x > Q3 + S] = Q1 + S
        return x
def rounding_result(x):
    x = np.where(x<0, 0., [e for e in x ])
    return x
def calculate_metrics(y_true, y_pred):
    # Menghitung metrik
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100  # dikalikan 100 untuk hasil dalam persen
    r2 = r2_score(y_true, y_pred)
    
    return rmse, mae, mape, r2

def prepare_evaluation_results(df_results):
    eval_results = []    
    df_filtered = df_results[df_results['Partition'].isin(['TRAIN', 'TEST'])]

    for (kanwil, segmen, produk, partition), group in df_filtered.groupby(['Kanwil', 'Segmen', 'Product', 'Partition']):
        y_true = group['Amount']
        y_pred = group['Final_Predictions']
        method = group['Method'].iloc[0]
        
        rmse, mae, mape, r2 = calculate_metrics(y_true, y_pred)
        eval_results.append({'kanwil': kanwil, 'segmen': segmen, 'produk': produk, 'Version': method, 'Partition': partition, 'Measure': 'RMSE', 'Value': rmse})
        eval_results.append({'kanwil': kanwil, 'segmen': segmen, 'produk': produk, 'Version': method, 'Partition': partition, 'Measure': 'MAE', 'Value': mae})
        eval_results.append({'kanwil': kanwil, 'segmen': segmen, 'produk': produk, 'Version': method, 'Partition': partition, 'Measure': 'MAPE', 'Value': mape})
        eval_results.append({'kanwil': kanwil, 'segmen': segmen, 'produk': produk, 'Version': method, 'Partition': partition, 'Measure': 'R²', 'Value': r2})
    
    df_eval_results = pd.DataFrame(eval_results)
    return df_eval_results
def prepare_tm1_results(df_results,df_feature_importances,df_all):
    df_all = df_all.rename(columns={'kanwil': 'Kanwil','segmen':'Segmen','produk':'Product','periode':'ds'})
    df_all_filtered = df_all[df_all[['Kanwil', 'Segmen', 'Product']].apply(tuple, axis=1).isin(df_results[['Kanwil', 'Segmen', 'Product']].apply(tuple, axis=1))]
    merged_df = pd.merge(df_results, df_all_filtered, how='outer', on=['ds', 'Kanwil', 'Segmen', 'Product'], indicator=True)
    merged_df = merged_df[merged_df.columns[~merged_df.columns.str.endswith('_x')]]
    merged_df.columns = merged_df.columns.str.replace('_y$', '', regex=True)
    merged_df.drop(columns=['Amount'],inplace=True)
    merged_df = merged_df.rename(columns={'amount': 'Amount'})
    merged_df['Amount']=merged_df['Amount']/1e9
    right_only = merged_df[merged_df['_merge'] == 'right_only']
    right_only.drop(columns=['_merge'],inplace=True)
    right_only['Method']='PROPHET_XGB'
    right_only['Amount_Predict']=0
    right_only['residu']=0
    right_only['predictions_xgb']=0
    right_only['Final_Predictions']=0
    completed_data=pd.concat([df_results,right_only])

    all_tm1_dump_results=pd.DataFrame()
    all_tm1_dump_feature_importances=pd.DataFrame()
    all_tm1_eval_results=prepare_evaluation_results(df_results)
    for idx, row in completed_data.iterrows():
        entries = [
            {'periode': pd.to_datetime(row['ds']).strftime('%b-%Y'), 'kanwil': row['Kanwil'], 
            'segmen': row['Segmen'], 'produk': row['Product'], 'Version': 'Actual', 
            'Value': row['Amount']},
            {'periode': pd.to_datetime(row['ds']).strftime('%b-%Y'), 'kanwil': row['Kanwil'], 
            'segmen': row['Segmen'], 'produk': row['Product'], 'Version': row['Method'], 
            'Value': row['Final_Predictions']}
        ]
        temp_df = pd.DataFrame(entries)
        all_tm1_dump_results = pd.concat([all_tm1_dump_results, temp_df], ignore_index=True)
    for model_key, df_feature_importances_percent in df_feature_importances.items():
        string_comb=model_key.split("_")
        for idx, row in df_feature_importances_percent.iterrows():
            temp_df = pd.DataFrame({
                'kanwil': [string_comb[0]],
                'segmen': [string_comb[1]],
                'produk': [string_comb[2]],
                'Version': ['PROPHET_XGB'],
                'Measure': [row['Feature']],
                'Value': [row['Importance (%)']]
            })
            all_tm1_dump_feature_importances = pd.concat([all_tm1_dump_feature_importances, temp_df], ignore_index=True)
    return all_tm1_dump_results,all_tm1_dump_feature_importances,all_tm1_eval_results,completed_data

