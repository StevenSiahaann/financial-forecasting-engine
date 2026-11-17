from ..global_import import *
from .all_utils import *
lists_log = [('BANDAR LAMPUNG', 'RITEL', 'GIRO'),
('JAYAPURA', 'MIKRO', 'GIRO'),
('MEDAN', 'MIKRO', 'GIRO'),
('JAYAPURA', 'RITEL', 'GIRO'),
('PALEMBANG', 'MIKRO', 'GIRO'),
('PADANG', 'RITEL', 'GIRO'),
('BANDUNG', 'MIKRO', 'GIRO'),
('MAKASSAR', 'MIKRO', 'DEPOSITO'),
('MANADO', 'MIKRO', 'DEPOSITO'),
('BANJARMASIN', 'RITEL', 'DEPOSITO'),
('MANADO', 'RITEL', 'DEPOSITO'),
('JAYAPURA', 'MIKRO', 'DEPOSITO'),
('JAYAPURA', 'RITEL', 'DEPOSITO'),
('PALEMBANG', 'MIKRO', 'DEPOSITO'),
('MALANG', 'MIKRO', 'GIRO'),
('BANJARMASIN', 'MIKRO', 'DEPOSITO'),
('BANDUNG', 'MIKRO', 'DEPOSITO'),
('MANADO', 'MIKRO', 'GIRO'),
('MAKASSAR', 'RITEL', 'DEPOSITO')]

def train_sarima(data,type_data):
    all_models_data = {}
    all_data_sarima=pd.DataFrame()

    df_all_amount=data[['periode','kanwil','segmen','produk','amount']]
    combinations = df_all_amount[['kanwil', 'segmen', 'produk']].drop_duplicates()
    combinations = list(combinations.itertuples(index=False, name=None))

    for comb in combinations:
        df_0_0_0=df_all_amount.loc[(df_all_amount['kanwil']==comb[0])&(df_all_amount['produk']==comb[2])&(df_all_amount['segmen']==comb[1])]
        df_0_0_0.sort_values(by=['periode'],inplace=True)
        df_0_0_0['amount']=df_0_0_0['amount']/1e9

        if(comb in lists_log):
            df_0_0_0['amount']=iqr(df_0_0_0['amount'])


        if(comb[0]=='JAKARTA 1' and comb[1]=='MIKRO' and type_data=='simpanan'):
            df_0_0_0_train_tmp=df_0_0_0.loc[(df_0_0_0['periode']>='2022-01-01')&(df_0_0_0['periode']<='2024-03-01')]
            df_0_0_0_test_tmp=df_0_0_0.loc[(df_0_0_0['periode']>'2024-03-01')&(df_0_0_0['periode']<='2024-06-01')]
            df_0_0_0_forecast_tmp=df_0_0_0.loc[(df_0_0_0['periode']>'2024-06-01')]
        elif(comb[2]=='KUPEDES'and type_data=='pinjaman'):
            df_0_0_0_train_tmp=df_0_0_0.loc[df_0_0_0['periode']<='2024-03-01']
            df_0_0_0_test_tmp=df_0_0_0.loc[(df_0_0_0['periode']>'2024-03-01')&(df_0_0_0['periode']<='2024-06-01')]
            df_0_0_0_forecast_tmp=df_0_0_0.loc[(df_0_0_0['periode']>'2024-06-01')]
        elif(comb[2]=='KUR'and type_data=='pinjaman'):
            df_0_0_0_train_tmp=df_0_0_0.loc[df_0_0_0['periode']<='2024-03-01']
            df_0_0_0_test_tmp=df_0_0_0.loc[(df_0_0_0['periode']>'2024-03-01')&(df_0_0_0['periode']<='2024-06-01')]
            df_0_0_0_forecast_tmp=df_0_0_0.loc[(df_0_0_0['periode']>'2024-06-01')]
        else:
            df_0_0_0_train_tmp=df_0_0_0.loc[df_0_0_0['periode']<='2024-03-01']
            df_0_0_0_test_tmp=df_0_0_0.loc[(df_0_0_0['periode']>'2024-03-01')&(df_0_0_0['periode']<='2024-06-01')]
            df_0_0_0_forecast_tmp=df_0_0_0.loc[(df_0_0_0['periode']>'2024-06-01')]




        #model config
        SARIMA_model = pmd.auto_arima((df_0_0_0_train_tmp[['amount']]),
                        start_p=1, start_q=1,
                        test='adf',
                        max_p=3, max_q=3, m=12,
                        start_P=0, seasonal=True,
                        d=None, D=1, 
                        trace=False,
                        error_action='ignore',  
                        suppress_warnings=True, 
                        stepwise=True)
        
        df_0_0_0_train_tmp.set_index('periode', inplace=True)
        df_0_0_0_test_tmp.set_index('periode', inplace=True)
        df_0_0_0_forecast_tmp.set_index('periode', inplace=True)

        df_0_0_0_train_tmp.index = pd.to_datetime(df_0_0_0_train_tmp.index)
        df_0_0_0_test_tmp.index = pd.to_datetime(df_0_0_0_test_tmp.index)
        df_0_0_0_forecast_tmp.index = pd.to_datetime(df_0_0_0_forecast_tmp.index)


        sarimax=SARIMAX((df_0_0_0_train_tmp['amount']),order=SARIMA_model.get_params()['order'],seasonal_order=SARIMA_model.get_params()['seasonal_order'])
        model=sarimax.fit()

        # Make predictions on train data
        df_0_0_0_train_tmp['Amount_Predict'] = (model.predict(start = df_0_0_0_train_tmp.index.min(), end = df_0_0_0_train_tmp.index.max()))
        # Make predictions on test data
        df_0_0_0_test_tmp['Amount_Predict'] = (model.predict(start = df_0_0_0_test_tmp.index.min(), end = df_0_0_0_test_tmp.index.max()))
        df_0_0_0_forecast_tmp['Amount_Predict'] = (model.predict(start = df_0_0_0_forecast_tmp.index.min(), end = df_0_0_0_forecast_tmp.index.max()))

        df_0_0_0_train_tmp=df_0_0_0_train_tmp.reset_index()
        df_0_0_0_test_tmp=df_0_0_0_test_tmp.reset_index()
        df_0_0_0_forecast_tmp=df_0_0_0_forecast_tmp.reset_index()

        df_0_0_0_train_tmp['Partition']='TRAIN'
        df_0_0_0_test_tmp['Partition']='TEST'
        df_0_0_0_forecast_tmp['Partition']='FORECAST'

        forecast_train=df_0_0_0_train_tmp[['periode','amount','Amount_Predict','Partition']]
        forecast_test=df_0_0_0_test_tmp[['periode','amount','Amount_Predict','Partition']]
        forecast_forecast=df_0_0_0_forecast_tmp[['periode','amount','Amount_Predict','Partition']]
        combined_forecast = pd.concat([forecast_train, forecast_test,forecast_forecast])

        combined_forecast['Method']='SARIMA'
        combined_forecast['Kanwil']=comb[0]
        combined_forecast['Segmen']=comb[1]
        combined_forecast['Product']=comb[2]
        combined_forecast.rename(columns={'periode':'ds','amount':'Amount'}, inplace=True)

        model_data = {
            'model': model,
            'scalers_numerical_train': None,
            'scalers_categorical_train':None,
            'scalers_log_col':None,
            'coefficients_df_sorted':None
        }
        model_key = f'{comb[0]}_{comb[1]}_{comb[2]}'
        all_models_data[model_key] = model_data
        all_data_sarima=pd.concat([all_data_sarima,combined_forecast])
        break
    return all_data_sarima,all_models_data