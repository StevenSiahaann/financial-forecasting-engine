from ..global_import import *
from .all_utils import *
def train_prophet(data,type_data):
    all_models_data = {}
    df_all_amount=data[['periode','kanwil','segmen','produk','amount']]
    combinations = df_all_amount[['kanwil', 'segmen', 'produk']].drop_duplicates()
    combinations = list(combinations.itertuples(index=False, name=None))
    all_data_prophet=pd.DataFrame()
    for comb in combinations:
        print(f"Processing {comb}")
        df_0_0_0=df_all_amount.loc[(df_all_amount['kanwil']==comb[0])&(df_all_amount['produk']==comb[2])&(df_all_amount['segmen']==comb[1])]
        # df_0_0_0['pre_corona'] = df_0_0_0['periode'].apply(is_precorona_period)
        # df_0_0_0['post_corona'] = df_0_0_0['periode'].apply(is_postcorona_period)
        df_0_0_0['corona'] = df_0_0_0['periode'].apply(is_corona_period)
        if(comb[0]!= 'JAKARTA 1' and  comb[1]!='MIKRO' and comb[2]=='KUPEDES'):
            df_0_0_0['outliers'] = df_0_0_0['periode'].apply(special_case_kupedes)
        # elif(comb[2]=='KUR'):
        #     df_0_0_0['outliers'] = df_0_0_0['periode'].apply(special_case_kur)

        if(comb[0]=='JAKARTA 1' and comb[1]=='MIKRO'):
            df_0_0_0_train_tmp=df_0_0_0.loc[(df_0_0_0['periode']>='2022-01-01')&(df_0_0_0['periode']<='2024-03-01')]
            df_0_0_0_test_tmp=df_0_0_0.loc[(df_0_0_0['periode']>'2024-03-01')&(df_0_0_0['periode']<='2024-06-01')]
            df_0_0_0_forecast_tmp=df_0_0_0.loc[(df_0_0_0['periode']>'2024-06-01')]
        #special case kupedes non jakarta 1 and mikro
        elif(comb[2]=='KUPEDES'and type_data=='pinjaman'):
            df_0_0_0_train_tmp = df_0_0_0.loc[(df_0_0_0['periode'] < '2023-01-01') | 
                                            ((df_0_0_0['periode'] >= '2024-01-01') & 
                                            (df_0_0_0['periode'] <= '2024-03-31'))]

            df_0_0_0_test_tmp=df_0_0_0.loc[((df_0_0_0['periode']>='2023-01-01')&(df_0_0_0['periode']<='2023-12-01'))|((df_0_0_0['periode']>='2024-04-01')&(df_0_0_0['periode']<='2024-06-01'))]
            df_0_0_0_forecast_tmp=df_0_0_0.loc[(df_0_0_0['periode']>'2024-06-01')]
        #special case kur non jakarta 1 and mikro
        elif(comb[2]=='KUR'and type_data=='pinjaman'):
            df_0_0_0_train_tmp = df_0_0_0.loc[(df_0_0_0['periode'] >= '2023-01-01')&
                                            (df_0_0_0['periode'] <= '2024-03-31')]

            df_0_0_0_test_tmp=df_0_0_0.loc[(df_0_0_0['periode']>='2024-04-01')&(df_0_0_0['periode']<='2024-06-01')]
            df_0_0_0_forecast_tmp=df_0_0_0.loc[(df_0_0_0['periode']>'2024-06-01')]
        else:
            df_0_0_0_train_tmp=df_0_0_0.loc[df_0_0_0['periode']<='2024-03-01']
            df_0_0_0_test_tmp=df_0_0_0.loc[(df_0_0_0['periode']>'2024-03-01')&(df_0_0_0['periode']<='2024-06-01')]
            df_0_0_0_forecast_tmp=df_0_0_0.loc[(df_0_0_0['periode']>'2024-06-01')]


        df_0_0_0_train_tmp.rename(columns={'amount':'y','periode':'ds'}, inplace=True)
        df_0_0_0_train_tmp['y']=df_0_0_0_train_tmp['y']/1e9

        df_0_0_0_test_tmp.rename(columns={'amount':'y','periode':'ds'}, inplace=True)
        df_0_0_0_test_tmp['y']=df_0_0_0_test_tmp['y']/1e9

        df_0_0_0_forecast_tmp.rename(columns={'amount':'y','periode':'ds'}, inplace=True)
        df_0_0_0_forecast_tmp['y']=df_0_0_0_forecast_tmp['y']/1e9

        df_0_0_0_train_tmp['y_original']=df_0_0_0_train_tmp['y']
        df_0_0_0_test_tmp['y_original']=df_0_0_0_test_tmp['y']

        outliers_amount, lower_amount, upper_amount,mean_amount = detect_outliers_iqr(df_0_0_0_train_tmp, 'y')

        if(comb[0]!= 'JAKARTA 1' and  comb[1]!='MIKRO' and comb[2] in ['KUPEDES'] and type_data=='pinjaman'):
            model = Prophet()
            model.add_seasonality(name='outliers',
                            period=30, 
                            prior_scale = 20,
                            fourier_order=5,
                            condition_name='outliers')

        else:
            df_0_0_0_train = replace_outliers_with_bound(df_0_0_0_train_tmp, 'y', lower_amount, upper_amount)
            df_0_0_0_test = replace_outliers_with_bound(df_0_0_0_test_tmp, 'y', lower_amount, upper_amount)
            model = Prophet()

        # model.add_seasonality(name='monthly-pre-corona', 
        #                 period=30, 
        #                 prior_scale = 10,
        #                 fourier_order=5, 
        #                 condition_name='pre_corona')

        # model.add_seasonality(name='monthly-post-corona',
        #                 period=30, 
        #                 prior_scale = 10,
        #                 fourier_order=5,
        #                 condition_name='post_corona')
        model.add_seasonality(name='monthly-corona',
                        period=30, 
                        prior_scale = 10,
                        fourier_order=5,
                        condition_name='corona')

        model.fit(df_0_0_0_train.drop(columns=['kanwil','segmen','produk','y_original']))
        pred=model.predict(df_0_0_0_train.drop(columns=['kanwil','y','segmen','produk','y_original']))
        pred_test=model.predict(df_0_0_0_test.drop(columns=['kanwil','y','segmen','produk','y_original']))
        pred_forecast=model.predict(df_0_0_0_forecast_tmp.drop(columns=['kanwil','y','segmen','produk']))

        pred['yhat'] = pred['yhat'].apply(lambda x: lower_amount if x < 0 else x)
        pred_test['yhat'] = pred_test['yhat'].apply(lambda x: lower_amount if x < 0 else x)
        pred_forecast['yhat'] = pred_forecast['yhat'].apply(lambda x: lower_amount if x < 0 else x)


        # assign back real value not the replaced outliers on actual amount
        pred['y']=df_0_0_0_train_tmp['y_original']
        pred_test['y']=df_0_0_0_test_tmp['y_original']
        df_0_0_0_train_tmp['y']=df_0_0_0_train_tmp['y_original']
        df_0_0_0_test_tmp['y']=df_0_0_0_test_tmp['y_original']

        forecast=Evaluate_Model(df_0_0_0_train, pred,'train')
        forecast_test=Evaluate_Model(df_0_0_0_test, pred_test,'test')
        forecast_forecast=Evaluate_Model(df_0_0_0_forecast_tmp, pred_forecast,'forecast')

        forecast['Partition'] = 'TRAIN'
        forecast_test['Partition'] = 'TEST'
        forecast_forecast['Partition']='FORECAST'

        combined_forecast = pd.concat([forecast, forecast_test,forecast_forecast])
        combined_forecast['Kanwil']=comb[0]
        combined_forecast['Segmen']=comb[1]
        combined_forecast['Product']=comb[2]
        combined_forecast['Method']='PROPHET'
        combined_forecast=combined_forecast.reset_index()
        combined_forecast.rename(columns={'Actual':'Amount','Predicted':'Amount_Predict'}, inplace=True)
        model_data = {
            'model': model,
            'scalers_numerical_train': None,
            'scalers_categorical_train':None,
            'scalers_log_col':None,
            'coefficients_df_sorted':None
        }
        model_key = f'{comb[0]}_{comb[1]}_{comb[2]}'
        all_models_data[model_key] = model_data
        all_data_prophet=pd.concat([all_data_prophet,combined_forecast])
    return all_data_prophet,all_models_data