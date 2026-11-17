from ..global_import import *
def calculate_feature_importance_in_percentage(feature_importances):
    total_importance = sum(feature_importances.values())
    feature_importances_percent = {feature: (importance / total_importance) * 100 for feature, importance in feature_importances.items()}
    df_feature_importances_percent = pd.DataFrame(
        list(feature_importances_percent.items()), 
        columns=['Feature', 'Importance (%)']
    )
    df_feature_importances_percent = df_feature_importances_percent.sort_values(by='Importance (%)', ascending=False)    
    return df_feature_importances_percent

def process_data_numeric_xgb(df,method):
    scalers={}
    list_of_col_log_scaled=[]
    if(method==None):
        return scalers,df,list_of_col_log_scaled
    for col in df.columns:
        if(method=='Standard'):
            scalers[col]=StandardScaler()
            df[col]=scalers[col].fit_transform(df[col].values.reshape(-1, 1))
        elif(method=='Robust'):
            scalers[col]=RobustScaler()
            df[col]=scalers[col].fit_transform(df[col].values.reshape(-1, 1))
        elif(method=='MinMax'):
            scalers[col]=MinMaxScaler()
            df[col]=scalers[col].fit_transform(df[col].values.reshape(-1, 1))
    return scalers,df,list_of_col_log_scaled
def process_data_numeric_with_existing_scalers_xgb(df,scalers,list_of_col_log_scaled):    
    if(len(scalers.keys())==0):
        return df
    for col in df.columns:
        df[col]=scalers[col].transform(df[col].values.reshape(-1, 1))
    return df
def feature_extraction_xgb(df,actual,ro_cols,nas_cols,level='ro'):
    df_dump=pd.DataFrame()
    if(level=='nas'):
        top_15_cor = df[df.columns].corrwith(actual).abs().sort_values(ascending=False).head(15)
        df_dump = df[top_15_cor.index.tolist()]

        return df_dump,top_15_cor
    elif(level=='ro'):
        ro_col_filtered = [col for col in ro_cols if col in df.columns]
        nas_col_filtered = [col for col in nas_cols if col in df.columns]
        if not ro_col_filtered and not nas_col_filtered:
            return df
        top_15_cor = pd.Series()
        above_average_corr_columns = pd.Series()

        if ro_col_filtered or nas_col_filtered:
            if nas_col_filtered:
                top_15_cor = df[nas_col_filtered].corrwith(actual).abs().sort_values(ascending=False).head(15)
            if ro_col_filtered:
                above_average_corr = df[ro_col_filtered].corrwith(actual).abs().sort_values(ascending=False)
                if(len(above_average_corr[above_average_corr > 0.6])==0):
                    above_average_corr_columns=above_average_corr.head(5)
                else:
                    above_average_corr_columns = above_average_corr[above_average_corr > 0.6]
        df = df[top_15_cor.index.tolist() + above_average_corr_columns.index.tolist()]
        combined_df = pd.concat([top_15_cor, above_average_corr_columns]).to_frame(name="Correlation")
        return df.reset_index(drop=True),combined_df
def train_xgboost_using_time_series_results(all_data_prophet,df_all,ro_cols,nas_cols,type_data):
    all_models_data = {}
    all_corr_df={}    
    all_feature_importances={}
    combinations = all_data_prophet[['Kanwil', 'Segmen', 'Product']].drop_duplicates()
    combinations = list(combinations.itertuples(index=False, name=None))
    df_all = df_all.rename(columns={'kanwil': 'Kanwil','segmen':'Segmen','produk':'Product','periode':'ds'})
    merged_df = pd.merge(all_data_prophet, df_all, how='inner', on=['ds', 'Kanwil', 'Segmen', 'Product'])
    merged_df_xgb=merged_df.drop(columns=['amount'])    
    merged_df_xgb['residu'] = merged_df_xgb['Amount'] - merged_df_xgb['Amount_Predict']
    all_xgb_results=pd.DataFrame()
    for comb in combinations:
        print(f"Processing {comb}")
        merged_df_xgb_test=merged_df_xgb.loc[(merged_df_xgb['Kanwil']==comb[0])&(merged_df_xgb['Product']==comb[2])&(merged_df_xgb['Segmen']==comb[1])]

        if(comb[0]=='JAKARTA 1' and comb[1]=='MIKRO'):
            train=merged_df_xgb_test.loc[(merged_df_xgb_test['ds']>='2022-01-01')&(merged_df_xgb_test['ds']<='2024-03-01')]
            test=merged_df_xgb_test.loc[(merged_df_xgb_test['ds']>'2024-03-01')&(merged_df_xgb_test['ds']<='2024-06-01')]
            forecast=merged_df_xgb_test.loc[(merged_df_xgb_test['ds']>'2024-06-01')]
        elif(comb[2]=='KUPEDES'and type_data=='pinjaman'):
            train=merged_df_xgb_test.loc[(merged_df_xgb_test['ds']<'2023-01-01')|
                                         ((merged_df_xgb_test['ds']>='2024-01-01')&
                                         (merged_df_xgb_test['ds']<='2024-03-01'))
                                         ]
            test=merged_df_xgb_test.loc[((merged_df_xgb_test['ds']>='2023-01-01')&(merged_df_xgb_test['ds']<='2023-12-01'))|((merged_df_xgb_test['ds']>='2024-04-01')&(merged_df_xgb_test['ds']<='2024-06-01'))]
            forecast=merged_df_xgb_test.loc[(merged_df_xgb_test['ds']>'2024-06-01')]
        elif(comb[2]=='KUR'and type_data=='pinjaman'):
            train = merged_df_xgb_test.loc[(merged_df_xgb_test['ds'] >= '2023-01-01')&
                                            (merged_df_xgb_test['ds'] <= '2024-03-31')]

            test=merged_df_xgb_test.loc[(merged_df_xgb_test['ds']>='2024-04-01')&(merged_df_xgb_test['ds']<='2024-06-01')]
            forecast=merged_df_xgb_test.loc[(merged_df_xgb_test['ds']>'2024-06-01')]
        else:
            train=merged_df_xgb_test.loc[merged_df_xgb_test['ds']<='2024-03-01']
            test=merged_df_xgb_test.loc[(merged_df_xgb_test['ds']>'2024-03-01')&(merged_df_xgb_test['ds']<='2024-06-01')]
            forecast=merged_df_xgb_test.loc[(merged_df_xgb_test['ds']>'2024-06-01')]


        X = train.drop(columns=['residu','Kanwil','Amount','Amount_Predict','Segmen','Product','Partition','Method'])
        y = train['residu']
        X['year'] = X['ds'].dt.year
        X['month'] = X['ds'].dt.month
        X = X.drop(columns=['ds'])

        X_test = test.drop(columns=['residu','Kanwil','Amount','Amount_Predict','Segmen','Product','Partition','Method'])
        y_test = test['residu']
        X_test['year'] = X_test['ds'].dt.year
        X_test['month'] = X_test['ds'].dt.month
        X_test = X_test.drop(columns=['ds'])

        X_forecast = forecast.drop(columns=['residu','Kanwil','Amount','Amount_Predict','Segmen','Product','Partition','Method'])
        y_forecast = forecast['residu']
        X_forecast['year'] = X_forecast['ds'].dt.year
        X_forecast['month'] = X_forecast['ds'].dt.month
        X_forecast = X_forecast.drop(columns=['ds'])
        
        scalers_train,X_scaled,list_of_col_log_scaled_train=process_data_numeric_xgb(X,'Robust')
        X_test_scaled=process_data_numeric_with_existing_scalers_xgb(X_test,scalers_train,list_of_col_log_scaled_train)
        X_forecast_scaled=process_data_numeric_with_existing_scalers_xgb(X_forecast,scalers_train,list_of_col_log_scaled_train)

        if(comb[0]=='NASIONAL'):
            X_train_scaled_extract,corr_df=feature_extraction_xgb(X_scaled.reset_index(drop=True),pd.Series(y),ro_cols,nas_cols,'nas')
        else:
            X_train_scaled_extract,corr_df=feature_extraction_xgb(X_scaled.reset_index(drop=True),pd.Series(y),ro_cols,nas_cols,'ro')

        X_test_scaled_extract=X_test_scaled[X_train_scaled_extract.columns].reset_index(drop=True)
        X_forecast_scaled_extract=X_forecast_scaled[X_train_scaled_extract.columns].reset_index(drop=True)


        scalers_train['residu']=RobustScaler()
        y=scalers_train['residu'].fit_transform(y.values.reshape(-1, 1)).flatten()
        y_test=scalers_train['residu'].transform(y_test.values.reshape(-1, 1)).flatten()
        y_forecast=scalers_train['residu'].transform(y_forecast.values.reshape(-1, 1)).flatten()

        model = XGBRegressor(objective='reg:squarederror', n_estimators=100, enable_categorical=True)

        param_grid = {
                'colsample_bytree': [0.5],
                'gamma': [0.1],
                'learning_rate': [0.01],
                'max_depth': [6],
                'min_child_weight': [3],
                'n_estimators': [100],
                'reg_alpha': [0.5],
                'reg_lambda': [0.5],
                'subsample': [0.6],
                'seed': [42],
        }
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, 
                                scoring='neg_root_mean_squared_error', 
                                cv=3, n_jobs=-1, verbose=1)

        grid_search.fit(X_train_scaled_extract, y)
        best_model = grid_search.best_estimator_


        y_pred = best_model.predict(X_train_scaled_extract)
        residuals_non_scaled = scalers_train['residu'].inverse_transform(y.reshape(-1, 1)).flatten()
        predictions_non_scaled = scalers_train['residu'].inverse_transform(y_pred.reshape(-1, 1)).flatten()
        train['predictions_xgb']=predictions_non_scaled
        train['Final_Predictions']=train['Amount_Predict']+train['predictions_xgb']

        rmse_scaled = mean_squared_error(y, y_pred, squared=False)
        rmse_non_scaled = mean_squared_error(residuals_non_scaled, predictions_non_scaled, squared=False)

        y_pred_test = best_model.predict(X_test_scaled_extract)
        residuals_non_scaled = scalers_train['residu'].inverse_transform(y_test.reshape(-1, 1)).flatten()
        predictions_non_scaled = scalers_train['residu'].inverse_transform(y_pred_test.reshape(-1, 1)).flatten()
        test['predictions_xgb']=predictions_non_scaled
        test['Final_Predictions']=test['Amount_Predict']+test['predictions_xgb']
        rmse_scaled = mean_squared_error(y_test, y_pred_test, squared=False)
        rmse_non_scaled = mean_squared_error(residuals_non_scaled, predictions_non_scaled, squared=False)

        y_pred_forecast = best_model.predict(X_forecast_scaled_extract)
        residuals_non_scaled = scalers_train['residu'].inverse_transform(y_forecast.reshape(-1, 1)).flatten()
        predictions_non_scaled = scalers_train['residu'].inverse_transform(y_pred_forecast.reshape(-1, 1)).flatten()
        forecast['predictions_xgb']=predictions_non_scaled
        forecast['Final_Predictions']=forecast['Amount_Predict']+forecast['predictions_xgb']
        combined_forecast_xgb = pd.concat([train, test,forecast])
        feature_importances = best_model.get_booster().get_score(importance_type='weight')
        df_feature_importances = pd.DataFrame(
            list(feature_importances.items()), 
            columns=['Feature', 'Importance']
        )
        df_feature_importances = df_feature_importances.sort_values(by='Importance', ascending=False)
        df_feature_importances_percent = calculate_feature_importance_in_percentage(feature_importances).reset_index(drop=True)
        model_data = {
            'model': best_model,
            'scalers_numerical_train': scalers_train,
            'scalers_categorical_train':None,
            'scalers_log_col':None,
            'coefficients_df_sorted':None
        }
        model_key = f'{comb[0]}_{comb[1]}_{comb[2]}'
        all_models_data[model_key] = model_data
        all_feature_importances[model_key]=df_feature_importances_percent
        all_corr_df[model_key]=corr_df
        all_xgb_results=pd.concat([all_xgb_results,combined_forecast_xgb])

        if(merged_df_xgb['Method'].unique()[0]=='PROPHET'):
            all_xgb_results['Method']='PROPHET_XGB'
        elif(merged_df_xgb['Method'].unique()[0]=='SARIMA'):
            all_xgb_results['Method']='SARIMA_XGB'
    return all_xgb_results,all_models_data,all_feature_importances,all_corr_df
