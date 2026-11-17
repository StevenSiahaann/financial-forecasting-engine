# bri-forecasting-project-dev

1. COMMAND to execute: 
python -m job-scripts.main --model_type {prophet_xgb/sarima_xgb} --type_data {simpanan/pinjaman}

example:
```
python -m job-scripts.main --model_type prophet_xgb --type_data simpanan
```

```
python -m job-scripts.main --model_type prophet_xgb --type_data pinjaman
```
2. If you want to add import put on job-scripts/global_import.py
3. If you want to do local testing, dont forget to create new directory and add to git ignore.
4. Structure Folder
a. data
Contains all the data needed (for local testing), on real case we used data from PA and cdsw. 
b. job-scripts
Contains all we need for automate all forecasting. Put all your utils that can be used for global configurations (for all models) on all_utils, and if it become spesific make the new utils for spesific models (e.g prophet.py)

Make the new script file for the new model on the parent of the job-scripts folder
c. modelling
If you want to put your local modelling, just put it here (optional)
d. results
All the results appears here (correlation analysis, dump_model, final_results for each model, feature importances, etc).
e. visualization
All the visualization appears here.
