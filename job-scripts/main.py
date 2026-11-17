from .global_import import *
from .forecasting_prophet_xgb import *
from .forecasting_sarima_xgb import *
from .forecast_with_existing_models import *
from .utils.logging import *
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Forecast and train models with the given type of data")
    parser.add_argument('--model_type', type=str, required=True, help="Specify the model type to use for forecasting and training")
    parser.add_argument('--type_data', type=str, required=True, help="Specify the type of data to use for forecasting and training")
    parser.add_argument('--pipeline', type=str, required=True, help="Specify the type of the pipeline to use for : forecasting or retraining")
    parser.add_argument('--version', type=str, required=True, help="Specify the type of the version control on IBM PA : actual or forecast or new type of version that you create on IBM PA")
    parser.add_argument('--hierarchy_type',type=str,required=True,help="Hierarchy level of forecasting/retraining : All RO or Specific RO")
    args = parser.parse_args()

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    file_handler_html = logging.FileHandler(f"log_monitoring/{args.type_data}/html/{args.model_type}_forecast.html")
    file_handler_html.setFormatter(HTMLFormatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler_html)

    file_handler_txt = logging.FileHandler(f"log_monitoring/{args.type_data}/txt/{args.model_type}_forecast.txt")
    file_handler_txt.setFormatter('%(asctime)s - %(levelname)s - %(message)s')
    logger.addHandler(file_handler_txt)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(CustomFormatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(console_handler)

    if(args.pipeline=='forecasting'):
        logger.info(f"Start Forecasting Pipeline using {args.model_type} on {args.type_data} data.")
        all_xgb_results=forecast_with_existing_model(args.model_type,args.type_data,args.version,args.hierarchy_type)
        logger.info(f"Finish Forecasting Pipeline using {args.model_type} on {args.type_data} data.")

        # TODO: handle store data to tm1 py here
    elif(args.pipeline=='retraining'):
        if args.model_type == "prophet_xgb" and args.model_type:
            logger.info(f"Running Prophet Xgboost model on {args.type_data} data.")
            # print(f"Running Prophet Xgboost model for type_data: {args.type_data}")
            forecast_and_train_all_prophet_xgb(args.model_type,args.type_data,args.version,args.hierarchy_type)
        elif args.model_type == "sarima_xgb":
            logger.info(f"Running Sarima Xgboost model on {args.type_data} data.")
            # print(f"Running Sarima Xgboost model for type_data: {args.type_data}")
            forecast_and_train_all_sarima_xgb(args.model_type,args.type_data,args.version,args.hierarchy_type)
        else:
            logger.critical(f"Invalid model choice: {args.model_choose}. Please choose 'prophet_xgb' or 'sarima_xgb'.")
    else:
        logger.critical(f"Invalid Pipeline choice: {args.pipeline}. Please choose 'forecasting' or 'retraining'.")