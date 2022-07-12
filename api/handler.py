import pickle
import pandas as pd
from flask import Flask, request, Response
from HealthInsurance import HealthInsurance

# loading model
path = '/home/vitor/Repos/cross_sell_project/cross-sell/'
model = pickle.load(open(path + 'model/logistic_regression.pkl', 'rb'))

# initialize API
app = Flask(__name__)

@app.route('/predict', methods=['POST'])                                  
def health_insurance_predict():
    test_json = request.get_json()          

    if test_json:                                                                      
        if isinstance(test_json, dict):                                             
            test_raw = pd.DataFrame(test_json, index=[0])
        
        else:                                                                      
            test_raw = pd.DataFrame(test_json, columns=test_json[0].keys())        


        # Instantiate the HI Class
        pipeline = HealthInsurance()

        # Data Cleaning
        df1 = pipeline.data_cleaning(test_raw)

        # Feature Engineering
        df2 = pipeline.feature_engineering(df1)

        # Data Preparation
        df3 = pipeline.data_preparation(df2)

        # Prediction
        df_response = pipeline.get_prediction(model, test_raw, df3)
        
        return df_response

    else:
        return Response ('{}', status=200, mimetype='application/json')

if __name__ == '__main__':
    app.run('0.0.0.0', debug=True)