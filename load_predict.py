# importing libraries
import joblib
import numpy as np
from sklearn import model_selection
from datetime import datetime


# uploading pikle file of the model
model = joblib.load('model.pkl')
encoders = joblib.load('encoders.pkl')
features = joblib.load('features.pkl')

# Function to take raw input
def raw_input_tensor():
    input_data = {
        "VehicleType": input("Vehicle Type : "),
        "Model": input("Model : "),
        "Kilometer": int(input("Kilometer : ")),
        "NotRepaired": input("NotRepaired :"),
        "NumberOfPictures": int(input("Number of Picture :")),
        "FuelType": input("Fuel type : "),
        "RegistrationYear": int(input("Registration Year : ")),
        "RegistrationMonth": int(input("Registration Month : ")),
        "Gearbox": input("Gearbox : "),
        "Brand": input("Brand : "),
        "PostalCode": int(input("Postal Code : ")),
        "Power": int(input("Power : ")),
        "DateCrawled": input("Date Crawled : "),
        "DateCreated": input("Date Created : "),
        "LastSeen": input("Last Seen : ")
    }
    return input_data


# Function to transform the input_tensor into numeric type for prediction
def transform_input_tensor(input_data,encoders):
    date_created_str = input_data['DateCreated']
    date_created_obj = datetime.strptime(date_created_str, '%Y-%m-%d %H:%M:%S')
    input_data['DateCreated_month'] = int(date_created_obj.month)
    input_data['DateCreated_hour'] = int(date_created_obj.hour)
    input_data['DateCreated_day'] = int(date_created_obj.day)
    date_LastSeen_str = input_data['LastSeen']
    date_LastSeen_obj = datetime.strptime(date_LastSeen_str, '%Y-%m-%d %H:%M:%S')
    input_data['LastSeen_hour'] = int(date_LastSeen_obj.hour)
    input_data['LastSeen_month'] = int(date_LastSeen_obj.month)
    input_data['LastSeen_day'] = int(date_LastSeen_obj.day)
    date_DateCrawled_str = input_data['DateCrawled']
    date_DateCrawled_obj = datetime.strptime(date_DateCrawled_str, '%Y-%m-%d %H:%M:%S')
    input_data['DateCrawled_hour'] = int(date_DateCrawled_obj.hour)
    input_data['DateCrawled_day'] = int(date_DateCrawled_obj.day)
    input_data['DateCrawled_month'] = int(date_DateCrawled_obj.month)

    model_encoder = encoders['Model']
    input_data['Model'] = model_encoder.transform([input_data['Model']])

    fueltype_encoder = encoders['FuelType']
    input_data['FuelType'] = fueltype_encoder.transform([input_data['FuelType']])

    brand_encoder = encoders['Brand']
    input_data['Brand'] = brand_encoder.transform([input_data['Brand']])

    vehicletype_encoder = encoders['VehicleType']
    input_data['VehicleType'] = vehicletype_encoder.transform([input_data['VehicleType']])

    if(input_data['NotRepaired']=='yes'):
        input_data['NotRepaired']=1
    else:
        input_data['NotRepaired'] = 0

    if(input_data['Gearbox']=='auto'):
        input_data['Gearbox']=1
    else:
        input_data['Gearbox']=0

    # Deleting the Keys
    del input_data['DateCreated']
    del input_data['LastSeen']
    del input_data['DateCrawled']

    return input_data


def give_input_tensor(features,input_data):
    input_tensor = []
    for f in features:
        input_tensor.append(input_data[f])
    return np.array(input_tensor)

input_data = raw_input_tensor()

input_data = transform_input_tensor(input_data,encoders)

data_tensor = give_input_tensor(features,input_data)

predictions = model.predict(data_tensor.reshape(1,-1))[0]

print("Pridicted Price : ",predictions)



