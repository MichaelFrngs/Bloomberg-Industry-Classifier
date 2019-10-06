import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#import seaborn as sns
#import matplotlib.pyplot as plt
#from sklearn.preprocessing import StandardScaler
#import scipy as sp 
#import sklearn
#import random 
#import time 
import datetime as dt
from sklearn import preprocessing, model_selection
from keras.layers import Dropout
from keras.models import Sequential 
from keras.layers import Dense 
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical
from sklearn.utils import shuffle
import os

os.chdir("C:/Users/Michael_Frangos/Desktop/Bootcamp project")
#Import industry metrics from CAPIQ
Airline_metrics = pd.read_excel("Airlines Key Stats Ratios.xlsx")
Bank_metrics = pd.read_excel("Banks Key Stats Ratios.xlsx")
Beverage_metrics = pd.read_excel("Beverages Key Stats Ratios.xlsx")
Telecommunication_metrics = pd.read_excel("Diversified Telecommunication Services Key Stats Ratios.xlsx")
Food_and_Staples_metrics = pd.read_excel("Food and Staples Retailing Key Stats Ratios.xlsx")
Oil_Gas_metrics = pd.read_excel("Oil Gas and Consumable Fuels Key Stats Ratios.xlsx")
Pharmaceutical_metrics = pd.read_excel("Pharmaceuticals Key Stats Ratios.xlsx")
Semiconductor_metrics = pd.read_excel("Semiconductors and Semiconductor Equipment Key Stats Ratios.xlsx")
Software_metrics = pd.read_excel("Software Key Stats Ratios.xlsx")

#Individual Data
os.chdir("C:/Users/Michael_Frangos/Desktop/Bootcamp project/Individual Company Data/")
I_Discount_Airlines = pd.read_excel("1. Discount Airlines.xlsx")
I_LiquorCompanies = pd.read_excel("2. Liquor Breweries.xlsx")
I_Banks = pd.read_excel("3. Commercial Banks.xlsx")
I_Software = pd.read_excel("4. Software Companies.xlsx")
I_Oil_and_Gas = pd.read_excel("5. Large Integrated Oil and Gas.xlsx")
I_Phones = pd.read_excel("6. Mobile Phone Service Providers.xlsx")
I_Pharma = pd.read_excel("7. Pharma.xlsx")
I_Groceries = pd.read_excel("8. Grocery Stores.xlsx")
I_Semiconductors = pd.read_excel("9. R&D Semiconductor Firms.xlsx")

#Industry Data
Data_List = [Airline_metrics,Bank_metrics,Beverage_metrics,Telecommunication_metrics,Food_and_Staples_metrics,Oil_Gas_metrics,Pharmaceutical_metrics,
             Semiconductor_metrics,Software_metrics]

Data_List2 = [I_Discount_Airlines, I_LiquorCompanies,I_Banks,I_Software,I_Oil_and_Gas,I_Phones,I_Pharma,I_Groceries,I_Semiconductors]

DataLists = [Data_List, Data_List2]


for data_set in Data_List:
    data_set.columns = ['Name', 'Return on Assets %', 'Return on Equity %', 'Gross Margin %',
       'Net Income Margin %', 'Total Asset Turnover', 'Current Ratio',
       'Avg Days Inventory Outstanding', 'Quick Ratio',
       'Avg Days Sales Outstanding', 'LT Debt/Equity', 'LT Debt/Capital',
       'Target Label']
    
for data_set in Data_List2:
    data_set.columns = ['Name', 'Current Ratio', 'Quick Ratio',
       'Avg Days Inventory Outstanding', 'Avg Days Sales Outstanding',
       'Total Asset Turnover', 'LT Debt/Capital', 'LT Debt/Equity',
       'Gross Margin %', 'Net Income Margin %', 'Return on Assets %',
       'Return on Equity %', 'Target Label']
    
data = pd.concat(Data_List)
data2 = pd.concat(Data_List2)

columns = data.columns
columns2 = data2.columns

for column in columns:
    try:
        data[column] = data[column].astype(str)
        data[column] = data[column].str.replace("%", "")
        data[column] = data[column].str.replace("(", "")
        data[column] = data[column].str.replace(")", "")
        data[column] = data[column].astype(float)
    except:
        pass
    
for column in columns2:
    try:
        data2[column] = data2[column].astype(str)
        data2[column] = data2[column].str.replace("%", "")
        data2[column] = data2[column].str.replace("(", "")
        data2[column] = data2[column].str.replace(")", "")
        data2[column] = data2[column].astype(float)
    except:
        pass
    

#data = data.fillna(0)
data["Gross Margin %"] = data["Gross Margin %"]/100
data["LT Debt/Capital"] = data["LT Debt/Capital"]/100
data["LT Debt/Equity"] = data["LT Debt/Equity"]/100
data["Net Income Margin %"] = data["Net Income Margin %"]/100
data["Return on Assets %"] = data["Return on Assets %"]/100
data["Return on Equity %"] = data["Return on Equity %"]/100

data2["Gross Margin %"] = data2["Gross Margin %"]/100
data2["LT Debt/Capital"] = data2["LT Debt/Capital"]/100
data2["LT Debt/Equity"] = data2["LT Debt/Equity"]/100
data2["Net Income Margin %"] = data2["Net Income Margin %"]/100
data2["Return on Assets %"] = data2["Return on Assets %"]/100
data2["Return on Equity %"] = data2["Return on Equity %"]/100
data2["Quick Ratio"] = data2["Quick Ratio"]/100
data2["Current Ratio"] = data2["Current Ratio"]/100
#Industry Dataframe
data = pd.concat([data,data2])

#Reorder the columns
cols = data.columns.tolist()
last_column = cols[-1:]
All_other_columns = cols[:-2]
Second_to_last_column = cols[-2]
cols = All_other_columns + last_column + [Second_to_last_column]
data = data[cols]

data = shuffle(data)
data = data.set_index("Name")


#data = data[i:].reset_index(drop = True)




X = data.drop(['Target Label'], axis = 1)
X = np.array(X)
X = preprocessing.scale(X)

Y = data['Target Label']


# Transform Target Labels into numerical values 
encoder = LabelEncoder()
encoder.fit(Y)
#Transform strings into a number label
Y = encoder.transform(Y)
Y = np_utils.to_categorical(Y)
#print(Y)

# We have 9 classes : the output looks like : 
#0,0,1 : Class 1
#0,1,0 : Class 2
#1,0,0 : Class 3

#Splits the data into a training set and a test set
#train_x, test_x, train_y, test_y = model_selection.train_test_split(X,Y,test_size = 0.1, random_state = 0)
train_x, test_x, train_y, test_y = model_selection.train_test_split(X,Y,test_size = 0, random_state = 0)

#Lets build the model
input_dim = len(data.columns) - 1
model = Sequential()
model.add(Dense(1000, input_dim = input_dim , activation = 'relu'))
model.add(Dropout(.3))
model.add(Dense(4000, activation = 'relu'))
model.add(Dropout(.3))
model.add(Dense(4000, activation = 'relu'))
model.add(Dropout(.3))
model.add(Dense(1000, activation = 'relu'))
model.add(Dropout(.3))
model.add(Dense(9, activation = 'softmax'))

#model.summary()

model.compile(loss = 'categorical_crossentropy' , optimizer = 'adam' , metrics = ["categorical_accuracy"])


os.chdir("C:/Users/Michael_Frangos/Desktop/Bootcamp project")
#Create logs to track on tensorboard ||| cd: main directory --> tensorboard --logdir=logs
Log_File_Name = f"log2"
from tensorflow.keras.callbacks import TensorBoard
tensorboard = TensorBoard(log_dir= f"logs\{Log_File_Name}")
#Fit the model
model.fit(train_x, train_y, epochs = 40, batch_size = 236, callbacks=[tensorboard], validation_split=0.1)
model.save("C:/Users/Michael_Frangos/Desktop/Bootcamp project/Classification.model")

##Checks for overfitting. Best so far is 91%
#scores = model.evaluate(test_x, test_y)
#print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


##Number of row
#i = 8model.compile(loss = 'sparse_categorical_crossentropy' , optimizer = 'adam' , metrics = "sparse_categorical_accuracy" )

#data_to_predict = data[:i].reset_index(drop = True)
#predict_species = data_to_predict["Target Label"]
#predict_species = np.array(predict_species)
#prediction = np.array(data_to_predict.drop(['Target Label'],axis= 1))


os.chdir("C:/Users/Michael_Frangos/Desktop/Bootcamp project")
#Data we want to predict on
Load = pd.read_excel("Make Predictions.xlsx")
prediction_data1 = Load
prediction_data1 = prediction_data1.set_index("Ratio")

#data = shuffle(data)
#prediction_data1 = data


#Dropping the index
data_to_predict = prediction_data1.reset_index(drop = True)
#data_to_predict = data


#Removes the target label. Prepares to predict
prediction_data1 = preprocessing.scale(np.array(data_to_predict.drop(['Target Label'],axis= 1)))

#test = pd.DataFrame(prediction)

#Make predictions
predictions = model.predict_classes(prediction_data1)
#Convert numbers in the array to position`in the array
prediction_ = np.argmax(to_categorical(predictions), axis = 1)
prediction_ = encoder.inverse_transform(prediction_)


#Print predictions
for i,num in zip(prediction_,predictions):
    print( f"The nn predicts {num}. {i}")
    
    
Classification_probabilities = pd.DataFrame(model.predict(prediction_data1))
#Tells you which number is assigned to each class.
def Order_Of_classes():
    i=0
    class_names = []
    Order_Of_classes = sorted(["Oil and Gas","Liquor","Airlines","Software","Banks","Semiconductors","TELECOM","Pharma", "Retail Grocery"])
    for classification in Order_Of_classes:
        print(i,classification)
        i=i+1
        class_names.append(f"{i}. " + classification)
    return class_names

Classification_probabilities.columns = Order_Of_classes()
Classification_probabilities_colored = model.predict(prediction_data1)
#Company 7 is a bank because they have highest days in inventory and gross margins
