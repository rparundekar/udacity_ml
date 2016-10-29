# Import libraries necessary for this project
import numpy as np
import pandas as pd
from IPython.display import display # Allows the use of display() for DataFrames

# Import supplementary visualizations code visuals.py
import visuals as vs

# Pretty display for notebooks
%matplotlib inline

# Load the dataset
in_file = 'titanic_data.csv'
full_data = pd.read_csv(in_file)

# Print the first few entries of the RMS Titanic data
display(full_data.head())
# Store the 'Survived' feature in a new variable and remove it from the dataset
outcomes = full_data['Survived']
data = full_data.drop('Survived', axis = 1)

# Show the new dataset with 'Survived' removed
display(data.head())

# Define the accuracy function 
def accuracy_score(truth, pred):
    """ Returns accuracy score for input truth and predictions. """
    
    # Ensure that the number of predictions matches number of outcomes
    if len(truth) == len(pred): 
        
        # Calculate and return the accuracy as a percent
        return "Predictions have an accuracy of {:.2f}%.".format((truth == pred).mean()*100)
    
    else:
        return "Number of predictions does not match number of outcomes!"
    
# Test the 'accuracy_score' function, by creating a list of ones & compare with the outcomes 
predictions = pd.Series(np.ones(5, dtype = int))
print accuracy_score(outcomes[:5], predictions)

#PRedictions 0 - No one survived
def predictions_0(data):
    """ Model with no features. Always predicts a passenger did not survive. """

    predictions = []
    for _, passenger in data.iterrows():
        
        # Predict the survival of 'passenger'
        predictions.append(0)
    
    # Return our predictions
    return pd.Series(predictions)

# Make the predictions
predictions = predictions_0(data)
print accuracy_score(outcomes, predictions)

#See the survival stats for gender
vs.survival_stats(data, outcomes, 'Sex')

#Predictions1 - If you're a female, then you survive
def predictions_1(data):
    """ Model with one feature: 
            - Predict a passenger survived if they are female. """
    
    predictions = []
    for _, passenger in data.iterrows():
        
        #print(passenger['Sex']=='female');
        #If female, then survived; else, did not survived
        if passenger['Sex']=='female':
            predictions.append(1)
        else: 
            predictions.append(0)
    
    # Return our predictions
    return pd.Series(predictions)

# Make the predictions
predictions = predictions_1(data)
print accuracy_score(outcomes, predictions)

#Check age of male passengers
vs.survival_stats(data, outcomes, 'Age', ["Sex == 'male'"])

#Add that children less than 10 also survived
def predictions_2(data):
    """ Model with two features: 
            - Predict a passenger survived if they are female.
            - Predict a passenger survived if they are male and younger than 10. """
    
    predictions = []
    for _, passenger in data.iterrows():
        
        #If female, then survived; else
        if passenger['Sex']=='female':
            predictions.append(1)
        #Else if age<10, then survived; 
        elif passenger['Age']<10:
            predictions.append(1)
        #Else, did not survived    
        else: 
            predictions.append(0)
            
    # Return our predictions
    return pd.Series(predictions)

# Make the predictions
predictions = predictions_2(data)
print accuracy_score(outcomes, predictions)

#Visualize other possible people we can save
vs.survival_stats(data, outcomes, 'Embarked', ["Sex == 'male'",  "Age >= 10",  "Age <= 44",  "Fare >= 25",  "Fare < 27", "Pclass == 1"])

def predictions_3(data):
    """ Model with more than 2 features: 
            - Predict a passenger survived if they are female.
            - Predict a passenger survived if they are male and younger than 10. 
            - If fare is very high, then predict that he survived
            - First class male passengers younger than 45 & fare is 25 or 26$ survived """
    predictions = []
    for _, passenger in data.iterrows():
        
        #If female, then survived; else
        if passenger['Sex']=='female':
            predictions.append(1)
        #Else if age<10, then survived; 
        elif passenger['Age']<10:
            predictions.append(1)
        #Else if fare >400
        elif passenger['Fare']>400:
            predictions.append(1)
        #Else if fare >400
        elif (passenger['Pclass']==1 and passenger['Age']<45 and passenger['Fare']>=25 and passenger['Fare']<27):
            predictions.append(1)
       #Else, did not survived    
        else: 
            predictions.append(0)
            
    # Return our predictions
    return pd.Series(predictions)
# Make the predictions
predictions = predictions_3(data)
print accuracy_score(outcomes, predictions)



