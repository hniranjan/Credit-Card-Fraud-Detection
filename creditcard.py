#------------------------------------------------------------------------------
#                              IMPORT LIBRARIES
#------------------------------------------------------------------------------
import matplotlib 
import scipy.stats as stats
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn import ensemble
from sklearn.ensemble import IsolationForest

#------------------------------------------------------------------------------
#                              IMPORT DATASET
#------------------------------------------------------------------------------
data=pd.read_csv("creditcard.csv")


#------------------------------------------------------------------------------
#                           FINDING CORRELATION
#------------------------------------------------------------------------------
data.corr()
sns.heatmap(data.corr())



#------------------------------------------------------------------------------
#                 DIVIDING INTO X AND Y AND SPLITTING THE DATASET
#------------------------------------------------------------------------------
X=data.drop(['Class'],1)
Y=data['Class']

# Using Skicit-learn to split data into training and testing sets 
from sklearn.model_selection import train_test_split 
# Split the data into training and testing sets 
xo_Train, xo_Test, yo_Train, yo_Test = train_test_split( 
        X, Y, test_size = 0.2, random_state = 40)

#------------------------------------------------------------------------------
#                          Anomally Detection
#------------------------------------------------------------------------------
model  =  ensemble.IsolationForest(n_estimators=50, max_samples=500, contamination=.01, max_features=30, 
                         bootstrap=False, n_jobs=1, random_state=1, verbose=0, warm_start=False).fit(xo_Train)

# Get Anomaly Scores and Predictions
anomaly_score = model.decision_function(xo_Train)
predictions = model.predict(xo_Test)
xo_Train['Anomaly_score']=anomaly_score

ypred=predictions==1
ypred=ypred+0
ypred^=1


#------------------------------------------------------------------------------
#        PRINTING CLASSIFICATION REPORT FOR ANAMOLY DETECTION
#------------------------------------------------------------------------------

from sklearn.metrics import classification_report
print(classification_report(yo_Test,ypred))

#------------------------------------------------------------------------------
#               Up Sampling the classification dataset
#------------------------------------------------------------------------------

from sklearn.utils import resample
fraud    = data[data.Class==1]
not_fraud = data[data.Class==0]

# Upsample minority class
fraud_upsampled = resample(fraud,  replace=True,     # sample with replacement
                                 n_samples=100000,    # to match majority class
                                 random_state=123) # reproducible results
 
# Combine majority class with upsampled minority class
df_upsampled = pd.concat([fraud_upsampled, not_fraud])


#------------------------------------------------------------------------------
#            DIVIDING INTO X AND Y AND SPLITTING THE UPSAMPLED DATASET
#------------------------------------------------------------------------------
X=df_upsampled.drop(['Class'],1)
Y=df_upsampled['Class']

# Using Skicit-learn to split data into training and testing sets 
from sklearn.model_selection import train_test_split 
# Split the data into training and testing sets 
xTrain, xTest, yTrain, yTest = train_test_split( 
        X, Y, test_size = 0.2, random_state = 40)
 

#------------------------------------------------------------------------------
#                                 MODELS
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
#                           LOGISTIC REGRESSION

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

model1 = LogisticRegression(solver='liblinear', random_state=0)
model1.fit(xTrain, yTrain)

logistic_prediction= model1.predict(xo_Test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(yo_Test, logistic_prediction)
print(classification_report(yo_Test,logistic_prediction))


#------------------------------------------------------------------------------
#                       Naives_Bayes

from sklearn.naive_bayes import GaussianNB 
gnb = GaussianNB()
gnb.fit(xTrain, yTrain)
yPred_NB=gnb.predict(xTest)

# Classification Report
from sklearn.metrics import classification_report
print(classification_report(yTest,yPred_NB))
#------------------------------------------------------------------------------