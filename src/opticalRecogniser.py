print "Setting up system"
print "Importing libraries"
import subprocess
import pandas as pd
from datetime import datetime as dt
import sklearn.neural_network as sknn
from sklearn.utils.validation import column_or_1d
from sklearn.neighbors import KNeighborsClassifier as knc

#seed=int(str(dt.now()).translate(None,":-. "))
i=dt.now()
seed=int(i.strftime("%H%M%S"))

## Read training Data file
print "Reading training dataset"
try:
    data = pd.read_csv(open('../DataSet/DigitRecognition/optdigits_raining.csv'))
except Exception as e:
    print "Error in opening Tarining Dataset"
    print "Error Message : "+str(e)
    exit(1)

## Read Test dataset
print "Reading test dataset"
try:
    testdata = pd.read_csv(open('../DataSet/DigitRecognition/optdigits_test.csv'))
except Exception as e:
    print "Failed to open testing data set : "+str(e)
    ip=input("Do you wish to continue(Y/N)?").lower()
    if ip.contanins('y'):
        print "Proceeding without test data set..."
        pass
    else:
        print "Exitting after Cleanup..."
        data.close()
        exit(1)

for x in ['a1','a8','h1','h8']:
    del data[x]
    del testdata[x]


## Separate features and results in training dataset
print "Processing training dataset"
targetDataResult = (data['result'])#.reshape(-1,1)
targetFeatures = list(data.columns[:-1])
targetDataFeatures=data[targetFeatures]

## Fit the training data set into Multi Layer Perceptron classifier
## This builds us a Neural Network
print "Building a Neural Network from training data"

knc_map = knc(n_neighbors=8, weights='distance')
knc_map.fit(targetDataFeatures,targetDataResult)

print "Processing test dataset"
## Separate features and results in Test dataset
testtarget = testdata['result']
testFeatureList = list(testdata.columns[:-1])
testResults=testdata['result']
testFeatures=testdata[testFeatureList]

## Predict test data based on the Neural Network formed from training dataset
predictedTestResults=list(knc_map.predict(testFeatures))

actualTestResults=list(testResults)
#print "Prediction complete"
print "Calculating Prediction results"
s=f=0
for i in range(len(actualTestResults)-2):
	if actualTestResults[i] == predictedTestResults[i]:
		s=s+1
	else:
		f=f+1
print "--------------------------------------------------------------------------"
print "\n\n"
print "Percentage of data that is predicted correctly is "+str(float(s)/len(actualTestResults)*100.0)
print "Percentage of data that is predicted wrong is "+str(float(f)/len(actualTestResults)*100.0)
print "Prediction rate is "+str(knc_map.score(testFeatures,testResults)*100)
print "Prediction rate is "+str(float(s)/len(actualTestResults)*100.0)
print "\n"
print "--------------------------------------------------------------------------"

