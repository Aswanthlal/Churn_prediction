import pandas as pd
import numpy as np
import pylab as pl
import scipy.optimize as opt
from sklearn import preprocessing
import matplotlib.pyplot as plt

#Logistic regression
#While Linear Regression is suited for estimating continuous values (e.g. estimating house price),
#it is not the best tool for predicting the class of an observed data point. 
#In order to estimate the class of a data point, we need some sort of guidance on 
#what would be the most probable class for that data point. For this, we use Logistic Regression


#A telecommunications company is concerned about the number of customers leaving their land-line business for cable competitors. 
#They need to understand who is leaving. Imagine that you are an analyst at this company and you have to find out who is leaving and why.

#loading dataset
churn_df=pd.read_csv('E:\\datascience\\ed_x\\ed_x_project_and_lab\\csv\\ChurnData.csv')
churn_df.head()
churn_df.columns


#data preprocessing and selection
churn_df=churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip', 'callcard', 'wireless','churn']]
churn_df['churn'].astype('int')
churn_df.head()
churn_df.shape

#defining x nad y
x=np.asanyarray(churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']])
x[0:5]
y=np.asanyarray(churn_df['churn'])
y[0:5]


#normalizing the dataset
x=preprocessing.StandardScaler().fit(x).transform(x)
x[0:5]


#train/test dataset
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=4)
print('Train set:', x_train.shape,y_train.shape)
print('Test set:',x_test.shape,y_test.shape)


#modeling(logistic regression with scikit-learn)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
LR=LogisticRegression(C=0.01,solver='liblinear').fit(x_train,y_train)
LR

#predictimg test set
yhat=LR.predict(x_test)
yhat
#predict_proba returns estimates for all classes, ordered by the label of classes.
#So, the first column is the probability of class 0, P(Y=0|X), and second column is probability of class 1, P(Y=1|X):
yhat_prob=LR.predict_proba(x_test)
yhat_prob


#Evaluation
#jaccard index
#we can define jaccard as the size of the intersection divided by the size of the union of the two label sets. 
#If the entire set of predicted labels for a sample strictly matches with the true set of labels, 
#then the subset accuracy is 1.0; otherwise it is 0.0.
from sklearn.metrics import jaccard_score
jaccard_score(y_test,yhat,pos_label=0)

#confusion matrix
#another way of looking at the accuracy
from sklearn.metrics import classification_report, confusion_matrix
import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
print(confusion_matrix(y_test, yhat, labels=[1,0]))


# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, yhat, labels=[1,0])
np.set_printoptions(precision=2)


# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['churn=1','churn=0'],normalize= False,  title='Confusion matrix')
plt.show()
# The first row is for customers whose actual churn value in the test set is 1.
# As you can calculate, out of 40 customers, the churn value of 15 of them is 1. 
# Out of these 15 cases, the classifier correctly predicted 6 of them as 1, and 9 of them as 0. 

# This means, for 6 customers, the actual churn value was 1 in test set and classifier also correctly predicted those as 1. 
# However, while the actual label of 9 customers was 1, the classifier predicted those as 0, which is not very good. 
# We can consider it as the error of the model for first row.

#Lets look at the second row.
#It looks like  there were 25 customers whom their churn value were 0. 
#The classifier correctly predicted 24 of them as 0, and one of them wrongly as 1. So, 
# it has done a good job in predicting the customers with churn value 0. 
# A good thing about the confusion matrix is that it shows the model’s ability to correctly predict or separate the classes.  
# In a specific case of the binary classifier, such as this example,  
# we can interpret these numbers as the count of true positives, false positives, true negatives, and false negatives. 



print(classification_report(y_test,yhat))
#Precision is a measure of the accuracy provided that a class label has been predicted. It is defined by: precision = TP / (TP + FP)

#Recall is the true positive rate. It is defined as: Recall =  TP / (TP + FN)
    
# So, we can calculate the precision and recall of each class.

# F1 score:
# Now we are in the position to calculate the F1 scores for each label based on the precision and recall of that label. 

# The F1 score is the harmonic average of the precision and recall, where an F1 score reaches its best value at 1 (perfect precision and recall) 
# and worst at 0. It is a good way to show that a classifer has a good value for both recall and precision.

# Finally, we can tell the average accuracy for this classifier is the average of the F1-score for both labels, which is 0.72 in our case.


#log loss for evaluation
from sklearn.metrics import log_loss
log_loss(y_test,yhat_prob)
#In logistic regression, the output can be the probability of customer churn is yes (or equals to 1). 
#This probability is a value between 0 and 1.
#Log loss( Logarithmic loss) measures the performance of a classifier where the predicted output is a probability value between 0 and 1.


#Building another LR model with diff solver and regularization value.
LR2=LogisticRegression(C=0.01,solver='sag').fit(x_train,y_train)
yhat_prob2=LR2.predict_proba(x_test)
print('Log loss: :%.2f'%log_loss(y_test,yhat_prob2))