import numpy as np
import pandas as pd
import math
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler

data=pd.read_csv('Creditcard_data.csv')

x=data.loc[ : , data.columns != 'Class']
y=data['Class']

# DataSet is Imbalanced since there are 793 entries
# for not fraud and 9 entries for fraud

resampler=RandomOverSampler(random_state=42)
X_resample,y_resample=resampler.fit_resample(x, y)

# Balanced Data Frame
new_data=pd.DataFrame(X_resample)
new_data['Class']=y_resample


# 1. Simple Random Sampling

z=1.96
p=0.5
E=0.05
sr=(z*z*p*(1-p))/(E*E)
sample_size = math.ceil(sr)

samples=[]
s1 = new_data.sample(n=sample_size, random_state=0)
samples.append(s1)

# 2. Systematic Sampling

n = len(new_data)
k = int(math.sqrt(n))
s2 = new_data.iloc[::k]
samples.append(s2)

# 3.Cluster Sampling


z=1.96
p=0.5
E=0.05
C=1.5
cs=((z**2)*p*(1-p))/((E/C)**2)
sample_size = round(cs)
num_select_clusters=2
df_new=new_data
N = len(new_data)
K = int(N/sample_size)
data = None
for k in range(K):
    sample_k = df_new.sample(sample_size)
    sample_k["cluster"] = np.repeat(k,len(sample_k))
    df_new = df_new.drop(index = sample_k.index)
    data = pd.concat([data,sample_k],axis = 0)

random_chosen_clusters = np.random.randint(0,K,size = num_select_clusters)
s3 = data[data.cluster.isin(random_chosen_clusters)]
s3.drop(['cluster'], axis=1, inplace=True)
samples.append(s3)


# 4.Stratified Sampling

s4=new_data.groupby('Class', group_keys=False).apply(lambda x: x.sample(190))
samples.append(s4)

# 5.Convinience Sampling

s5=new_data.head(400)
samples.append(s5)

# Calculating accuracies for different types of sampling and making the data frame for the same

heading=['Simple Random','Systematic','Cluster','Stratified','Convenience']
ans=pd.DataFrame(columns=heading, index=['Logistic Regression','SVM','KNN','Decision Tree','Random Forest'])

# Functions to calculate accuracies of different ML models

def logistic(xtrain, xtest, y_train, y_test,i,j,ans):
    classifier = LogisticRegression(random_state = 0,max_iter=2000)
    classifier.fit(xtrain, y_train)
    y_pred = classifier.predict(xtest)
    acc = accuracy_score(y_test, y_pred)
    ans.iloc[j,i]=acc*100

def svm(xtrain, xtest, y_train, y_test,i,j,ans):
    clf = SVC(kernel='rbf')
    clf.fit(xtrain, y_train) 
    y_pred=clf.predict(xtest)
    acc = accuracy_score(y_test, y_pred)
    ans.iloc[j+1,i]=acc*100
def knnn(xtrain, xtest, y_train, y_test,i,j,ans):
    knn = KNeighborsClassifier(n_neighbors=7)
    knn.fit(xtrain, y_train)
    y_pred=knn.predict(xtest)
    acc = accuracy_score(y_test, y_pred)
    ans.iloc[j+2,i]=acc*100
def dtree(xtrain, xtest, y_train, y_test,i,j,ans):
    clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100,max_depth = 3, min_samples_leaf = 5)
    clf_entropy.fit(xtrain, y_train)
    y_pred=clf_entropy.predict(xtest)
    acc = accuracy_score(y_test, y_pred)
    ans.iloc[j+3,i]=acc*100
def randforest(xtrain, xtest, y_train, y_test,i,j,ans):
    clf = RandomForestClassifier(n_estimators = 100) 
    clf.fit(xtrain, y_train)
    y_pred = clf.predict(xtest)
    acc = accuracy_score(y_test, y_pred)
    ans.iloc[j+4,i]=acc*100

# Applying Models on each samples obtained from each sampling technique

for i in range(5):
    j=0
    x_sam=samples[i].drop('Class',axis=1)
    y_sam=samples[i]['Class']

    # Splitting into train and test dataset
    xtrain, xtest, y_train, y_test = train_test_split(x_sam ,y_sam , random_state=104, test_size=0.25, shuffle=True)
    logistic(xtrain, xtest, y_train, y_test,i,j,ans)
    svm(xtrain, xtest, y_train, y_test,i,j,ans)
    knnn(xtrain, xtest, y_train, y_test,i,j,ans)
    dtree(xtrain, xtest, y_train, y_test,i,j,ans)
    randforest(xtrain, xtest, y_train, y_test,i,j,ans)

print(ans)