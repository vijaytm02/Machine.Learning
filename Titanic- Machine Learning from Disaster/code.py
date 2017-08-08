# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
np.set_printoptions(threshold=np.inf)
dataset = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

def get_title(name):
    if '.' in name:
        return name.split(',')[1].split('.')[0].strip()
    else:
        return 'Unknown'
    
def title_map(title):
    if title in ['Mr']:
        return 1
    elif title in ['Master']:
        return 3
    elif title in ['Ms','Mlle','Miss']:
        return 4
    elif title in ['Mme','Mrs']:
        return 5
    else:
        return 2
dataset['title'] = dataset['Name'].apply(get_title).apply(title_map)   
test_data['title'] = test_data['Name'].apply(get_title).apply(title_map)

# drop unnecessary columns, these columns won't be useful in analysis and prediction
dataset = dataset.drop(['PassengerId','Name','Ticket'], axis=1)
test_data    = test_data.drop(['Name','Ticket'], axis=1)

dataset["Embarked"] = dataset["Embarked"].fillna("S")

embark_dummies_titanic  = pd.get_dummies(dataset['Embarked'])
embark_dummies_titanic.drop(['S'], axis=1, inplace=True)
#print(embark_dummies_titanic)

embark_dummies_test  = pd.get_dummies(test_data['Embarked'])
embark_dummies_test.drop(['S'], axis=1, inplace=True)

dataset = dataset.join(embark_dummies_titanic)
#print(titanic_df)
test_data    = test_data.join(embark_dummies_test)

dataset.drop(['Embarked'], axis=1,inplace=True)
test_data.drop(['Embarked'], axis=1,inplace=True)

test_data["Fare"].fillna(test_data["Fare"].median(), inplace=True)

dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare'] = 2
dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
test_data.loc[ test_data['Fare'] <= 7.91, 'Fare'] = 0
test_data.loc[(test_data['Fare'] > 7.91) & (test_data['Fare'] <= 14.454), 'Fare'] = 1
test_data.loc[(test_data['Fare'] > 14.454) & (test_data['Fare'] <= 31), 'Fare'] = 2
test_data.loc[test_data['Fare'] > 31, 'Fare'] = 3

dataset['Fare'] = dataset['Fare'].astype(int)
test_data['Fare']    = test_data['Fare'].astype(int)

dataset.drop("Cabin",axis=1,inplace=True)
test_data.drop("Cabin",axis=1,inplace=True)

dataset['Family'] =  dataset["Parch"] + dataset["SibSp"]
dataset['Family'].loc[dataset['Family'] > 0] = 1
dataset['Family'].loc[dataset['Family'] == 0] = 0

test_data['Family'] =  test_data["Parch"] + test_data["SibSp"]
test_data['Family'].loc[test_data['Family'] > 0] = 1
test_data['Family'].loc[test_data['Family'] == 0] = 0

# drop Parch & SibSp
dataset = dataset.drop(['SibSp','Parch'], axis=1)
test_data    = test_data.drop(['SibSp','Parch'], axis=1)

sexes = sorted(dataset['Sex'].unique())
genders_mapping = dict(zip(sexes, range(0, len(sexes) + 1)))
dataset['Sex'] = dataset['Sex'].map(genders_mapping).astype(int)
test_data['Sex'] = test_data['Sex'].map(genders_mapping).astype(int)


d1=dataset.drop('Survived',axis=1)
d2=test_data.drop('PassengerId',axis=1)
frames = [d1,d2]
dfull= pd.concat(frames,ignore_index=True)

df1=dfull.loc[dfull['Age'].isnull(),:]
df2=dfull.loc[dfull['Age'].isnull()==False,:]

X=df2.iloc[:,[0,1,3,4,5,6,7]].values
Y=df2.iloc[:,2].values

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
Y = sc_y.fit_transform(Y)

# Fitting SVR to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X, Y)

X_t=df1.iloc[:,[0,1,3,4,5,6,7]].values
X_t=sc_X.transform(X_t)

y_pred = regressor.predict(X_t)
y_pred = sc_y.inverse_transform(y_pred)
dfull['Age'].loc[dfull['Age'].isnull()]=y_pred



dfull['Age'] = dfull['Age'].astype(int)

dfull.loc[ dfull['Age'] <= 16, 'Age'] = 0
dfull.loc[(dfull['Age'] > 16) & (dfull['Age'] <= 32), 'Age'] = 1
dfull.loc[(dfull['Age'] > 32) & (dfull['Age'] <= 48), 'Age'] = 2
dfull.loc[(dfull['Age'] > 48) & (dfull['Age'] <= 64), 'Age'] = 3
dfull.loc[(dfull['Age'] > 64), 'Age'] = 4




dfull['age_class'] = dfull['Age'] * dfull['Pclass']
X_train = dfull.iloc[0:891,:].values
Y_train = dataset.iloc[:,0].values
X_test = dfull.iloc[891:,:].values



#from sklearn.svm import SVC
#classifier = SVC(kernel = 'rbf', random_state = 0)
#classifier.fit(X_train, Y_train)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=100, criterion='entropy', max_depth=10, max_features='sqrt', min_samples_split=5)

classifier.fit(X_train, Y_train)

Y_pred = classifier.predict(X_test)

classifier.score(X_train, Y_train)

submission = pd.DataFrame({
        "PassengerId": test_data["PassengerId"],
        "Survived": Y_pred
    })
submission.to_csv('titanic.csv', index=False)
