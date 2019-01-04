
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Importing dataset
dataset = pd.read_csv('Data.csv')
X= dataset.iloc[ : , :-1].values
Y= dataset.iloc[ : , 3].values
print('origin \n\nX=\n', X , '\n\nY=\n' , Y , '\n')



# Handling the missing data
Imputer = SimpleImputer(missing_values = np.nan, strategy = "mean")
Imputer.fit(X[:,1:3])
X[ : , 1:3] = Imputer.transform(X[ : , 1:3])

print('Handling the missing data \n\nX=\n', X ,'\n')
# Encoding categorical data
labelencoder_X = LabelEncoder()
# X第一列的文字 = LabelEncoder轉換成數字(只針對第一列)
X[ : , 0] = labelencoder_X.fit_transform(X[ : , 0])

labelencoder_Y = LabelEncoder()
Y =  labelencoder_Y.fit_transform(Y)
print(' Encoding categorical data \n\nX=\n', X , '\n\nY=\n' , Y ,'\n')

# Creating a dummy variable
onehotencoder = OneHotEncoder(categorical_features=[0], dtype=int)
X= onehotencoder.fit_transform(X).toarray()
print('Creating a dummy variable \n\nX=\n', X,'\n')

#Splitting the datasets into training sets and Test sets
X_train, X_test, Y_train, Y_test = train_test_split( X , Y , test_size = 0.2, random_state = 0)
print('training sets and Test sets\nX_train=\n', X_train,"\nX_test\n",X_test,'\nY_train=\n', Y_train,"\nY_test\n",Y_test,'\n')

#Feature Scaling
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)

print('Feature Scaling\nX_train=\n', X_train,"\nX_test\n",X_test,'\n')