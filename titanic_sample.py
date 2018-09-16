import numpy as np
import pandas as pd

import tensorflow as  tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import np_utils
from keras.backend import tensorflow_backend
config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
session = tf.Session(config=config)
tensorflow_backend.set_session(session)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics

df_train = pd.read_csv("/data/train.csv")
df_test = pd.read_csv("/data/test.csv")

features = list(df_train.columns.values)
# Remove unwanted features
features.remove('Name')
features.remove('PassengerId')
features.remove('Survived')
features.remove('Ticket')
features.remove('SibSp')
features.remove('Parch')
features.remove('Fare')
features.remove('Cabin')
features.remove('Embarked')
print(features)

# encode sex into ints
le = LabelEncoder()
df_train['Sex'] = le.fit_transform(df_train['Sex'])
df_test['Sex'] = le.fit_transform(df_test['Sex'])

df_train['Fare'] = df_train['Fare'].fillna(df_train['Fare'].mean())
df_test['Fare'] = df_train['Fare'].fillna(df_train['Fare'].mean())

df_train['Age'] = df_train['Age'].fillna(df_train['Age'].mean())
df_test['Age'] = df_train['Age'].fillna(df_train['Age'].mean())

df_train['Embarked'] = df_train['Embarked'].fillna("S")
df_test['Embarked'] = df_test['Embarked'].fillna("S")
df_train['Embarked'] = le.fit_transform(df_train['Embarked'])
df_test['Embarked'] = le.fit_transform(df_test['Embarked'])

df_train['Cabin'] = df_train['Cabin'].fillna("None")
df_test['Cabin'] = df_test['Cabin'].fillna("None")
df_train['Cabin'] = le.fit_transform(df_train['Cabin'])
df_test['Cabin'] = le.fit_transform(df_test['Cabin'])

df_train['Ticket'] = le.fit_transform(df_train['Ticket'])
df_test['Ticket'] = le.fit_transform(df_test['Ticket'])

y = df_train['Survived']
x = df_train[features]
x_t = df_test[features]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25,random_state=32)

model = Sequential()
model.add(Dense(output_dim=64, input_dim=3))
model.add(Activation("relu"))
model.add(keras.layers.core.Dropout(0.2))
model.add(Dense(output_dim=64))
model.add(Activation("relu"))
model.add(Dense(output_dim=2))
model.add(Activation("softmax"))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

y_train = np_utils.to_categorical(y_train)
a = model.fit(X_train.values, y_train, nb_epoch=500)

y_test = np_utils.to_categorical(y_test)
loss_and_metrics = model.evaluate(X_test.values, y_test)
print(loss_and_metrics)

classes = model.predict_classes(x_t.values, batch_size=32)
print(classes)

submission = pd.DataFrame({
    "PassengerId": df_test["PassengerId"],
    "Survived": classes})
print(submission)

submission.to_csv('submission.csv', index=False)

