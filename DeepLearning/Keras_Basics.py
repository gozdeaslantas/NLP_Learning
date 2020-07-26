import numpy as np
from sklearn.datasets import load_iris

iris = load_iris()
#print(iris.DESCR)
print(iris.feature_names)
X = iris.data
y = iris.target
#class 0 --> [1,0,0]
#class 1 --> [0,1,0]
#class 2 --> [0,0,1]

from keras.utils import to_categorical
y = to_categorical(y)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33,random_state=42)
from sklearn.preprocessing import MinMaxScaler
#np.array([5,10,15,20])/20

scaler_object = MinMaxScaler()
scaler_object.fit(X_train)

scaled_X_train = scaler_object.transform(X_train)
scaled_X_test = scaler_object.transform(X_test)


from keras.models import Sequential
from keras.layers import Dense


model = Sequential()
#add in layers
model.add(Dense(8,input_dim=4,activation='relu'))  #input neuron
model.add(Dense(8,input_dim=4,activation='relu'))  #hidden neuron
model.add(Dense(3,activation='softmax')) # [0.2,0.5,0.3] #output neuron
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


model.summary()

model.fit(scaled_X_train,y_train,epochs=150,verbose=2)
model.predict_classes(scaled_X_test)
predictions = model.predict_classes(scaled_X_test)

#convert test classes into format as indexes
y_test.argmax(axis=1)

from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

confusion_matrix(y_test.argmax(axis=1),predictions)
print(classification_report(y_test.argmax(axis=1),predictions))
accuracy_score(y_test.argmax(axis=1),predictions)
model.save('myfirstmodel.h5') #to save the whole network

from keras.models import load_model

new_model = load_model('myfirstmodel.h5')

new_model.predict_classes(scaled_X_test)
