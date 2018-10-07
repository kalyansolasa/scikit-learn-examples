import numpy as np
from sklearn import preprocessing

input_class  = ['car', 'bycle', 'moped', 'jeep', 'auto']
label_encoder = preprocessing.LabelEncoder()
label_encoder.fit(input_class)
print('actual data')
print(input_class)

print('lables using enmurate function')
for i,item in enumerate(input_class):
    print(item,i)

print('lables using lable classfier')
print(dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))



print("getting labes for ['moped', 'auto']")
test_lables=['moped', 'auto']
print(label_encoder.transform(test_lables))

print('getting values for [1,4,2] lables')
print(label_encoder.inverse_transform([1,4,2]))



