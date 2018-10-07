import numpy as np
from sklearn import preprocessing

data = np.array([[3, -1.5, 2, -5.4],
                 [0, 4, -0.3, 2.1],
                 [1, 3.3, -1.9,-4.3]])

print(data)

print("mean and std")

data_standardised = preprocessing.scale(data)
print('mean:',data_standardised.mean(axis=0))
print('stdv:',data_standardised.std(axis=0))

print('Minmax and Minabs scalling')
data_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
data_scalered = data_scaler.fit_transform(data)
print('min max')
print(data_scalered)


print('min abs')
data_scaler = preprocessing.MaxAbsScaler()
print(data_scaler.fit_transform(data))


print('normalizing data')
data_norm = preprocessing.normalize(data , norm='l1')
print(data_norm)

print('binarizing data')
data_bina = preprocessing.Binarizer(threshold=0.5).transform(data)
print(data_bina)


print('one hot encoder')
print('data for one hot encoding')
print([0,2,1,12],[1,3,5,3],[2,3,2,12],[1,2,4,3])
encoder  = preprocessing.OneHotEncoder()
encoder.fit([[0,2,1,12],[1,3,5,3],[2,3,2,12],[1,2,4,3]])
encode_vect = encoder.transform([[2,3,4,3]]).toarray()
print(encode_vect)







