import numpy as np

x_train = np.load('/home/daniel/Desktop/splits/splits_1/x_train_spec.npy')
hr_train = np.load('/home/daniel/Desktop/splits/splits_1/hr_train_spec.npy')
y_train = np.load('/home/daniel/Desktop/splits/splits_1/y_train_spec.npy')

x_test = np.load('/home/daniel/Desktop/splits/splits_1/x_test_spec.npy')
hr_test = np.load('/home/daniel/Desktop/splits/splits_1/hr_test_spec.npy')
y_test = np.load('/home/daniel/Desktop/splits/splits_1/y_test_spec.npy')

print x_train.shape
print hr_train.shape
print y_train.shape

print x_test.shape
print hr_test.shape
print y_test.shape