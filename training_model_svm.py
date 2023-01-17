from model_svm import *
from training_and_validation_svm import *

cnn.compile(optimizer = 'adam', loss = 'hinge', metrics = ['accuracy'])
r=cnn.fit(x = training_set, validation_data = test_set, epochs = 15)

cnn.save('/model_cat_dogs/model_svm.h5')