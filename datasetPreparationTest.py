from training_and_validation_model import *

# test on our data set
test_filenames = os.listdir("./test1")
test_df = pd.DataFrame({
    'filename': test_filenames
})
nb_samples = test_df.shape[0]

predict = model.predict_generator(test_generator, steps=np.ceil(nb_samples/batch_size))
print(predict)
