from learningRate import *

# data generator training
train_datagen = ImageDataGenerator(rotation_range=15,rescale=1./255,shear_range=0.1, zoom_range=0.2, horizontal_flip=True, width_shift_range=0.1, height_shift_range=0.1)
train_generator = train_datagen.flow_from_dataframe(train_df, "./train/",x_col='filename',y_col='category', target_size=Image_Size,class_mode='categorical',batch_size=batch_size)

# validation data generator
validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_dataframe(validate_df, "./train/", x_col='filename', y_col='category', target_size=Image_Size, class_mode='categorical', batch_size=batch_size)

# test data generator
test_datagen = ImageDataGenerator(rotation_range=15, rescale=1./255, shear_range=0.1, zoom_range=0.2,horizontal_flip=True, width_shift_range=0.1, height_shift_range=0.1)
test_generator = train_datagen.flow_from_dataframe(train_df, "./test1/",x_col='filename',y_col='category',  target_size=Image_Size,  class_mode='categorical', batch_size=batch_size)

# model training
epochs=10
history = model.fit_generator(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=total_validate//batch_size,
    steps_per_epoch=total_train//batch_size,
    callbacks=callbacks
)

model.save("model_cat_dogs/model.h5")