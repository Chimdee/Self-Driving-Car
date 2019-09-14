def model():
    import numpy as np
    import csv
    import cv2
    import matplotlib.image as mpimg
    from keras.layers.core import Dense, Flatten, Lambda
    from keras.layers import Conv2D, AveragePooling2D, Cropping2D
    from keras.models import Sequential
    from keras import callbacks
    from keras.preprocessing.image import ImageDataGenerator
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    #from keras.applications

    #Data preparation dataset 1
    lines=[]
    with open('data1/driving_log1.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

    images = []
    measurements = []
    for line in lines[1:]:
        source_path= line[0]
        filename =  source_path.split('\\')[-1]
        image_path = 'data1/IMG/' + filename
        img = mpimg.imread(image_path)
        images.append(img)
        measurements.append(float(line[3]))

    # Dataset 2: driving counter clock wise and driving on the track 2
    lines2 = []
    with open('data2/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines2.append(line)

    images2 = []
    measurements2 = []
    for line in lines2[1:]:
        source_path= line[0]
        filename =  source_path.split('\\')[-1]
        image_path = 'data2/IMG/' + filename
        img = mpimg.imread(image_path)
        images2.append(img)
        measurements2.append(float(line[3]))

    # Merging two datasets
    #lines.extend(lines2)
    images.extend(images2)
    measurements.extend(measurements2)

    print('\nData loaded. NUmber of datapoints: {}, {}, {}'.format(len(lines), len(images), len(measurements)))

    #Data augmentation
    aug_images , aug_measurements = [], []
    for img, mea in zip(images, measurements):
        aug_images.append(img)
        aug_measurements.append(mea)
        aug_images.append(cv2.flip(img, 1))
        aug_measurements.append(mea*-1.0)

    X_data = np.array(aug_images)    
    y_data = np.array(aug_measurements)
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size = .2, random_state = 42)
    print('\nData augmented\n')

    #Generator
    datagen = ImageDataGenerator()
    validgen = ImageDataGenerator()

    #Model Architecture
    model = Sequential()
    model.add(Cropping2D(cropping=((70, 25), (0,0)), input_shape=(160, 320, 3)))
    model.add(Lambda(lambda x: x/255.0 - 0.5))
    model.add(Conv2D(24,(5,5), strides = (2,2), activation='relu'))
    #model.add(AveragePooling2D())
    model.add(Conv2D(36, (5,5), strides = (2,2), activation='relu'))
    model.add(Conv2D(48, (5,5), strides = (2,2), activation='relu'))
    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(Conv2D(64, (3,3), activation='relu'))
    #model.add(AveragePooling2D())
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))       
    model.add(Dense(1))

    model.summary()
    model.compile(optimizer='adam', loss='mse')

    #Training
    print('Training starts...')
    b_size = 32
    epoch = 20
    call_back_list = [callbacks.EarlyStopping(monitor='val_loss', patience = 50),
                 callbacks.ModelCheckpoint(filepath = 'model2.h5', monitor = 'val_loss', save_best_only = True)
                 ]
    history_object = model.fit_generator(datagen.flow( X_train, y_train, batch_size=b_size), 
                                         steps_per_epoch= len(X_train)/b_size, epochs = epoch, verbose=1,
                                         validation_data=validgen.flow( X_test, y_test, batch_size=b_size),
                                         validation_steps = len(X_test)/b_size, callbacks = call_back_list)

    #Saving model ckpt
    model.save('model2.h5')

    #loss visualization
    ### print the keys contained in the history object
    print(history_object.history.keys())
    
    return 

model()

### plot the training and validation loss for each epoch
# plt.plot(history_object.history['loss'])
# plt.plot(history_object.history['val_loss'])
# plt.title('model mean squared error loss')
# plt.ylabel('mean squared error loss')
# plt.xlabel('epoch')
# plt.legend(['training set', 'validation set'], loc='upper right')
# plt.show()

