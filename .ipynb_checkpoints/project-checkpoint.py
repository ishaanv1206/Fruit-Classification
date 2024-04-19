import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.preprocessing import image
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 100,
                                                 class_mode = 'sparse')
test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 100,
                                            class_mode = 'sparse')
cnn = tf.keras.models.Sequential()
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
cnn.add(tf.keras.layers.Flatten())
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
cnn.add(tf.keras.layers.Dense(units=5, activation='softmax'))
cnn.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
cnn.fit(x = training_set, validation_data = test_set, epochs = 25)



import cv2
import numpy as np
from keras.preprocessing import image

cap = cv2.VideoCapture(0)
cap.set(3, 640)  # set Width
cap.set(4, 480)  # set Height

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Process the frame for prediction
    img = cv2.resize(frame, (64, 64))  # Resize image to match the model input size
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = img / 255.0  # Normalize pixel values
    
    # Make prediction
    test_image = image.load_img('dataset/banana.png', target_size = (64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = cnn.predict(test_image)
    if result[0][0] == 0:
        prediction = 'Apple'
    elif result[0][0] == 1:
        prediction = 'Banana'
    elif result[0][0] == 2:
        prediction = 'Grape'
    elif result[0][0] == 3:
        prediction = 'Mango'
    elif result[0][0] == 4:
        prediction = 'Strawberry'
    else:
        prediction = 'Unidentified'
    
    # Display prediction
    cv2.putText(frame, "Predicted class: " + prediction, (30, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (100, 200, 56), 2)
    cv2.imshow("Video Output", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

#{'Apple': 0, 'Banana': 1, 'Grape': 2, 'Mango': 3, 'Strawberry': 4}