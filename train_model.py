import tensorflow as tf

train_dataset_path = 'melanoma_cancer_dataset/train'
test_dataset_path = 'melanoma_cancer_dataset/test'

train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    train_dataset_path, image_size=(224, 224), batch_size=32,
    label_mode='int', )

test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    test_dataset_path, image_size=(224, 224), batch_size=32,
    label_mode='int', )

# print type of classes, should be two
class_names = train_dataset.class_names
print(class_names)

model = tf.keras.models.Sequential([
    # changing the 3 channels to be 0-1 values
    tf.keras.layers.Rescaling(1. / 255, input_shape=(224, 224, 3)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),

    tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(len(class_names), activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_dataset, validation_data=test_dataset, epochs=10)

model.save('tf_model.keras')
