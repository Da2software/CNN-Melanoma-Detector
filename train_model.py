import tensorflow as tf
import sys

epochs = 10
if len(sys.argv) > 1 and sys.argv[1]:
    epochs = int(sys.argv[1])

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

# but got more accuracy
model = tf.keras.models.Sequential([
    # changing the 3 channels to be 0-1 values
    tf.keras.layers.Rescaling(1. / 255, input_shape=(224, 224, 3)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(class_names), activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_dataset, validation_data=test_dataset, epochs=epochs)

model.save('tf_model.keras')
