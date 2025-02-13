import tensorflow as tf
import matplotlib.pyplot as plt
img_height, img_width= 256, 256
batch_size=20

train_ds = tf.keras.utils.image_dataset_from_directory(
    "Dataset/train",
    image_size=(img_height, img_width),
    batch_size=batch_size,
    shuffle=True  
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    "Dataset/validation",
    image_size=(img_height, img_width),
    batch_size=batch_size,
    shuffle=True 
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    "Dataset/test",
    image_size=(img_height, img_width),
    batch_size=batch_size,
    shuffle=True  
)
class_names = ["apple", "asus", "barcelona", "chipsy", "cocacola", "dell", "fanta", "google", "gorenje", "hp", "lenovo", "lg", "microsoft", "nike", "niveamen", "pepsi", "philips", "redbull", "samsung", "sebamed"]
plt.figure(figsize=(10,10))
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3,3,i+1)
    plt.imshow(images[i].numpy().astype('uint8'))
    plt.title(class_names[labels[i]])
    plt.axis("off")

model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=input_shape),
    tf.keras.layers.Rescaling(1./255),

    tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)
model.fit(
    train_ds,
    validation_data = val_ds,
    epochs = 15
)
model.evaluate(test_ds)
