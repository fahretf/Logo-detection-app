!wget https://bitbucket.org/ffijuljani1/logo-recognition/raw/50f2d59f58b6e75de9f39533befd6aa1f01273f0/Dataset.zip
!unzip Dataset.zip
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

model = tf.keras.Sequential(
    [
        tf.keras.layers.Rescaling(1./255.),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(20)

    ]
)
model.compile(
    optimizer='adam',
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits = True),
    metrics=['accuracy']
)
model.fit(
    train_ds,
    validation_data = val_ds,
    epochs = 15
)
model.evaluate(test_ds)
