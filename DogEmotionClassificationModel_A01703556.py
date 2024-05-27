import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers, models, layers
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.regularizers import L2

#Ariann Fernando Arriaga Alcántara A01703556

# Directorios de los datos
train_dir = 'train'
test_dir = 'test'
validation_dir = 'validation'

# Generadores de datos con ImageDataGenerator
train_datagen = ImageDataGenerator(
    rescale=1./255,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.1,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)

# Generadores de datos
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=16,  # Tamaño del batch reducido
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=16,  # Tamaño del batch reducido
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=16,  # Tamaño del batch reducido
    class_mode='categorical'
)

# Modelo base VGG16
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense

base_model = VGG16(weights="imagenet", include_top=False, input_shape=(150, 150, 3))
base_model.trainable = False

# Capas densas con regularización L2
dense_layer_1 = Dense(256, activation='relu', kernel_regularizer=L2(0.00001))
dense_layer_2 = Dense(128, activation='relu', kernel_regularizer=L2(0.00001))

# Definición del modelo secuencial
model = models.Sequential()
model.add(base_model)
model.add(layers.Flatten())
model.add(dense_layer_1)
model.add(layers.Dropout(0.3))
model.add(dense_layer_2)
model.add(layers.Dropout(0.3))
model.add(layers.Dense(3, activation='softmax'))

model.summary()

# Compilación del modelo
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(learning_rate=1e-5),
              metrics=['acc'])

# Entrenamiento del modelo
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=81,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size
)

# Visualización de la precisión y pérdida
acc = history.history['acc']
loss = history.history['loss']
validation_acc = history.history['val_acc']
validation_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'b', label='Train Accuracy')
plt.plot(epochs, validation_acc, 'r', label='Validation Accuracy')
plt.title('Train vs Validation Accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'b', label='Train Loss')
plt.plot(epochs, validation_loss, 'r', label='Validation Loss')
plt.title('Train vs Validation Loss')
plt.legend()

plt.show()

# Evaluación del modelo en el conjunto de prueba
test_loss_vgg_16, test_acc_vgg_16 = model.evaluate(test_generator, steps=test_generator.samples // test_generator.batch_size)
print('\nTest Accuracy:\n', test_acc_vgg_16)
print('\nTest Loss:\n', test_loss_vgg_16)

# Predicciones y matriz de confusión
test_generator.reset()
Y_pred = model.predict(test_generator)
y_pred = np.argmax(Y_pred, axis=1)

# Matriz de confusión
cm = confusion_matrix(test_generator.classes, y_pred)

# Reporte de clasificación
target_names = list(test_generator.class_indices.keys())
print('Classification Report')
print(classification_report(test_generator.classes, y_pred, target_names=target_names))

# Visualización de la matriz de confusión
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()

# Guardado del modelo
model.save('modelclassification.h5')
model.save('modelclassification.keras')