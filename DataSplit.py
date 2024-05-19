import os
import shutil
import random

# Definir las rutas de las carpetas de origen y destino
origin_path = 'dog_emotion'
destination_path_train = 'train'
destination_path_test = 'test'
destination_path_validation = 'validation'

# Crear las carpetas de destino si no existen
if not os.path.exists(destination_path_train):
    os.makedirs(destination_path_train)
if not os.path.exists(destination_path_test):
    os.makedirs(destination_path_test)
if not os.path.exists(destination_path_validation):
    os.makedirs(destination_path_validation)

# Obtener la lista de carpetas de dentro del la carpeta principal (las clases)
classes_directory = os.listdir(origin_path)

# Iterar sobre cada carpeta (clase)
for class_directory in classes_directory:
    # Obtener la lista de imágenes en la clase
    class_directory_path = os.path.join(origin_path, class_directory)
    images = os.listdir(class_directory_path)
    
    # Mezclar aleatoriamente las imágenes
    random.shuffle(images)
    
    # Calcular el tamaño de cada conjunto (train, test, validation)
    total_images = len(images)
    num_train = int(0.7 * total_images)
    num_test = int(0.15 * total_images)
    num_validation = total_images - num_train - num_test
    
    # Crear las carpetas de destino dentro de la carpeta clase si no existen
    class_train_path = os.path.join(destination_path_train, class_directory)
    class_test_path = os.path.join(destination_path_test, class_directory)
    class_validation_path = os.path.join(destination_path_validation, class_directory)
    if not os.path.exists(class_train_path):
        os.makedirs(class_train_path)
    if not os.path.exists(class_test_path):
        os.makedirs(class_test_path)
    if not os.path.exists(class_validation_path):
        os.makedirs(class_validation_path)
    
    # Dividir las imágenes en conjuntos (train, test, validation)
    images_train = images[:num_train]
    images_test = images[num_train:num_train + num_test]
    images_validation = images[num_train + num_test:]
    
    # Mover las imágenes a las carpetas de destino copiandolas
    for image in images_train:
        shutil.copy(os.path.join(class_directory_path, image), os.path.join(class_train_path, image))
    for image in images_test:
        shutil.copy(os.path.join(class_directory_path, image), os.path.join(class_test_path, image))
    for image in images_validation:
        shutil.copy(os.path.join(class_directory_path, image), os.path.join(class_validation_path, image))
