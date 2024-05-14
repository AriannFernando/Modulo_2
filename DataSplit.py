import os
import shutil
import random

# Definir las rutas de las carpetas de origen y destino
origin_path = 'asl_alphabet_train'
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

# Obtener la lista de carpetas de letras
letters_directory = os.listdir(origin_path)

# Iterar sobre cada carpeta de letra
for letter_directory in letters_directory:
    # Obtener la lista de imágenes en la carpeta de letra
    letter_directory_path = os.path.join(origin_path, letter_directory)
    images = os.listdir(letter_directory_path)
    
    # Mezclar aleatoriamente las imágenes
    random.shuffle(images)
    
    # Calcular el tamaño de cada conjunto (train, test, validation)
    total_images = len(images)
    num_train = int(0.8 * total_images)
    num_test = int(0.1 * total_images)
    num_validation = total_images - num_train - num_test
    
    # Crear las carpetas de destino dentro de la carpeta de letra si no existen
    letter_train_path = os.path.join(destination_path_train, letter_directory)
    letter_test_path = os.path.join(destination_path_test, letter_directory)
    letter_validation_path = os.path.join(destination_path_validation, letter_directory)
    if not os.path.exists(letter_train_path):
        os.makedirs(letter_train_path)
    if not os.path.exists(letter_test_path):
        os.makedirs(letter_test_path)
    if not os.path.exists(letter_validation_path):
        os.makedirs(letter_validation_path)
    
    # Dividir las imágenes en conjuntos (train, test, validation)
    images_train = images[:num_train]
    images_test = images[num_train:num_train + num_test]
    images_validation = images[num_train + num_test:]
    
    # Mover las imágenes a las carpetas de destino copiandolas
    for image in images_train:
        shutil.copy(os.path.join(letter_directory_path, image), os.path.join(letter_train_path, image))
    for image in images_test:
        shutil.copy(os.path.join(letter_directory_path, image), os.path.join(letter_test_path, image))
    for image in images_validation:
        shutil.copy(os.path.join(letter_directory_path, image), os.path.join(letter_validation_path, image))
