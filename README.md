# Modulo 2

Desarrollo de aplicaciones avanzadas de ciencias computacionales TC3002B

Ariann Fernando Arriaga Alcántara A01703556

En el siguiente repositorio se alojarán todos los archivos desarrollados referentes al modelo para el Módulo 2 de la materia de Desarrollo de aplicaciones avanzadas de ciencias computacionales TC3002B

Todos los avances presentados, fueron desarrollados en la siguiente liga a una carpeta de drive: [Repositorio](https://drive.google.com/drive/folders/14AKGu8siQM9iV9aBoJnX9msSJgTD-qDk?usp=sharing)


# Clasificación de emociones de perros

## Dataset
El dataset seleccionado para el desarrollo de este proyecto fue creado por Daniel Shan Balico en 2023, el cual fue obtenido en la plataforma Kaggle, donde se encontró indentificado como ["Dog Emotion Image Classification"](https://www.kaggle.com/datasets/danielshanbalico/dog-emotion/data)


El dataset contiene imagenes las cuales están categorizadas por clases referentes a las emociones que podría tener un perro. Dicho esto este dataset fue construido para poderse utilizar con machine learning y generar un clasificador referente a las emociones de un perro.

La construcción del dataset fue por medio de imagenes recopiladas de varias fuentes en línea, como Kaggle y motores de búsqueda de imágenes. Y para asegurar la representatividad correcta de los datos, se seleccionaron imágenes de una variedad de razas de perros.

La estructura de data set esta por carpetas referentes a las clases de emociones:
- Angry
- Happy
- Relaxed
- Sad

Cada una de estas carpetas contiene 1000 imágenes de perros, de diferentes razas represenentado la respectiva emoción

Para hacer la separación de los datos, decidí dividir la estructura de la carpeta original del dataset en carpetas para el training, validation y testing del modelo a construir
La separación fue hecha por medio de un script de python DataSplit.py el cual genera una división dada por porcentajes, en donde la carpeta train contiene el 70% de las imagenes del dataset y tanto la carpeta de validation y test contienen el 15% respectivamente. La división fue hecha de esa forma por lo descrito en la siguiente (fuente)[https://www.baeldung.com/cs/train-test-datasets-ratio]. Debido al tamaño del dataset, al ser menor de 10000 imágenes, se realizó la división 70%, 15% y 15%.

Para la división de las imagenes, se seleccionaron de manera aleatoria para poder respetar la representatividad de la muestra del dataset. 
Contando con una estructura final del conjunto de imagenes de la siguiente forma:

### Train
- Angry : 800 imágenes
- Happy : 800 imágenes
- Relaxed : 800 imágenes
- Sad : 800 imágenes
### Validation
- Angry : 100 imágenes
- Happy : 100 imágenes
- Relaxed : 100 imágenes
- Sad : 100 imágenes
### Test
- Angry : 100 imágenes
- Happy : 100 imágenes
- Relaxed : 100 imágenes
- Sad : 100 imágenes

## Preprocesado de los datos






