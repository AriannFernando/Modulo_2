# Modulo 2

Desarrollo de aplicaciones avanzadas de ciencias computacionales TC3002B

Ariann Fernando Arriaga Alcántara A01703556

En el siguiente repositorio se alojarán todos los archivos desarrollados para el Módulo 2 de la materia de Desarrollo de aplicaciones avanzadas de ciencias computacionales TC3002B

Todos los avances presentados, fueron desarrollados en la siguiente liga a una carpeta de drive: [Repositorio](https://drive.google.com/drive/folders/14AKGu8siQM9iV9aBoJnX9msSJgTD-qDk?usp=sharing)


# Clasificación de emociones de perros

## Dataset
El dataset seleccionado para el desarrollo de este proyecto fue creado por " ", el cual fue obtenido en la plataforma Kaggle, donde se encontró indentificado como ["Dog Emotion Image Classification"](https://www.kaggle.com/datasets/danielshanbalico/dog-emotion/data)


El dataset contiene imagenes las cuales están categorizadas por clases referentes a las emociones que podría tener un perro. Dicho esto este dataset fue construido para poderse utilizar con machine learning y generar un clasificador referente a las emociones de un perro.

La construcción del dataset fue por medio de imagenes referentes de 

La estructura de data set esta por carpetas referentes a las clases de:
- Enojo
- Felicidad
- Tristeza
- Relajación

Cada una de estas carpetas contiene 1000 imagenes de perros efectuando la respectiva emoción

Para hacer la separación de los datos, decidí dividir la estructura de la carpeta original del dataset en carpetas para el training, validation y testing del modelo a construir
La separación fue hecha por medio de un script DataSplit.py el cual genera una división dada por porcentajes, en donde la carpeta train contiene el 80% de las imagenes del dataset y tanto la carpeta de validation y test contienen el 10% respectivamente
Contando con una estructura final del conjunto de imagenes de la siguiente forma:


### Train
- Enojo : 800 imágenes
- Felicidad : 800 imágenes
- Tristeza : 800 imágenes
- Relajación : 800 imágenes
### Validation
- Enojo : 100 imágenes
- Felicidad : 100 imágenes
- Tristeza : 100 imágenes
- Relajación : 100 imágenes
### Test
- Enojo : 100 imágenes
- Felicidad : 100 imágenes
- Tristeza : 100 imágenes
- Relajación : 100 imágenes




