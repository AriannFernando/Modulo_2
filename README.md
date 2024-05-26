# Módulo 2

Desarrollo de aplicaciones avanzadas de ciencias computacionales TC3002B

Ariann Fernando Arriaga Alcántara A01703556

En el siguiente repositorio se alojarán todos los archivos desarrollados referentes al modelo para el Módulo 2 de la materia de Desarrollo de aplicaciones avanzadas de ciencias computacionales TC3002B

Todos los avances presentados en este repositorio se encuentran disponibles en la siguiente carpeta de Google Drive: [Módulo 2](https://drive.google.com/drive/folders/14AKGu8siQM9iV9aBoJnX9msSJgTD-qDk?usp=sharing)


# Clasificación de emociones de perros

## Dataset
El dataset seleccionado para el desarrollo de este proyecto fue creado por Daniel Shan Balico en 2023, el cual fue obtenido en la plataforma Kaggle, donde se encontró identificado como ["Dog Emotion Image Classification"](https://www.kaggle.com/datasets/danielshanbalico/dog-emotion/data)


El dataset contiene imágenes las cuales están categorizadas por clases referentes a las emociones que podría tener un perro. Dicho esto, este dataset fue construido para poderse utilizar con machine learning y generar un clasificador referente a las emociones de un perro.

La construcción del dataset fue por medio de imágenes recopiladas de varias fuentes en línea, como Kaggle y motores de búsqueda de imágenes. Y para asegurar la representatividad correcta de los datos, se seleccionaron imágenes de una variedad de razas de perros.

La estructura del dataset está por carpetas referentes a las clases de emociones:
- Angry
- Happy
- Relaxed
- Sad

Cada una de estas carpetas contiene 1000 imágenes de perros, de diferentes razas, represen entado la respectiva emoción

Para hacer la separación de los datos para poder entrenar el modelo, se dividió la estructura de la carpeta original del dataset en carpetas para el entrenamiento, validación y pruebas del modelo a construir
La separación fue hecha por medio de un script de python **DataSplit.py** el cual genera una división dada por porcentajes, en donde la carpeta **train** contiene el 70% de las imágenes del dataset y tanto la carpeta de **validation** y **test** contienen el 15% respectivamente. La división fue hecha de esa forma por lo descrito en la siguiente [fuente](https://www.baeldung.com/cs/train-test-datasets-ratio). Debido al tamaño del dataset, al ser menor de 10000 imágenes, se realizó la división 70%, 15% y 15%.

Para la división de las imágenes, se seleccionaron de manera aleatoria para poder respetar la representatividad de la muestra del dataset. 
Contando con una estructura final del conjunto de imágenes de la siguiente forma:

- **Train**
  - Angry : 700 imágenes
  - Happy : 700 imágenes
  - Relaxed : 700 imágenes
  - Sad : 700 imágenes
- **Validation**
  - Angry : 150 imágenes
  - Happy : 150 imágenes
  - Relaxed : 150 imágenes
  - Sad : 150 imágenes
- **Test**
  - Angry : 150 imágenes
  - Happy : 150 imágenes
  - Relaxed : 150 imágenes
  - Sad : 150 imágenes

## Preprocesamiento de los datos
Para poder entrenar el modelo de manera correcta se deben de preparar las imágenes para poder maximizar la posible efectividad de los resultados de este mismo. Después de realizar una investigación, K. Pal [1] describe posibles modificaciones de preprocesamiento de los datos para poder mejorar la calidad de las predicciones generadas por un modelo de clasificación de imágenes.
Para el modelo presentado se utiliza ImageDataGenerator de la librería TensorFlow, dedicada a machine learnig.
Para el prepocesamiento se realizó lo siguiente para el conjunto de **train**:
- **Normalización del valor de los píxeles:** Se utilizó la propiedad de re-escalamiento que ofrece el ImageDataGenerator, esto se realiza para que los valores de un píxel este normalizado a 1, es decir ir de 0 a 1, en vez de 255
- **Redimensionamiento de las imágenes:** Para esto se definió un redimensionamiento de las imágenes del dataset a 150*150 píxeles para mantener consistencia entre los datasets.
-   **Data augmentation:** Debido al tamaño reducido del dataset, se decidió aplicar un data augmentation, que significa el transformar las imágenes del conjunto **train** del dataset, y así poder aumentar la cantidad de imágenes diferentes que recibe el modelo para su entrenamiento. Es por eso que lo aplicado para del conjunto de “train” fueron las siguientes transformaciones:
  -   **Rotation:** Se estableció un rango posible de rotación de 180 grados.
  -   **Width shift:** Se definió que se pueda modificar la imagen en su ancho, es decir, ampliarlo hasta un 30% de la imagen original.
  -   **Height shift:** Se definió que se pueda modificar la imagen en su altura, es decir, estirarlo hasta un 30% de la imagen original.
  -   **Shear:** Se estableció un rango de cizallamiento de 0.1, lo que significa que la imagen original puede ser inclinada aleatoriamente en un rango de -0.1 a 0.1 radianes.
  -   **Zoom:** Se definió un rango de acercamiento de 0.1, lo cual es referente a que la imagen original puede ser modificada al acercar la imagen hasta un 10% de su tamaño original.
  -   **Horizontal flip:** Se estableció como verdadero este parámetro, el cual indica que se puede modificar la imagen original al rotarla completamente en su eje horizontal

Cabe mencionar que para cada modificación, cuando se generan las imágenes por medio del ImageDataGenerator se puede aplicar esta modificación de manera aleatoria, es decir, cuando se genera una imagen modificada hay una probabilidad que se aplique una modificación definida previamente. Y al analizar con detenimiento las imágenes existentes en el conjunto de **train**, se puede concluir que las modificaciones definidas previamente no implican un cambio en el significado de las imágenes originales. No se pierde su interpretación, por lo cual se pueden utilizar en el entrenamiento del modelo. Con esto se busca mejorar la calidad de las predicciones y evitar problemas como el overfitting, donde el modelo se adapta a las imágenes existentes en vez de aprender.

Otro atributo definido es el modo de clase que se aplica para la generación de imágenes, en este caso al contar con diferentes clases para la clasificación de imágenes se definió el atributo como 'categorical' para que procese las diferentes clases presentes.

Las modificaciones de preprocesamiento aplicadas para el conjunto **test** y **validation** fueron la normalización del valor de los píxeles y redimensionamiento de las imágenes, como se realizó para el conjunto **train**. Esto se realizó para que todo el conjunto de imágenes tenga cohesividad y se pueda probar de manera efectiva. Solo se aplicaron estas modificaciones para que estos conjuntos sean representativos de imágenes que se puedan encontrar en el mundo real y se pueda mostrar la efectividad del modelo con certeza.

La generación de imágenes por medio del data augmentation implica el definir un batch size, el cual es un tamaño de lote que representa el tamaño del número de imágenes nuevas que se pueden crear al llamar el generador de imágenes y que se utilizarán para entrenar el modelo. En este caso, para el preprocesador se definió un batch size de 50 para el generador de imágenes del conjunto train y 50 para los conjuntos de **test** y **validation** respectivamente. Con esto se entrenará el modelo con un conjunto de 2100 imágenes modificadas por cada vez que se llame el generador de imágenes.

El preprocesamiento de los datos descrito se encuentra realizado en la carpeta de drive [Módulo 2](https://drive.google.com/drive/folders/14AKGu8siQM9iV9aBoJnX9msSJgTD-qDk?usp=sharing), específicamente en el archivo: **DogEmotionClassificationModel_A01703556.ipynb.** En el cual se implementó un código de lo previamente descrito en un notebook de python.

Con propósitos de demostración se mandó a llamar el generador de imágenes para el conjunto train y se guardaron los resultados en la capeta **files > augmented**. Se utilizó un batch size de 50 con el objetivo de demostrar él data augmentation aplicado para el preprocesamiento de los conjuntos de datos.
Por último, se utilizaron las librerías de matplotlib y numpy para poder visualizar las imágenes creadas por el generador de imágenes con el lote previamente definido. Como se muestra a continuación:

![Imagen de referencia](files/augmented/Evidence.png)


## Implementación del modelo
Para poder implementar el modelo, se utilizó transfer learning para poder generar una red neuronal convolucional que produzca predicciones más precisas y certeras. Al realizar una investigación a profundidad de los mejores modelos basados en sus resultados para la clasificación de imagenes basadas en las emociones expresadas en rostros, lo cual es una aproximación a la temática de la clasificación de emociones de perros, M.A.H Akhand [4] describe un modelo de aprendizaje profundo basado en la arquitectura VGG-16 para la detección de emociones con los mejores resultados descritos. De acuerdo a lo presentado por Sushma, L [5], la arquitectura VGG-16 representa una variante de la red VGG net, una arquitectura de Red Neuronal Convolucional (CNN) que se utilizó para ganar el concurso ILSVRC (ImageNet) en 2014 [14].

**VGG16** es una arquitectura de Red Neuronal Convolucional (CNN) desarrollada por el Visual Geometry Group de la Universidad de Oxford. Fue utilizada para ganar el concurso ILSVRC (ImageNet) en 2014. La arquitectura VGG16 es conocida por su simplicidad y profundidad, ya que consiste en 16 capas con pesos entrenables, la razón del nombre de la arquitectura.

Características principales de VGG16:

1. **Capas Convolucionales**: Utiliza capas convolucionales con filtros de tamaño 3x3, lo que permite capturar características espaciales finas en las imágenes. Todas las convoluciones tienen un stride de 1 y el mismo padding, lo que conserva las dimensiones de las imágenes a través de las capas.

2. **Capas de Max Pooling**: Después de cada grupo de convoluciones, se utiliza una capa de max pooling con filtros de tamaño 2x2 y un stride de 2. Esto reduce las dimensiones espaciales de las características y ayuda a reducir la complejidad computacional.

3. **Capas Totalmente Conectadas**: Al final de la red, hay tres capas totalmente conectadas. Las dos primeras tienen 4096 unidades cada una y la última capa tiene un número de unidades igual al número de clases en el conjunto de datos (por ejemplo, 1000 para ImageNet).

4. **Capa de SoftMax**: La última capa de la red es una capa de SoftMax, que produce una distribución de probabilidad sobre las clases de salida.

5. **Entrada de la Red**: La red toma imágenes de entrada con dimensiones de 224x224x3 píxeles originalmente (altura, anchura, canales de color) con las modificaciones actuales se cuenta con las dimensiones 150x150x3.

Además de utilizar la arquitectura VGG-16 el modelo creado utiliza las siguientes capas con base a los artículos descritos previamente:

1. **base model**: Es la capa base del modelo secuencial, la cual es la arquitectura VGG-16 descrita previamente

2. **flatten**: es una capa de la red que aplana la entrada a las neuronas

3. **Dense**: el modelo cuenta con 2 capas densas las cuales cuentan con 1024 y 256 neuronas respectivamente, y las cuales cuentan con una función de activación "relu" la cual se activa si el valor de la función es positivo entonces toma el valor de la x, además de un regularizers lo cuales añaden un término de penalización al costo del modelo. Esto se hizo para desalentar el modelo a aprender pesos grandes. Lo cual se hace para evitar el sobreajuste

4. **Dropout**: dentro de las capas densas se encuentran las capas dropout las cuales omiten aleatoriamente algunas neuronas para evitar que el modelo se vuelva dependiente de cualquier neurona en particular.

5. **Salida**: por ultimo se genera una capa densa definida como salida con la función de activación softmax la cual genera una salida con una capa de 4 neuronas para la determinación de la clasificación basado en una función softmax

### Pérdida
Para la función de perdida se utilizó categorical crossentropy, la cual genera un calculo de perdida basado en como disminuye a medida que la probabilidad predicha converge hacia el label real. Mide el rendimiento de un modelo de clasificación cuya salida predicha es un valor de probabilidad entre las clases definidas.

### Función de optimización
La función de optimización que se utilizó es el RMSprop (Root Mean Square Propagation) el cual usa una media móvil ponderada del cuadrado de los gradientes y divide el gradiente de cada parámetro por la raíz cuadrada de esta media.

Con esto tomando en cuenta el batchsize definido previamente se determina el número de steps que se realizan por epoch, los cuales su multiplicación deberá de dar un resultado igual al total de imágenes de set de train, o test, o validation. Con una taza de aprendizaje definida de 1e-5 que define el peso de cuanto se actualiza el resultado del ajuste del aprendizaje

Para este modelo inicial se definieron 20 epochs donde se analizaran sus resultados. Todo el modelo desarrollado se encuentra en [Módulo 2](https://drive.google.com/drive/folders/14AKGu8siQM9iV9aBoJnX9msSJgTD-qDk?usp=sharing),

## Evaluación inicial del modelo

# Referencias bibliográficas

[1] K. K. Pal and K. S. Sudeep, "Preprocessing for image classification by convolutional neural networks," 2016 IEEE International Conference on Recent Trends in Electronics, Information & Communication Technology (RTEICT), Bangalore, India, 2016, pp. 1778-1781, doi: 10.1109/RTEICT.2016.7808140.

[2] Gu, Shanqing; Pednekar, Manisha; and Slater, Robert, "Improve Image Classification Using Data Augmentation and Neural Networks," 2019 SMU Data Science Review: Vol. 2: No. 2, Article 1. https://scholar.smu.edu/datasciencereview/vol2/iss2/1 

[3] A. Aylin Tokuç, “Baeldung,” Baeldung on Computer Science, Jan. 14, 2021. https://www.baeldung.com/cs/train-test-datasets-ratio.

[4] Akhand, M.A.H.; Roy, S.;Siddique, N.; Kamal, M.A.S.;Shimamura, T. Facial Emotion Recognition Using Transfer Learning in the Deep CNN. Electronics 2021, 10, 1036. https://doi.org/10.3390/electronics10091036

[5]Sushma, L., & Lakshmi, K. P. 2020. "An analysis of convolution neural network for image classification using different models". International Journal of Engineering Research and Technology (IJERT), 9(10).
‌
