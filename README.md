# Predicción de Cancelación en Hoteles
## Autor: Juan Gonzalo Martínez Rubio
En este proyecto se han desarrollado 5 modelos predictivos de clasificación binaria (Regresión Logística, Árbol de decisión, Random Forest, XGBoost y Red neuronal multicapa) para el mismo dataset.

### Datos
Nuestro dataset consta de 32 columnas y 119390 filas, cada una representando una reservas de hotel que fue cancelada o no. En total el 37% de nuestras muestras representan reservas canceladas.

### Exploratory Data Analysis (EDA)
- Eliminamos un total de 31.994 filas duplicadas
- Eliminamos la columna "company" ya que tenía un 94% de valores nulos.
- 448 nulos en la columna "country", los cuales imputamos al valor más frecuente (median) de entre los demás
- 4 nulos en la columna "Children", los cuales imputamos con la mediana (median) del resto.
- 13.69% de valores nulos en la colmna "agent", los cuales ponemos valor 0 (sin agente asociado)
- Eliminamos las columnas "reservation_status" y "reservation_status_date" ya que no nos aportan información de utilidad para predecir si habrá cancelación de reserva.

En este punto tenemos un DataFrame de 29 columnas y 87.396 filas.

- Aplicamos One Hot Encoding (pd.get_dummies) a las variables categóricas, quedando un total de 247 columnas.
- Escalamos las variables numéricas (StandardScaler) para que darles a todas la misma importancia a priori.

Finalmente separamos un 80% de los datos para entrenamiento y el 20% restante para test.

### Modelos
Para optimizar los hiperparámetros he utilizado GridSearchCV para la Regresión logística y el Árbol de decisión, RandomizedSearchCV para Random Forest y XGBoost. El modelo de red Neuronal no ha pasado por ningún proceso de optimización de hiperparámetros.

En los Pipelines (archivos .py) ha habido ligeras modificaciones para generar los modelos en archivos .joblib que después puedan servir una API REST. Loas métricas resultantes han sido las siguientes.


<table border="1" cellpadding="8" cellspacing="0">  <thead>  <tr>  <th></th>  <th>Accuracy</th>  <th>Precision</th>  <th>Recall</th>  <th>F1-Score</th>  <th>AUC</th>  </tr>  </thead>  <tbody>  <tr>  <td>Regresión Logística</td>  <td>0.75</td>  <td>0.53</td>  <td>0.79</td>  <td>0.63</td>  <td><b>0.84</b></td>  </tr>  <tr>  <td>Árbol de decisión</td>  <td>0.81</td>  <td>0.69</td>  <td>0.60</td>  <td><b>0.64</b></td>  <td>0.87</td>  </tr>  <tr>  <td>Random Forest</td>  <td>0.85</td>  <td>0.76</td>  <td>0.65</td>  <td>0.70</td>  <td><b>0.91</b></td>  </tr>  <tr>  <td>XGBoost</td>  <td>0.85</td>  <td>0.78</td>  <td>0.62</td>  <td>0.69</td>  <td><b>0.92</b></td>  </tr>  <tr>  <td>Red Neuronal</td>  <td>0.81</td>  <td>0.62</td>  <td>0.83</td>  <td>0.71</td>  <td><b>0.90</b></td>  </tr>  </tbody>  </table>

Según el algoritmo he seleccionado la métrica principal de calidad:

- **Regresión Logística: AUC (0.84)**
Entrega probabilidades bien calibradas y la AUC mide su capacidad de etiquetar correctamente positivos y negativos sin fijar umbral, siendo la métrica que mejor aprovecha la salida continua del modelo.

- **Árbol de Decisión: F1-Score (0.64)**
Un solo árbol de decisión suele dar predicciones directas y es sensible al sobreajuste en escenarios de desbalanceo de clases. El F1-Score penaliza de forma equilibrada los falsos positivos y falsos negativos, de mejor manera que el resto de métricas.

- **Random Forest: AUC (0.91)**
Al promediar muchos árboles, produce probabilidades más estables. La AUC captura esa capacidad de discriminación global sin depender del umbral y es robusta a cambios en la proporción de clases.

- **XGBoost: AUC (0.92)**
El "Gradiente Boosting" Funciona como Random Forest pero cada árbol intenta corregir los errores más graves del anterior, optimizando "log-loss" 

- **Red Neuronal Multicapa: AUC (0.90)**
Al entrenarse con "cross-entropy" entrega probabilidades contínuas y AUC las evalúa de forma coherente independiente del umbral e inmune al desbalanceo de clases.

En los Notebooks se puede ver tanto la matriz de confusión como la curva ROC de todos los algoritmos.

### Automatización y estructura del sistema
1. **data_loader**: Carga el dataset y valida su contenido
2. **preprocessing**: Limpia el dataset como explicamos en el EDA. Posteriormente separa las columnas numéricas y construye un preprocesador, el cual aplica un escalado numérico a las variables numéricas y One Hot Encoding a las columnas categóricas.
3. **models**: Definimos los 5 modelos
4. **tuner**: Optimización de hiperparámetros mediante GridSearchCV para regresión logística y árbol de decisión y, RandomizedSearchCV para Random Forest y XGBoost. La red neuronal no se ha sometido a un proceso de estos ya que me daban errores contínuamente.
5. **trainer**:  Define el flujo total de la automatización usando funciones de los archivos anteriores para finalmente almacenar los modelos en archivos con extensión .joblib, junto a un archivo JSON que muestra las métricas resultantes y los mejores hiperparámetros calculados para los distintos modelos.

### Experimento en MLFlow
Estos 5 modelos los he subido a MLFlow , donde se han calculado las anteriores métricas. Esta plataforma me ha permitido compararlos entre sí auqneu también ofrece otras funcionalidades como control de versiones de modelos, preprocesamiento de datos o facilitar el despliegue de los mismos. 
En este proyecto ha sido una prueba sencilla para conocer la herramienta y ver todas las posibilidades que ofrece.

### API REST (Fast API)
He desarrollado una API REST que me permite ejecutar predicciones con un modelo específico o con todos a la vez. Esta API es la que servirá los modelos a nuestra UI, permitiendo ejecutar predicciones directamente desde la interfaz de usuario.

### Interfaz de Usuario (Streamlit)
He desarrollado una aplicación web con Streamlit, la cual consta de 3 páginas:

- Dashboard del dataset original. Aquí hay medidas y gráficas sobre los datos originales, pudiendo ver los mismos tanto en valor absoluto como en porcentaje, así como una previsualización del archvio CSV a modo de tabla en la parte inferior. Incluye filtrado por meses, años u hotel seleccionado.

- Comparación de modelos. En esta página se puede ver de forma gráfica comparativa todas las métricas de calidad de los diferentes modelos, así como la información específica de cada modelo en la parte derecha. Estos datos se obtienen del archivo JSON mencionado anteriormente en este documento.

- Predicciones. En esta página tenemos un formulario con 15 de las variables más importantes a la hora de hacer predicciones y, gracias a los endpoints de la API, podemos hacer predicciones tanto de 1 solo modelo como de todos a la vez y comparar resultados.

## Ejecución del proyecto

Es importante tener el CSV con los datos en una carpeta /data/. De no ser así, no se mostrarán los datos en la UI

- Para lanzar la automatización que crea los modelos en archivos .joblib
```
python automatization/trainer.py --data <ruta al csv>
```

- Lanzar el experimento en MLFlow y mostrarlo en el puerto 500 de tu localhost
```
python mlflow_eval.py --data <ruta al csv>
mlflow ui --port 5000
```

- Lanzar la API REST
```
uvicorn api:app --reload
```

- lanzar la UI de Streamlit (Si la API no está activa, la página de predicciones fallará)
```
streamlit run ui/app.py
```