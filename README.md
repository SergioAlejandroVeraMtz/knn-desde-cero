# KNN desde cero

Implementación del algoritmo de los k vecinos más cercanos (KNN) escrita a mano en Python, con una interfaz gráfica en Tkinter para cargar datasets, configurar el valor de k y comparar resultados con validación cruzada.

Práctica 5 de Inteligencia Artificial, ESCOM-IPN.

## Qué incluye

- **Clasificador KNN propio** (`KNNFromScratch`): calcula distancias, selecciona los k vecinos y vota por la clase mayoritaria.
- **Validación cruzada de 10 pliegues** y **leave-one-out**, con accuracy, sensibilidad, especificidad y accuracy balanceada.
- **Carga de datasets** en formato `.arff`, `.csv` y `.data` (formato UCI).
- **Interfaz gráfica** para cargar hasta dos datasets, configurar k y ver resultados y gráficas.

## Aclaración

El algoritmo KNN está implementado desde cero. Se usan librerías de scikit-learn solo para utilidades: `KFold`, `StandardScaler` y `LabelEncoder`.

## Requisitos

- Python 3.9 o superior, con soporte para Tkinter.
- Dependencias listadas en `requirements.txt`.

```bash
python -m venv venv
source venv/bin/activate        # En Windows: venv\Scripts\activate
pip install -r requirements.txt
python knn_desde_cero.py
```

## Datasets

Los datasets no se incluyen en el repositorio. Para reproducir los resultados, descarga Iris y Wine del [repositorio UCI Machine Learning](https://archive.ics.uci.edu/) y cárgalos desde la interfaz.

## Resultados

Accuracy con validación cruzada de 10 pliegues (LOO entre paréntesis):

| k | Iris | Wine |
|---|------|------|
| 1 | 0.953 (0.947) | 0.944 (0.955) |
| 3 | 0.960 (0.947) | 0.949 (0.955) |
| 5 | 0.953 (0.947) | 0.972 (0.972) |
| 7 | 0.953 (0.960) | 0.966 (0.966) |
| 9 | 0.953 (0.953) | 0.967 (0.966) |
| 11 | 0.947 (0.953) | 0.978 (0.978) |

Los resultados completos están en la carpeta `resultados/`.

## Autor

Sergio Alejandro Vera Martínez, ESCOM-IPN.
