"""
PRÁCTICA 5 - KNN DESDE CERO CON INTERFAZ GRÁFICA
Interfaz completa para cargar archivos, configurar KNN y visualizar resultados.
Autor: Sergio Alejandro Vera Martínez
Fecha: Diciembre 2025
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
from collections import Counter
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
import threading
from datetime import datetime

# ============================================================================
# 1. IMPLEMENTACIÓN DE KNN DESDE CERO
# ============================================================================

class KNNFromScratch:
    """Implementación del algoritmo K-Nearest Neighbors desde cero."""
    
    def __init__(self, k=3, distance_metric='euclidean', weights='uniform'):
        self.k = k
        self.distance_metric = distance_metric
        self.weights = weights
        self.X_train = None
        self.y_train = None
        
    def _calculate_distance(self, x1, x2):
        """Calcula la distancia entre dos puntos."""
        if self.distance_metric == 'euclidean':
            return np.sqrt(np.sum((x1 - x2) ** 2))
        elif self.distance_metric == 'manhattan':
            return np.sum(np.abs(x1 - x2))
        elif self.distance_metric == 'minkowski':
            return np.sum(np.abs(x1 - x2) ** 3) ** (1/3)
        else:
            raise ValueError(f"Métrica desconocida: {self.distance_metric}")
    
    def fit(self, X, y):
        """Almacena los datos de entrenamiento."""
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        return self
    
    def predict(self, X):
        """Predice las clases para múltiples instancias."""
        predictions = []
        for x in X:
            distances = [self._calculate_distance(x, x_train) 
                        for x_train in self.X_train]
            k_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = [self.y_train[i] for i in k_indices]
            
            if self.weights == 'uniform':
                most_common = Counter(k_nearest_labels).most_common(1)
                predictions.append(most_common[0][0])
            else:
                k_distances = [distances[i] for i in k_indices]
                weights = 1 / (np.array(k_distances) + 1e-10)
                weight_by_class = {}
                for label, weight in zip(k_nearest_labels, weights):
                    weight_by_class[label] = weight_by_class.get(label, 0) + weight
                predictions.append(max(weight_by_class, key=weight_by_class.get))
        
        return np.array(predictions)
    
    def score(self, X, y):
        """Calcula la exactitud."""
        predictions = self.predict(X)
        return np.mean(predictions == y)

# ============================================================================
# 2. FUNCIONES DE CARGA DE ARCHIVOS
# ============================================================================

def load_arff_file(filepath):
    """Carga un archivo .arff."""
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        data_start = False
        data_lines = []
        attributes = []
        
        for line in lines:
            line = line.strip()
            if line.lower().startswith('@attribute'):
                # Extraer nombre del atributo
                parts = line.split()
                if len(parts) >= 2:
                    attr_name = parts[1]
                    attributes.append(attr_name)
            elif line.lower().startswith('@data'):
                data_start = True
                continue
            elif data_start and line and not line.startswith('%'):
                data_lines.append(line)
        
        # Parsear datos
        data = []
        for line in data_lines:
            parts = line.split(',')
            row = []
            for i, part in enumerate(parts):
                part = part.strip().strip("'\"")
                try:
                    row.append(float(part))
                except ValueError:
                    row.append(part)
            data.append(row)
        
        # Convertir a DataFrame
        if attributes:
            # Usar los nombres de atributos encontrados
            df = pd.DataFrame(data, columns=attributes[:len(data[0])])
        else:
            # Usar nombres genéricos
            df = pd.DataFrame(data, columns=[f'Attr_{i}' for i in range(len(data[0]))])
        
        # Separar características y etiquetas (asumir última columna es clase)
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        
        # Convertir etiquetas a numéricas si es necesario
        if isinstance(y[0], str):
            le = LabelEncoder()
            y = le.fit_transform(y)
            classes = le.classes_
        else:
            classes = np.unique(y)
        
        return X, y, df.columns.tolist(), classes.tolist(), df
        
    except Exception as e:
        raise ValueError(f"Error al cargar archivo ARFF: {str(e)}")

def load_data_file(filepath):
    """Carga un archivo .data."""
    try:
        # Intentar leer como CSV
        df = pd.read_csv(filepath, header=None)
        
        # Asumir que la primera columna es la clase (común en .data de UCI)
        X = df.iloc[:, 1:].values
        y = df.iloc[:, 0].values
        
        # Convertir etiquetas a numéricas si es necesario
        if isinstance(y[0], str):
            le = LabelEncoder()
            y = le.fit_transform(y)
            classes = le.classes_
        else:
            classes = np.unique(y)
        
        # Crear nombres de columnas
        feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
        
        return X, y, feature_names, classes.tolist(), df
        
    except Exception as e:
        raise ValueError(f"Error al cargar archivo .data: {str(e)}")

def load_csv_file(filepath):
    """Carga un archivo .csv."""
    try:
        df = pd.read_csv(filepath)
        
        # Asumir que la última columna es la clase
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        
        # Convertir etiquetas a numéricas si es necesario
        if isinstance(y[0], str):
            le = LabelEncoder()
            y = le.fit_transform(y)
            classes = le.classes_
        else:
            classes = np.unique(y)
        
        return X, y, df.columns.tolist()[:-1], classes.tolist(), df
        
    except Exception as e:
        raise ValueError(f"Error al cargar archivo CSV: {str(e)}")

def load_dataset(filepath):
    """Carga un dataset según su extensión."""
    ext = os.path.splitext(filepath)[1].lower()
    
    if ext == '.arff':
        return load_arff_file(filepath)
    elif ext == '.data':
        return load_data_file(filepath)
    elif ext == '.csv':
        return load_csv_file(filepath)
    else:
        # Intentar cargar como CSV por defecto
        try:
            return load_csv_file(filepath)
        except:
            raise ValueError(f"Formato de archivo no soportado: {ext}")

# ============================================================================
# 3. FUNCIONES DE VALIDACIÓN Y MÉTRICAS
# ============================================================================

def k_fold_cross_validation(X, y, k_folds=10, k_neighbors=3, metric='euclidean'):
    """Implementa k-fold cross validation."""
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    accuracies = []
    sensitivities = []
    specificities = []
    
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        knn = KNNFromScratch(k=k_neighbors, distance_metric=metric)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        
        # Calcular métricas
        accuracy = np.mean(y_pred == y_test)
        accuracies.append(accuracy)
        
        # Calcular sensibilidad y especificidad por clase
        classes = np.unique(y_test)
        for cls in classes:
            tp = np.sum((y_test == cls) & (y_pred == cls))
            fn = np.sum((y_test == cls) & (y_pred != cls))
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            tn = np.sum((y_test != cls) & (y_pred != cls))
            fp = np.sum((y_test != cls) & (y_pred == cls))
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            sensitivities.append(sensitivity)
            specificities.append(specificity)
    
    avg_accuracy = np.mean(accuracies)
    avg_sensitivity = np.mean(sensitivities)
    avg_specificity = np.mean(specificities)
    balanced_accuracy = (avg_sensitivity + avg_specificity) / 2
    
    return {
        'accuracy': avg_accuracy,
        'sensitivity': avg_sensitivity,
        'specificity': avg_specificity,
        'balanced_accuracy': balanced_accuracy,
        'accuracy_std': np.std(accuracies),
        'sensitivity_std': np.std(sensitivities),
        'specificity_std': np.std(specificities)
    }

def leave_one_out_cross_validation(X, y, k_neighbors=3, metric='euclidean'):
    """Implementa Leave-One-Out Cross Validation."""
    n_samples = len(X)
    correct_predictions = 0
    sensitivities = []
    specificities = []
    
    for i in range(n_samples):
        X_train = np.delete(X, i, axis=0)
        y_train = np.delete(y, i, axis=0)
        X_test = X[i:i+1]
        y_test = y[i:i+1]
        
        knn = KNNFromScratch(k=k_neighbors, distance_metric=metric)
        knn.fit(X_train, y_train)
        prediction = knn.predict(X_test)[0]
        
        if prediction == y_test[0]:
            correct_predictions += 1
        
        # Calcular métricas
        cls = y_test[0]
        y_pred_all = knn.predict(X_train)
        
        tp = np.sum((y_train == cls) & (y_pred_all == cls))
        fn = np.sum((y_train == cls) & (y_pred_all != cls))
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        tn = np.sum((y_train != cls) & (y_pred_all != cls))
        fp = np.sum((y_train != cls) & (y_pred_all == cls))
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        sensitivities.append(sensitivity)
        specificities.append(specificity)
    
    accuracy = correct_predictions / n_samples
    avg_sensitivity = np.mean(sensitivities)
    avg_specificity = np.mean(specificities)
    balanced_accuracy = (avg_sensitivity + avg_specificity) / 2
    
    return {
        'accuracy': accuracy,
        'sensitivity': avg_sensitivity,
        'specificity': avg_specificity,
        'balanced_accuracy': balanced_accuracy
    }

# ============================================================================
# 4. INTERFAZ GRÁFICA
# ============================================================================

class KNNGUI:
    """Interfaz gráfica para KNN desde cero."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Práctica 5 - KNN desde Cero")
        self.root.geometry("1200x800")
        
        # Variables
        self.current_dataset = None
        self.datasets = {}
        self.results_history = []
        
        # Configurar estilo
        self.setup_styles()
        
        # Crear interfaz
        self.create_widgets()
        
    def setup_styles(self):
        """Configura los estilos de la interfaz."""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configurar colores
        self.bg_color = '#f0f0f0'
        self.frame_bg = '#ffffff'
        self.button_color = '#4a90e2'
        self.button_hover = '#3a7bc8'
        
        self.root.configure(bg=self.bg_color)
        
    def create_widgets(self):
        """Crea todos los widgets de la interfaz."""
        # Frame principal
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configurar grid
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(3, weight=1)
        
        # Título
        title_label = tk.Label(main_frame, 
                              text="PRÁCTICA 5: K-NEAREST NEIGHBORS DESDE CERO",
                              font=('Arial', 16, 'bold'),
                              bg=self.bg_color)
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # ========== SECCIÓN 1: CARGAR DATASETS ==========
        load_frame = ttk.LabelFrame(main_frame, text="Cargar Datasets", padding="10")
        load_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        load_frame.columnconfigure(1, weight=1)
        
        # Dataset 1
        ttk.Label(load_frame, text="Dataset 1:").grid(row=0, column=0, padx=(0, 5), sticky=tk.W)
        self.dataset1_entry = ttk.Entry(load_frame, width=40)
        self.dataset1_entry.grid(row=0, column=1, padx=(0, 5), sticky=(tk.W, tk.E))
        ttk.Button(load_frame, text="Cargar", 
                  command=lambda: self.load_dataset(1)).grid(row=0, column=2, padx=(5, 0))
        
        # Dataset 2
        ttk.Label(load_frame, text="Dataset 2:").grid(row=1, column=0, padx=(0, 5), pady=(10, 0), sticky=tk.W)
        self.dataset2_entry = ttk.Entry(load_frame, width=40)
        self.dataset2_entry.grid(row=1, column=1, padx=(0, 5), pady=(10, 0), sticky=(tk.W, tk.E))
        ttk.Button(load_frame, text="Cargar", 
                  command=lambda: self.load_dataset(2)).grid(row=1, column=2, padx=(5, 0), pady=(10, 0))
        
        # ========== SECCIÓN 2: CONFIGURACIÓN KNN ==========
        config_frame = ttk.LabelFrame(main_frame, text="Configuración KNN", padding="10")
        config_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Selección de dataset
        ttk.Label(config_frame, text="Dataset activo:").grid(row=0, column=0, padx=(0, 5), sticky=tk.W)
        self.dataset_var = tk.StringVar()
        self.dataset_combo = ttk.Combobox(config_frame, textvariable=self.dataset_var, state="readonly", width=20)
        self.dataset_combo.grid(row=0, column=1, padx=(0, 20), sticky=tk.W)
        self.dataset_combo.bind('<<ComboboxSelected>>', self.on_dataset_selected)
        
        # Valor de K
        ttk.Label(config_frame, text="Valor de K:").grid(row=0, column=2, padx=(20, 5), sticky=tk.W)
        self.k_var = tk.StringVar(value="3")
        self.k_spinbox = ttk.Spinbox(config_frame, from_=1, to=20, textvariable=self.k_var, width=10)
        self.k_spinbox.grid(row=0, column=3, padx=(0, 20), sticky=tk.W)
        
        # Métrica de distancia
        ttk.Label(config_frame, text="Métrica:").grid(row=0, column=4, padx=(20, 5), sticky=tk.W)
        self.metric_var = tk.StringVar(value="euclidean")
        metric_combo = ttk.Combobox(config_frame, textvariable=self.metric_var, 
                                   values=["euclidean", "manhattan", "minkowski"], 
                                   state="readonly", width=15)
        metric_combo.grid(row=0, column=5, padx=(0, 20), sticky=tk.W)
        
        # Método de validación
        ttk.Label(config_frame, text="Validación:").grid(row=1, column=0, padx=(0, 5), pady=(10, 0), sticky=tk.W)
        self.validation_var = tk.StringVar(value="10-fold")
        validation_combo = ttk.Combobox(config_frame, textvariable=self.validation_var,
                                       values=["10-fold", "Leave-One-Out", "5-fold", "20-fold"],
                                       state="readonly", width=15)
        validation_combo.grid(row=1, column=1, padx=(0, 20), pady=(10, 0), sticky=tk.W)
        
        # ========== SECCIÓN 3: PANEL DE RESULTADOS ==========
        results_frame = ttk.Frame(main_frame)
        results_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)
        
        # Panel de texto para resultados
        self.results_text = tk.Text(results_frame, wrap=tk.WORD, height=15, font=('Consolas', 10))
        results_scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=results_scrollbar.set)
        
        self.results_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        results_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Insertar texto inicial
        self.results_text.insert(tk.END, "="*60 + "\n")
        self.results_text.insert(tk.END, "PRÁCTICA 5 - KNN DESDE CERO\n")
        self.results_text.insert(tk.END, "="*60 + "\n\n")
        self.results_text.insert(tk.END, "1. Carga al menos un dataset (ARFF, .data o CSV)\n")
        self.results_text.insert(tk.END, "2. Configura los parámetros de KNN\n")
        self.results_text.insert(tk.END, "3. Ejecuta el experimento\n\n")
        
        # ========== SECCIÓN 4: BOTONES DE ACCIÓN ==========
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Botón para ejecutar experimento
        self.run_button = tk.Button(button_frame, text="▶ EJECUTAR EXPERIMENTO", 
                                   command=self.run_experiment,
                                   bg=self.button_color, fg='white',
                                   font=('Arial', 10, 'bold'),
                                   padx=20, pady=10,
                                   cursor="hand2")
        self.run_button.grid(row=0, column=0, padx=(0, 10))
        
        # Botón para limpiar resultados
        tk.Button(button_frame, text="🗑️ LIMPIAR", 
                 command=self.clear_results,
                 bg='#e74c3c', fg='white',
                 padx=15, pady=5).grid(row=0, column=1, padx=(0, 10))
        
        # Botón para guardar resultados
        tk.Button(button_frame, text="💾 GUARDAR RESULTADOS", 
                 command=self.save_results,
                 bg='#27ae60', fg='white',
                 padx=15, pady=5).grid(row=0, column=2, padx=(0, 10))
        
        # Botón para ver gráficas
        tk.Button(button_frame, text="📊 VER GRÁFICAS", 
                 command=self.show_graphs,
                 bg='#9b59b6', fg='white',
                 padx=15, pady=5).grid(row=0, column=3)
        
        # ========== SECCIÓN 5: INFORMACIÓN DEL DATASET ==========
        info_frame = ttk.LabelFrame(main_frame, text="Información del Dataset", padding="10")
        info_frame.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E))
        info_frame.columnconfigure(1, weight=1)
        
        self.dataset_info_text = tk.Text(info_frame, wrap=tk.WORD, height=6, font=('Consolas', 9))
        info_scrollbar = ttk.Scrollbar(info_frame, orient=tk.VERTICAL, command=self.dataset_info_text.yview)
        self.dataset_info_text.configure(yscrollcommand=info_scrollbar.set)
        
        self.dataset_info_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        info_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        info_frame.columnconfigure(0, weight=1)
        
        # Configurar etiquetas para hover
        self.setup_hover_effects()
        
    def setup_hover_effects(self):
        """Configura efectos hover para los botones."""
        def on_enter(e):
            e.widget['background'] = self.button_hover
        
        def on_leave(e):
            e.widget['background'] = self.button_color
        
        self.run_button.bind("<Enter>", on_enter)
        self.run_button.bind("<Leave>", on_leave)
    
    def load_dataset(self, dataset_num):
        """Carga un dataset desde archivo."""
        filetypes = [
            ("Todos los archivos soportados", "*.arff;*.data;*.csv"),
            ("Archivos ARFF", "*.arff"),
            ("Archivos .data", "*.data"),
            ("Archivos CSV", "*.csv")
        ]
        
        filepath = filedialog.askopenfilename(
            title=f"Seleccionar Dataset {dataset_num}",
            filetypes=filetypes
        )
        
        if filepath:
            try:
                # Cargar dataset
                X, y, features, classes, df = load_dataset(filepath)
                
                # Crear identificador único
                dataset_id = f"Dataset_{dataset_num}_{os.path.basename(filepath)}"
                
                # Almacenar dataset
                self.datasets[dataset_id] = {
                    'X': X,
                    'y': y,
                    'features': features,
                    'classes': classes,
                    'dataframe': df,
                    'filepath': filepath,
                    'name': os.path.basename(filepath)
                }
                
                # Actualizar entrada
                if dataset_num == 1:
                    self.dataset1_entry.delete(0, tk.END)
                    self.dataset1_entry.insert(0, filepath)
                else:
                    self.dataset2_entry.delete(0, tk.END)
                    self.dataset2_entry.insert(0, filepath)
                
                # Actualizar combobox
                self.update_dataset_combo()
                
                # Seleccionar automáticamente
                self.dataset_var.set(dataset_id)
                self.on_dataset_selected()
                
                messagebox.showinfo("Éxito", f"Dataset cargado exitosamente:\n{dataset_id}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Error al cargar el dataset:\n{str(e)}")
    
    def update_dataset_combo(self):
        """Actualiza la lista de datasets disponibles."""
        dataset_list = list(self.datasets.keys())
        self.dataset_combo['values'] = dataset_list
    
    def on_dataset_selected(self, event=None):
        """Muestra información del dataset seleccionado."""
        dataset_id = self.dataset_var.get()
        
        if dataset_id in self.datasets:
            dataset = self.datasets[dataset_id]
            
            info_text = f"📊 DATASET: {dataset['name']}\n"
            info_text += f"📁 Ruta: {dataset['filepath']}\n"
            info_text += f"📈 Instancias: {dataset['X'].shape[0]}\n"
            info_text += f"🔢 Características: {dataset['X'].shape[1]}\n"
            info_text += f"🎯 Clases: {len(dataset['classes'])} {dataset['classes']}\n"
            info_text += f"📋 Distribución: {np.bincount(dataset['y']).tolist()}\n"
            
            self.dataset_info_text.delete(1.0, tk.END)
            self.dataset_info_text.insert(tk.END, info_text)
            
            self.current_dataset = dataset_id
    
    def run_experiment(self):
        """Ejecuta el experimento KNN con los parámetros configurados."""
        if not self.current_dataset:
            messagebox.showwarning("Advertencia", "Por favor, selecciona un dataset primero.")
            return
        
        # Obtener parámetros
        try:
            k = int(self.k_var.get())
            metric = self.metric_var.get()
            validation = self.validation_var.get()
            
            if k <= 0:
                raise ValueError("K debe ser mayor que 0")
            
        except ValueError as e:
            messagebox.showerror("Error", f"Parámetros inválidos:\n{str(e)}")
            return
        
        # Obtener dataset
        dataset = self.datasets[self.current_dataset]
        X = dataset['X']
        y = dataset['y']
        
        # Normalizar datos
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Determinar número de folds
        if validation == "Leave-One-Out":
            n_folds = len(X)
        else:
            n_folds = int(validation.split('-')[0])
        
        # Ejecutar en un hilo separado para no bloquear la interfaz
        self.run_button.config(state=tk.DISABLED, text="⏳ EJECUTANDO...")
        
        thread = threading.Thread(target=self._run_experiment_thread,
                                 args=(X_scaled, y, k, metric, n_folds, dataset['name']))
        thread.daemon = True
        thread.start()
    
    def _run_experiment_thread(self, X, y, k, metric, n_folds, dataset_name):
        """Ejecuta el experimento en un hilo separado."""
        try:
            # Ejecutar validación cruzada
            if n_folds == len(X):  # Leave-One-Out
                results = leave_one_out_cross_validation(X, y, k_neighbors=k, metric=metric)
                validation_type = "Leave-One-Out"
            else:
                results = k_fold_cross_validation(X, y, k_folds=n_folds, k_neighbors=k, metric=metric)
                validation_type = f"{n_folds}-Fold CV"
            
            # Agregar metadatos
            results['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            results['dataset'] = dataset_name
            results['k'] = k
            results['metric'] = metric
            results['validation'] = validation_type
            
            # Guardar en historial
            self.results_history.append(results)
            
            # Mostrar resultados en la interfaz
            self.root.after(0, self._display_results, results)
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Error en el experimento:\n{str(e)}"))
        finally:
            self.root.after(0, lambda: self.run_button.config(state=tk.NORMAL, text="▶ EJECUTAR EXPERIMENTO"))
    
    def _display_results(self, results):
        """Muestra los resultados en el panel de texto."""
        self.results_text.insert(tk.END, "\n" + "="*60 + "\n")
        self.results_text.insert(tk.END, f"📅 {results['timestamp']}\n")
        self.results_text.insert(tk.END, "="*60 + "\n\n")
        
        self.results_text.insert(tk.END, f"📊 DATASET: {results['dataset']}\n")
        self.results_text.insert(tk.END, f"🎯 CONFIGURACIÓN:\n")
        self.results_text.insert(tk.END, f"   • K: {results['k']}\n")
        self.results_text.insert(tk.END, f"   • Métrica: {results['metric']}\n")
        self.results_text.insert(tk.END, f"   • Validación: {results['validation']}\n\n")
        
        self.results_text.insert(tk.END, "📈 RESULTADOS:\n")
        self.results_text.insert(tk.END, f"   • Accuracy: {results['accuracy']:.4f}\n")
        
        if 'accuracy_std' in results:
            self.results_text.insert(tk.END, f"      (Desviación estándar: {results['accuracy_std']:.4f})\n")
        
        self.results_text.insert(tk.END, f"   • Sensibilidad: {results['sensitivity']:.4f}\n")
        self.results_text.insert(tk.END, f"   • Especificidad: {results['specificity']:.4f}\n")
        self.results_text.insert(tk.END, f"   • Balanced Accuracy: {results['balanced_accuracy']:.4f}\n\n")
        
        # Desplazar al final
        self.results_text.see(tk.END)
    
    def clear_results(self):
        """Limpia el panel de resultados."""
        if messagebox.askyesno("Confirmar", "¿Estás seguro de que quieres limpiar todos los resultados?"):
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, "="*60 + "\n")
            self.results_text.insert(tk.END, "PRÁCTICA 5 - KNN DESDE CERO\n")
            self.results_text.insert(tk.END, "="*60 + "\n\n")
            self.results_text.insert(tk.END, "Resultados limpiados.\n")
            self.results_history.clear()
    
    def save_results(self):
        """Guarda los resultados en un archivo."""
        if not self.results_history:
            messagebox.showwarning("Advertencia", "No hay resultados para guardar.")
            return
        
        filepath = filedialog.asksaveasfilename(
            title="Guardar Resultados",
            defaultextension=".txt",
            filetypes=[("Archivos de texto", "*.txt"), ("Todos los archivos", "*.*")]
        )
        
        if filepath:
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write("="*60 + "\n")
                    f.write("RESULTADOS PRÁCTICA 5 - KNN DESDE CERO\n")
                    f.write("="*60 + "\n\n")
                    
                    for i, result in enumerate(self.results_history, 1):
                        f.write(f"EXPERIMENTO {i}\n")
                        f.write("-"*40 + "\n")
                        f.write(f"Fecha: {result['timestamp']}\n")
                        f.write(f"Dataset: {result['dataset']}\n")
                        f.write(f"K: {result['k']}\n")
                        f.write(f"Métrica: {result['metric']}\n")
                        f.write(f"Validación: {result['validation']}\n\n")
                        
                        f.write("Resultados:\n")
                        f.write(f"  • Accuracy: {result['accuracy']:.4f}\n")
                        if 'accuracy_std' in result:
                            f.write(f"     (Desviación estándar: {result['accuracy_std']:.4f})\n")
                        f.write(f"  • Sensibilidad: {result['sensitivity']:.4f}\n")
                        f.write(f"  • Especificidad: {result['specificity']:.4f}\n")
                        f.write(f"  • Balanced Accuracy: {result['balanced_accuracy']:.4f}\n\n")
                
                messagebox.showinfo("Éxito", f"Resultados guardados en:\n{filepath}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Error al guardar resultados:\n{str(e)}")
    
    def show_graphs(self):
        """Muestra gráficas de los resultados."""
        if len(self.results_history) < 2:
            messagebox.showwarning("Advertencia", "Se necesitan al menos 2 experimentos para generar gráficas comparativas.")
            return
        
        # Crear ventana para gráficas
        graph_window = tk.Toplevel(self.root)
        graph_window.title("Gráficas de Resultados")
        graph_window.geometry("1000x700")
        
        # Crear figura
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Comparación de Resultados KNN', fontsize=16, fontweight='bold')
        
        # Preparar datos
        experiment_labels = [f"Exp {i+1}" for i in range(len(self.results_history))]
        accuracies = [r['accuracy'] for r in self.results_history]
        sensitivities = [r['sensitivity'] for r in self.results_history]
        specificities = [r['specificity'] for r in self.results_history]
        balanced_accs = [r['balanced_accuracy'] for r in self.results_history]
        
        # Configurar colores
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.results_history)))
        
        # Gráfica 1: Accuracy por experimento
        bars1 = axes[0, 0].bar(experiment_labels, accuracies, color=colors, edgecolor='black')
        axes[0, 0].set_title('Accuracy por Experimento', fontweight='bold')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_ylim([0, 1.1])
        
        # Añadir valores en las barras
        for bar, acc in zip(bars1, accuracies):
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                           f'{acc:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Gráfica 2: Sensibilidad vs Especificidad
        x = np.arange(len(experiment_labels))
        width = 0.35
        
        bars2a = axes[0, 1].bar(x - width/2, sensitivities, width, label='Sensibilidad', color='lightblue', edgecolor='black')
        bars2b = axes[0, 1].bar(x + width/2, specificities, width, label='Especificidad', color='lightcoral', edgecolor='black')
        
        axes[0, 1].set_title('Sensibilidad vs Especificidad', fontweight='bold')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(experiment_labels)
        axes[0, 1].legend()
        axes[0, 1].set_ylim([0, 1.1])
        
        # Gráfica 3: Balanced Accuracy
        bars3 = axes[1, 0].bar(experiment_labels, balanced_accs, color='lightgreen', edgecolor='black')
        axes[1, 0].set_title('Balanced Accuracy', fontweight='bold')
        axes[1, 0].set_ylabel('Balanced Accuracy')
        axes[1, 0].set_ylim([0, 1.1])
        
        # Añadir valores en las barras
        for bar, acc in zip(bars3, balanced_accs):
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                           f'{acc:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Gráfica 4: Comparación de K
        k_values = [r['k'] for r in self.results_history]
        axes[1, 1].plot(k_values, accuracies, 'o-', linewidth=2, markersize=8, label='Accuracy')
        axes[1, 1].plot(k_values, balanced_accs, 's--', linewidth=2, markersize=8, label='Balanced Accuracy')
        axes[1, 1].set_title('Evolución con Diferentes Valores de K', fontweight='bold')
        axes[1, 1].set_xlabel('Valor de K')
        axes[1, 1].set_ylabel('Métrica')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Ajustar layout
        plt.tight_layout()
        
        # Mostrar en Tkinter
        canvas = FigureCanvasTkAgg(fig, master=graph_window)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        
        # Botón para guardar gráfica
        def save_graph():
            filepath = filedialog.asksaveasfilename(
                title="Guardar Gráfica",
                defaultextension=".png",
                filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("Todos", "*.*")]
            )
            if filepath:
                fig.savefig(filepath, dpi=300, bbox_inches='tight')
                messagebox.showinfo("Éxito", f"Gráfica guardada en:\n{filepath}")
        
        save_button = tk.Button(graph_window, text="💾 GUARDAR GRÁFICA", 
                               command=save_graph,
                               bg='#27ae60', fg='white',
                               padx=15, pady=5)
        save_button.pack(pady=10)

# ============================================================================
# 5. FUNCIÓN PRINCIPAL
# ============================================================================

def main():
    """Función principal para ejecutar la interfaz gráfica."""
    # Crear ventana principal
    root = tk.Tk()
    
    # Establecer icono (opcional)
    try:
        root.iconbitmap('knn_icon.ico')  # Si tienes un icono
    except:
        pass
    
    # Crear aplicación
    app = KNNGUI(root)
    
    # Ejecutar
    root.mainloop()

if __name__ == "__main__":
    # Instalar dependencias si es necesario
    try:
        import matplotlib
        import seaborn
    except ImportError:
        print("Instalando dependencias necesarias...")
        import subprocess
        subprocess.check_call(["pip", "install", "numpy", "pandas", "matplotlib", "seaborn", "scikit-learn"])
    
    # Ejecutar la interfaz
    main()