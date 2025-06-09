


# 📰 Fake News Detector

Este proyecto utiliza procesamiento de lenguaje natural (NLP) para clasificar titulares de noticias como reales (`1`) o falsas (`0`) usando modelos de aprendizaje automático.

## 📂 Estructura del proyecto

```
fake-news-detector/
├── dataset/                # Contiene archivos CSV de entrada
│   ├── training_data.csv
│   └── testing_data.csv
├── src/                    # Módulos reutilizables (preprocessing, vectorización, modelos)
├── app/                    # App Streamlit
│   └── streamlit_app.py
├── train_model.py          # Script principal para entrenamiento
├── requirements.txt        # Dependencias del proyecto
└── README.md               # Este archivo
```

## 🚀 Cómo ejecutar

### 1. Clona el repositorio y entra en la carpeta

```bash
git clone https://github.com/tu-usuario/fake-news-detector.git
cd fake-news-detector
```

### 2. Crea y activa un entorno virtual

```bash
python -m venv venv
source venv/bin/activate  # o venv\Scripts\activate en Windows
```

### 3. Instala las dependencias

```bash
pip install -r requirements.txt
```

### 4. Entrena el modelo

```bash
python train_model.py
```

Generará:
- `modelo_entrenado.pkl`
- `vectorizer.pkl`

### 5. Ejecuta la app Streamlit

```bash
streamlit run app/streamlit_app.py
```

## 📊 Resultado del modelo

El modelo entrenado con regresión logística alcanza una precisión de aproximadamente **94%**.

## ✨ Tecnologías utilizadas

- Python
- Pandas, scikit-learn
- NLTK
- Streamlit

## 📌 Autor

Proyecto desarrollado por [Jhoan Gallego](https://github.com/JhoanG956)