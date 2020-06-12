# EJERCICIO NLP (NATURAL LANGUAGE PROCESSING)
# Carlos Alfonsel (carlos.alfonsel@mbitschool.com)


## 1. Análisis Exploratorio del Dataset (EDA)

- Importación de Librerías y Conjunto de Datos.
- Estudio y representación gráfica de las 8 clases: análisis del balanceo de clases.


## 2. Limpieza del Texto

Se programa la función **clean_text()** que elimina los números y los signos de puntuación, y convierte todas las palabras a minúsculas:

******************************************************************
pattern = re.compile('[{}]'.format(re.escape(string.punctuation)))

def clean_text(doc):
    doc = re.sub(r'\d+', '', doc)
    tokens = nlp(doc)
    tokens = [tok.lower_ for tok in tokens if not tok.is_punct and not tok.is_space]
    filtered_tokens = [pattern.sub('', token) for token in tokens]
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text
******************************************************************


## 3. Definición de Funciones Auxiliares

Se definen las funciones **bow_extractor()** y **tfidf_extractor** para calcular el corpus del texto que se pasa como parámetro:

******************************************************************
def bow_extractor(corpus, ngram_range = (1,1), min_df = 1, max_df = 1.0):
    vectorizer = CountVectorizer(min_df = 1, max_df = 0.95)
    features   = vectorizer.fit_transform(corpus)
    return vectorizer, features
    
def tfidf_extractor(corpus, ngram_range = (1,1), min_df = 1, max_df = 1.0):
    vectorizer = TfidfVectorizer(min_df = 1, max_df = 0.95)
    features   = vectorizer.fit_transform(corpus)
    return vectorizer, features
******************************************************************


## 4. División del Dataset para Entrenamiento y Validación

X_train, X_test, y_train, y_test = train_test_split(datos['Observaciones'], datos['Tipología'], test_size = 0.3, random_state = 0)

## 5. Algoritmos de Clasificación

En este apartado aplicamos los siguientes modelos a nuestros datos: **Logistic Regression**, **Multinomial Naive-Bayes** y **Linear SVM**, con los siguientes resultados en términos de precisión (*accuracy*):

Usando características **BoW** (Bag-of-Words):
**LGR**: 0.61
**MNB**: 0.58
**SVM**: 0.56
Usando características **TF-IDF**:

**LGR**: 0.55
**MNB**: 0.47
**SVM**: 0.64

Optimizando el Modelo Linear SVM con características TF-IDF conseguimos un 0.70 de *accuracy*.

## 6. MEJORAS DE LOS CLASIFICADORES

En este apartado se plantean varias alternativas para ver si se mejoran los resultados del clasificador:

### 6.1. LEMMATIZADO

Se define la función **lemmatize_text()** para extraer las raíces de las palabras:

******************************************************************
def lemmatize_text(text):
    tokens = nlp(text)
    lemmatized_tokens = [tok.lemma_ for tok in tokens]
    lemmatized_text = ' '.join(lemmatized_tokens)
    
    return lemmatized_text
******************************************************************

### 6.2. NUEVOS CLASIFICADORES

Definimos tres nuevos clasificadores: **Árboles de Decisión**, **Random Forest** y **K-Nearest Neighbors**, con estos resultados, una vez realizado el lemmatizado del texto:

Usando características **BoW** (Bag-of-Words) y lemmatizado:
**CART**: 0.58
**RF**  : 0.67
**KNN** : 0.39

Usando características **TF-IDF** y lemmatizado:
**CART**: 0.56
**RF**  : 0.64
**KNN** : 0.61

Optimizando el Modelo Decision Tree Classifier (CART) con características TF-IDF conseguimos un 0.65 de *accuracy*.

## 6.3. REDUCCIÓN DE DIMENSIONALIDAD LSA (Latent Semantic Analysis)

Por último, vamos a probar con una de las técnicas de reducción de dimensionalidad, y analizamos los resultados. Definimos la función **lsa_extractor**, que genera un modelo Latent Semantic Analysis sobre un corpus de texto y utilizando 100 dimensiones:

******************************************************************
def lsa_extractor(corpus, n_dim = 100):
    tfidf      = TfidfVectorizer(use_idf = True)
    svd        = TruncatedSVD(n_dim)
    vectorizer = make_pipeline(tfidf, svd, Normalizer(copy = False))
    features   = vectorizer.fit_transform(corpus)
    return vectorizer, features
******************************************************************

A continuación, aplicamos los siguientes modelos sobre nuestros datos lemmatizados y habiendo aplicado al texto una reducción LSA de 100 dimensiones: **Logistic Regression**, **Random Forest**, **K-Nearest Neighbors** y **Linear SVM**.

Usando características **TF-IDF**, lemmatizado y reducción de dimensionalidad LSA-100:
**LGR**: 0.68
**RF** : 0.55
**KNN**: 0.61
**SVM**: 0.64

## 6.4. MODELO CON WORD EMBEDDINGS

Para finalizar este apartado de mejoras, se aplica un modelo con **Word Embeddings** promediados sobre los siguientes clasificadores:

**LGR** : 0.45
**CART**: 0.24
**RF**  : 0.30
**KNN** : 0.30
**SVM** : 0.39


## CONCLUSIONES:

- EL LEMMATIZADO DE LA VARIABLE TARGET MEJORA LOS RESULTADOS.
- APLICAR UNA REDUCCIÓN DE DIMENSIONALIDAD LSA (Latent Semantic Analysis) MEJORA SIGNIFICATIVAMENTE LOS RESULTADOS.
- LOS MODELOS CON WORD EMBEDDING PROMEDIADO FUNCIONAN PEOR QUE LOS MODELOS MÁS SIMPLES (BoW, TF-IDF) DEBIDO A QUE NUESTRO CONJUNTO DE DATOS ES MUY PEQUEÑO.
- MEJOR ALGORITMO ENCONTRADO: MODELO DE REGRESIÓN LOGÍSTICA, CON CARACTERÍSTICAS TF-IDF, CON DATASET LEMMATIZADO Y REDUCCIÓN LSA DE 100 DIMENSIONES. 
