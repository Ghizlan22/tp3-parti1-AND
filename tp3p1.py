import streamlit as st
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import matplotlib.pyplot as plt
from mealpy import FloatVar, SCA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Configuration de l'interface Streamlit
st.title("Analyse et Imputation des Données Manquantes")

# Étape 1 : Chargement des données
uploaded_file = st.file_uploader("Charger un fichier CSV", type=["csv"])

# Initialisation de la méthode d'imputation par défaut
method = None
data_imputed = None  # Initialiser la variable 'data_imputed'

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("### Aperçu des données :")
    st.write(data.head())
    # Détection des valeurs manquantes
    missing_values = data.isnull().sum()
    st.write("### Nombre de valeurs manquantes par colonne :")
    st.write(missing_values)

    # Options d'imputation
    st.write("### Méthode d'imputation :")
    method = st.selectbox("Choisir une méthode", ["KNN", "Optimisation avancée", "Aucune (moyenne)"])

    if method == "KNN":
        k = st.slider("Nombre de voisins (k)", 2, 20, 5)  # Ajout d'une valeur par défaut pour k
        if st.button("Imputer les valeurs manquantes"):
            imputer = KNNImputer(n_neighbors=k)
            data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
            st.write("### Données après imputation :")
            st.write(data_imputed)

    elif method == "Optimisation avancée":
        st.write("Optimisation avancée en cours de développement...")
    
    elif method == "Aucune (moyenne)":
        if st.button("Imputer les valeurs manquantes par la moyenne"):
            data_imputed = data.fillna(data.mean())
            st.write("### Données après imputation par la moyenne :")
            st.write(data_imputed)

# Fonction de fitness pour Mealpy
def fitness_function(solution, original_data, missing_mask):
    data_imputed = original_data.copy()
    data_imputed[missing_mask] = solution
    # Évaluation basée sur la cohérence statistique (ou autre critère défini)
    return -accuracy_score(original_data, data_imputed)  # Exemple avec l'accuracy inversée

# Vérification de la méthode choisie pour l'imputation
if method == "Optimisation avancée":
    st.write("Optimisation avancée en cours...")
    if st.button("Imputer avec optimisation"):
        # Création de la variable manquante à optimiser
        missing_mask = data.isnull()
        initial_guess = data.mean().values  # Moyenne comme point de départ
        var_boundaries = [(data.min().min(), data.max().max()) for _ in range(missing_mask.sum().sum())]

        # Configuration de Mealpy
        model = SCA.BaseSCA(epoch=50, pop_size=30)
        solution = model.solve(problem={
            "fit_func": fitness_function,
            "lb": [b[0] for b in var_boundaries],
            "ub": [b[1] for b in var_boundaries],
            "minmax": "min",
            "obj_weights": [1.0]
        })
        
        # Remplir les données avec la solution trouvée
        data_imputed = data.copy()
        data_imputed[missing_mask] = solution.best_position
        st.write("### Données après optimisation avancée :")
        st.write(data_imputed)

# Étape 2 : Entraînement des modèles
if st.button("Entraîner les modèles"):
    if uploaded_file is not None:
        # Vérifier si les données ont été imputées, sinon utiliser les données brutes
        clean_data = data_imputed if data_imputed is not None else data.dropna()
        st.write("Données utilisées pour l'entraînement :")
        st.write(clean_data.head())  # Affichage des données utilisées pour entraîner les modèles
        
        X = clean_data.iloc[:, :-1]
        Y = clean_data.iloc[:, -1]

        # Séparation des données
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

        # Modèle KNN
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(x_train, y_train)
        y_pred = knn.predict(x_test)

        # Calcul des métriques
        acc = accuracy_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred, average='macro')
        prec = precision_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')

        # Affichage des résultats
        st.write("### Résultats du modèle KNN :")
        st.write(f"Accuracy : {acc:.2f}")
        st.write(f"Recall : {rec:.2f}")
        st.write(f"Précision : {prec:.2f}")
        st.write(f"F1 Score : {f1:.2f}")

# Entraînement du modèle MLP
if st.button("Entraîner le modèle MLP"):
    if uploaded_file is not None and data_imputed is not None:  # Vérifier que les données sont imputées
        # Préparer les données
        clean_data = data_imputed.dropna()  # Utiliser les données imputées
        st.write("Données utilisées pour entraîner le modèle MLP :")
        st.write(clean_data.head())  # Affichage des données utilisées pour l'entraînement
        
        X = clean_data.iloc[:, :-1].values
        Y = pd.get_dummies(clean_data.iloc[:, -1]).values  # Encodage One-Hot

        # Séparer les données
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

        # Construire le modèle MLP
        model = Sequential([
            Dense(64, activation='relu', input_shape=(X.shape[1],)),
            Dense(32, activation='relu'),
            Dense(Y.shape[1], activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Entraîner le modèle
        history = model.fit(x_train, y_train, epochs=50, batch_size=32, validation_data=(x_test, y_test))

        # Afficher les performances
        _, acc = model.evaluate(x_test, y_test)
        st.write("### Résultats du modèle MLP :")
        st.write(f"Accuracy : {acc:.2f}")
        
        # Affichage des courbes de performance
        fig, ax = plt.subplots()
        ax.plot(history.history['accuracy'], label='Train Accuracy')
        ax.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax.set_title("Courbes de performance")
        ax.legend()
        st.pyplot(fig)
    else:
        st.write("Erreur : Aucune donnée imputée disponible pour l'entraînement du modèle MLP.")
