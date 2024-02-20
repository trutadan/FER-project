import numpy as np
import joblib

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from landmarks.LandmarksDifference import LandmarksDifference
from landmarks.LandmarksExtractor import LandmarksExtractor


class Training:
    def __init__(self, landmarks_dictionary, labels):
        self.landmarks = np.array(list(landmarks_dictionary.values())).reshape(-1, 196)
        self.labels = labels
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.landmarks, self.labels, test_size=0.2, random_state=41)
        self.models = {
            'random_forest': RandomForestClassifier(),
            'mlp': MLPClassifier(max_iter=1000),
            'svm': SVC(),
            'knn': KNeighborsClassifier()
        }

    def grid_search(self, model='random_forest', use_pca=False, use_lda=False):
        pipeline_steps = [('clf', self.models[model])]
        params = {
            'random_forest': {
                'clf__n_estimators': [50, 100, 200, 300, 400, 500],
                'clf__max_depth': [10, 15, 20, 25],
                'clf__bootstrap': [True, False],
                'clf__max_features': ['sqrt', 'log2'],
                'clf__min_samples_leaf': [1, 2, 3, 4, 5],
                'clf__min_samples_split': [2, 3, 4, 5]
            },
            'mlp': {
                'clf__hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
                'clf__activation': ['tanh', 'relu'],
                'clf__solver': ['sgd', 'adam'],
                'clf__alpha': [0.0001, 0.05],
                'clf__learning_rate': ['constant', 'adaptive'],
                'clf__max_iter': [500, 1000, 1500, 2000, 3000]
            },
            'svm': {
                'clf__C': [0.1, 1, 10, 100],
                'clf__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                'clf__gamma': ['scale', 'auto']
            },
            'knn': {
                'clf__n_neighbors': [3, 5, 7, 9],
                'clf__weights': ['uniform', 'distance'],
                'clf__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
            }
        }

        pca_components = [0.95, 0.90, 0.85]

        if use_pca:
            pipeline_steps.insert(0, ('pca', PCA()))
            params['pca__n_components'] = pca_components
        elif use_lda:
            pipeline_steps.insert(0, ('lda', LDA()))

        pipeline = Pipeline(pipeline_steps)
        grid_params = params[model]

        if use_pca:
            grid_params.update({'pca__n_components': pca_components})

        gridsearch = GridSearchCV(pipeline, grid_params, cv=5, verbose=1, n_jobs=-1)
        gridsearch.fit(self.X_train, self.y_train)
        print(f"Best Grid Search Parameters for {model}{' with PCA' if use_pca else (' with LDA' if use_lda else '')}: "
              f"{gridsearch.best_params_}")

    def train(self, model='random_forest'):
        self._train_model(self.X_train, self.X_test, model)

    def train_with_pca(self, n_components=0.95, model='random_forest'):
        pca = PCA(n_components=n_components)
        X_train_pca = pca.fit_transform(self.X_train)
        X_test_pca = pca.transform(self.X_test)
        self._train_model_with_pca(X_train_pca, X_test_pca, model)

    def train_with_lda(self, model='random_forest'):
        lda = LDA()
        X_train_lda = lda.fit_transform(self.X_train, self.y_train)
        X_test_lda = lda.transform(self.X_test)
        self._train_model_with_lda(X_train_lda, X_test_lda, model)

    def _train_model(self, X_train, X_test, model='random_forest'):
        raise NotImplementedError("Subclass must implement abstract method")

    def _train_model_with_pca(self, X_train, X_test, model='random_forest'):
        raise NotImplementedError("Subclass must implement abstract method")

    def _train_model_with_lda(self, X_train, X_test, model='random_forest'):
        raise NotImplementedError("Subclass must implement abstract method")

    @staticmethod
    def load_landmarks(dataset_name):
        image_landmarks = joblib.load(f"../landmarks_{dataset_name}.joblib")
        landmarks = LandmarksDifference(image_landmarks)
        landmarks.process()
        landmarks_dictionary = landmarks.get_differences()

        keys = list(landmarks_dictionary.keys())
        labels = [key.split("-")[1] for key in keys]

        return landmarks_dictionary, labels

    @staticmethod
    def save_landmarks(dataset_name, dataset_path, image_format, emotions):
        landmarks_extractor = LandmarksExtractor(dataset_path, dataset_name, image_format)
        landmarks_extractor.populate_landmarks_dictionary(emotions)
        landmarks_extractor.save()

    @staticmethod
    def _print_metrics(y_true, y_pred):
        accuracy = accuracy_score(y_true, y_pred)
        print(f"\nAccuracy: {accuracy}\n")
        print(f"Classification report:\n{classification_report(y_true, y_pred)}")
