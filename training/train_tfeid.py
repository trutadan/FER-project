from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from training.Training import Training


class TFEIDTraining(Training):
    def __init__(self, landmarks_dictionary, labels):
        super().__init__(landmarks_dictionary, labels)

    def _train_model(self, X_train, X_test, model='random_forest'):
        if model == 'random_forest':
            clf = RandomForestClassifier(n_estimators=400, max_depth=25, bootstrap=True,
                                         max_features='sqrt', min_samples_leaf=1, min_samples_split=2)
        elif model == 'mlp':
            clf = MLPClassifier(activation='tanh', alpha=0.0001, hidden_layer_sizes=(100, 50), learning_rate='constant',
                                solver='adam', max_iter=2000)
        elif model == 'svm':
            clf = SVC(C=100, gamma='scale', kernel='rbf')
        elif model == 'knn':
            clf = KNeighborsClassifier(algorithm='auto', n_neighbors=5, weights='uniform')
        else:
            raise ValueError("Unsupported model type")

        clf.fit(X_train, self.y_train)
        y_pred = clf.predict(X_test)
        self._print_metrics(self.y_test, y_pred)

    def _train_model_with_pca(self, X_train, X_test, model='random_forest'):
        if model == 'random_forest':
            clf = RandomForestClassifier(n_estimators=400, max_depth=20, bootstrap=True,
                                         max_features='sqrt', min_samples_leaf=1, min_samples_split=2)
        elif model == 'mlp':
            clf = MLPClassifier(activation='relu', alpha=0.05, hidden_layer_sizes=(100, 50), learning_rate='adaptive',
                                solver='adam', max_iter=3000)
        elif model == 'svm':
            clf = SVC(C=100, gamma='scale', kernel='rbf')
        elif model == 'knn':
            clf = KNeighborsClassifier(algorithm='auto', n_neighbors=5, weights='uniform')
        else:
            raise ValueError("Unsupported model type")

        clf.fit(X_train, self.y_train)
        y_pred = clf.predict(X_test)
        self._print_metrics(self.y_test, y_pred)

    def _train_model_with_lda(self, X_train, X_test, model='random_forest'):
        if model == 'random_forest':
            clf = RandomForestClassifier(n_estimators=50, max_depth=20, bootstrap=True,
                                         max_features='sqrt', min_samples_leaf=3, min_samples_split=2)
        elif model == 'mlp':
            clf = MLPClassifier(activation='tanh', alpha=0.0001, hidden_layer_sizes=(100, 50), learning_rate='adaptive',
                                solver='sgd', max_iter=1000)
        elif model == 'svm':
            clf = SVC(C=1, gamma='scale', kernel='rbf')
        elif model == 'knn':
            clf = KNeighborsClassifier(algorithm='auto', n_neighbors=5, weights='uniform')
        else:
            raise ValueError("Unsupported model type")

        clf.fit(X_train, self.y_train)
        y_pred = clf.predict(X_test)
        self._print_metrics(self.y_test, y_pred)


if __name__ == '__main__':
    dataset_name = "tfeid"
    dataset_path = "../dataset_tfeid"
    image_format = "jpg"
    emotions = ["anger", "contempt", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"]

    # TFEIDTraining.save_landmarks(dataset_name, dataset_path, image_format, emotions)

    landmarks_dictionary, labels = TFEIDTraining.load_landmarks(dataset_name)
    trainer = TFEIDTraining(landmarks_dictionary, labels)

    # Train with RandomForest
    # trainer.train(model='random_forest')

    # Train with PCA and RandomForest
    # trainer.train_with_pca(n_components=0.95, model='random_forest')

    # Train with LDA and RandomForest
    # trainer.train_with_lda(model='random_forest')

    # Train with MLP
    # trainer.train(model='mlp')

    # Train with PCA and MLP
    trainer.train_with_pca(n_components=0.95, model='mlp')

    # Train with LDA and MLP
    trainer.train_with_lda(model='mlp')

    # Train and evaluate SVM
    # trainer.train(model='svm')

    # Train and evaluate SVM with PCA
    # trainer.train_with_pca(n_components=0.95, model='svm')

    # Train and evaluate SVM with LDA
    # trainer.train_with_lda(model='svm')

    # Train and evaluate kNN
    # trainer.train(model='knn')

    # Train and evaluate kNN with PCA
    # trainer.train_with_pca(n_components=0.95, model='knn')

    # Train and evaluate kNN with LDA
    # trainer.train_with_lda(model='knn')

    # Perform grid search for RandomForest
    # trainer.grid_search(model='random_forest', use_pca=False, use_lda=True)

    # Perform grid search for MLP
    # trainer.grid_search(model='mlp', use_pca=False, use_lda=True)

    # Perform grid search for SVM
    # trainer.grid_search(model='svm', use_pca=False, use_lda=True)

    # Perform grid search for kNN
    # trainer.grid_search(model='knn', use_pca=False, use_lda=True)
