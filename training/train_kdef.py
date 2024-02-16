from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from training.Training import Training


class KDEFTraining(Training):
    def __init__(self, landmarks_dictionary, labels):
        super().__init__(landmarks_dictionary, labels)

    def _train_model(self, X_train, X_test, model='random_forest'):
        if model == 'random_forest':
            clf = RandomForestClassifier(n_estimators=50, max_depth=15, bootstrap=False,
                                         max_features='sqrt', min_samples_leaf=1, min_samples_split=2)
        elif model == 'mlp':
            clf = MLPClassifier(activation='tanh', alpha=0.05, hidden_layer_sizes=(100,), learning_rate='constant',
                                solver='adam', max_iter=500)
        elif model == 'svm':
            clf = SVC(C=10, gamma='scale', kernel='rbf')
        elif model == 'knn':
            clf = KNeighborsClassifier(algorithm='auto', n_neighbors=9, weights='distance')
        else:
            raise ValueError("Unsupported model type")

        clf.fit(X_train, self.y_train)
        y_pred = clf.predict(X_test)
        self._print_metrics(self.y_test, y_pred)

    def _train_model_with_pca(self, X_train, X_test, model='random_forest'):
        if model == 'random_forest':
            clf = RandomForestClassifier(n_estimators=50, max_depth=15, bootstrap=True,
                                         max_features='log2', min_samples_leaf=2, min_samples_split=5)
        elif model == 'mlp':
            clf = MLPClassifier(activation='tanh', alpha=0.0001, hidden_layer_sizes=(100,), learning_rate='constant',
                                solver='sgd', max_iter=3000)
        elif model == 'svm':
            clf = SVC(C=1, gamma='scale', kernel='linear')
        elif model == 'knn':
            clf = KNeighborsClassifier(algorithm='auto', n_neighbors=9, weights='distance')
        else:
            raise ValueError("Unsupported model type")

        clf.fit(X_train, self.y_train)
        y_pred = clf.predict(X_test)
        self._print_metrics(self.y_test, y_pred)

    def _train_model_with_lda(self, X_train, X_test, model='random_forest'):
        if model == 'random_forest':
            clf = RandomForestClassifier(n_estimators=50, max_depth=15, bootstrap=True,
                                         max_features='sqrt', min_samples_leaf=1, min_samples_split=3)
        elif model == 'mlp':
            clf = MLPClassifier(activation='tanh', alpha=0.05, hidden_layer_sizes=(100,), learning_rate='adaptive',
                                solver='sgd', max_iter=500)
        elif model == 'svm':
            clf = SVC(C=0.1, gamma='scale', kernel='rbf')
        elif model == 'knn':
            clf = KNeighborsClassifier(algorithm='auto', n_neighbors=7, weights='distance')
        else:
            raise ValueError("Unsupported model type")

        clf.fit(X_train, self.y_train)
        y_pred = clf.predict(X_test)
        self._print_metrics(self.y_test, y_pred)


if __name__ == '__main__':
    dataset_name = "kdef"
    dataset_path = "../dataset_kdef"
    image_format = "jpg"
    emotions = ["neutral", "afraid", "angry", "disgusted", "happy", "sad", "surprised"]

    # KDEFTraining.save_landmarks(dataset_name, dataset_path, image_format, emotions)

    landmarks_dictionary, labels = KDEFTraining.load_landmarks(dataset_name)
    trainer = KDEFTraining(landmarks_dictionary, labels)

    # Train with RandomForest
    # trainer.train(model='random_forest')

    # Train with PCA and RandomForest
    # trainer.train_with_pca(n_components=0.95, model='random_forest')

    # Train with LDA and RandomForest
    # trainer.train_with_lda(model='random_forest')

    # Train with MLP
    # trainer.train(model='mlp')

    # Train with PCA and MLP
    # trainer.train_with_pca(n_components=0.95, model='mlp')

    # Train with LDA and MLP
    # trainer.train_with_lda(model='mlp')

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
    # trainer.grid_search(model='random_forest', use_pca=True, use_lda=False)

    # Perform grid search for MLP
    # trainer.grid_search(model='mlp', use_pca=True, use_lda=False)

    # Perform grid search for SVM
    # trainer.grid_search(model='svm', use_pca=True, use_lda=False)

    # Perform grid search for kNN
    # trainer.grid_search(model='knn', use_pca=True, use_lda=False)
