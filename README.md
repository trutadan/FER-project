# How to set up the project
## Install SPIGA
```
git clone https://github.com/andresprados/SPIGA.git
cd spiga
pip install -e .
```

## Set up the project
```
git clone https://github.com/trutadan/FER-project.git
cd FER_project
pip install -r requirements.txt
```

- Copy the "spiga" directory from the "SPIGA" directory installed above, and put it into the current working directory(FER-project).

### Note: 
Comment CUDA code in SPIGA implementation, if you are not using it.

# How to use the project
To effectively use this Facial Emotion Recognition (FER) project, follow these structured steps to preprocess your data, extract facial landmarks, and finally, train and evaluate classifiers on the processed data.

## Data Preparation
Place your dataset into the appropriate directory structured as "raw_dataset_DATASETNAME". This organization is crucial for the preprocessing scripts to locate and process the data correctly.

## Processing Pipeline
To ensure the Facial Emotion Recognition (FER) project operates efficiently, follow this refined processing pipeline:

1. ### File Preprocessing
Initially, preprocess your dataset files to convert them into a format that's more suitable for further processing. This step might involve converting file formats, renaming files for consistency, or organizing the dataset structure.

2. ### Face Detection
Utilize the RetinaFace library to perform face detection on the images. This process identifies the bounding box for each face within the images, which is critical for accurately extracting facial landmarks in the subsequent step.

3. ### Landmark Extraction
With the bounding boxes defined, proceed to extract facial landmarks from each detected face using the specified landmark detection method(SPIGA). This step transforms the raw facial images into a structured set of landmarks, highlighting key facial features necessary for emotion recognition.

4. ### Feature Vector Creation
Generate feature vectors by computing the differences between neutral and expressive facial landmarks. These vectors encapsulate the essential characteristics of facial expressions and serve as the input features for training the classifiers.

## Visualizer
The Visualizer tool is a feature of this project, enabling users to visually compare the facial landmarks of different poses or expressions on a single image. This functionality is particularly useful for analyzing the landmark detection accuracy and understanding how facial expressions are represented through landmarks.

## Model Training and Evaluation
This project supports training and evaluating four different classifiers: Random Forest Classifier (RFC), Support Vector Machine (SVM), K-Nearest Neighbors (kNN), and Multi-Layer Perceptron (MLP). Each classifier can be fine-tuned to optimize performance through grid search, a process that has been predefined in the project to identify the best parameters for each model.

### Classifier Variations
For each classifier, three distinct training variations are available:
- ### Without Dimensionality Reduction:
Trains the classifier on the raw feature vectors without any modification.
- ### With PCA (Principal Component Analysis):
Applies PCA to the feature vectors before training, reducing dimensionality while preserving variance.
- ### With LDA (Linear Discriminant Analysis):
Utilizes LDA for dimensionality reduction, focusing on maximizing class separability which can be particularly beneficial for classification tasks.

### Grid Search for Optimal Parameters
A comprehensive grid search has been implemented for each classifier and training variation, ensuring the selection of optimal hyperparameters. This process systematically explores a range of parameter values, evaluating each configuration based on its performance, to identify the most effective settings for emotion recognition.

### Running the Models
To initiate training and evaluation for a specific classifier and variation, select the corresponding script and configuration. Ensure that the dataset, preprocessing steps, and desired classifier are correctly specified. The project's structure facilitates easy experimentation with different classifiers and preprocessing techniques, allowing for a thorough exploration of their impact on emotion recognition accuracy.

# Available datasets
- Extended Cohn-Kanade (CK+) dataset
- Karolinska Directed Emotional Faces (KDEF) dataset

# Results
![results_image](https://github.com/trutadan/FER-project/blob/master/results.png)

# Future Work
### Integration with Additional Datasets
While this project currently supports the Extended Cohn-Kanade (CK+) and Karolinska Directed Emotional Faces (KDEF) datasets, future work could include adapting the pipeline for compatibility with newer or more diverse datasets. This flexibility would allow for broader research and application opportunities.

# External libraries used
- Retina face: [GitHub - serengil/retinaface](https://github.com/serengil/retinaface)
- SPIGA: [GitHub - andresprados/SPIGA](https://github.com/andresprados/SPIGA/tree/main)
