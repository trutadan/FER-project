from files_preprocessor.TFEIDFilesPreprocessor import TFEIDFilesPreprocessor


if __name__ == '__main__':
    source_path = "../../raw_dataset_tfeid"
    destination_path = "../../dataset_tfeid"

    preprocessor = TFEIDFilesPreprocessor(source_path, destination_path)
    preprocessor.preprocess()
