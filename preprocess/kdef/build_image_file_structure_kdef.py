from files_preprocessor.KDEFFilesPreprocessor import KDEFFilesPreprocessor


if __name__ == '__main__':
    source_path = "../../raw_dataset_kdef"
    destination_path = "../../dataset_kdef"

    preprocessor = KDEFFilesPreprocessor(source_path, destination_path)
    preprocessor.preprocess()
