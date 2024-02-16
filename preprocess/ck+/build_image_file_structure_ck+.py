from files_preprocessor.CKPlusFilesPreprocessor import CKPlusFilesPreprocessor


if __name__ == '__main__':
    source_path = "../../raw_dataset_ck+"
    destination_path = "../../dataset_ck+"

    preprocessor = CKPlusFilesPreprocessor(source_path, destination_path)
    preprocessor.preprocess()
