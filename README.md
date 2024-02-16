References:
    Retina face: https://github.com/serengil/retinaface
    SPIGA: https://github.com/andresprados/SPIGA/tree/main

Install SPIGA:
    $ git clone https://github.com/andresprados/SPIGA.git
    $ cd spiga
    $ pip install -e .

Install this project:
    $ git clone https://github.com/trutadan/FER_project.git
    $ cd FER_project
    $ pip install -r requirements.txt
    Then, copy the "spiga" directory from the "SPIGA" directory, and put it into the current working directory(FER_project).
    Note: Comment CUDA code in SPIGA implementation, if you are not using it.

Prerequisites:
    Place the dataset in the "raw_dataset_DATASETNAME" directory.