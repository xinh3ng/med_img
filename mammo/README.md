## Image Classification - Mammogram

### Folder Structure
```
train_mammo_model.py
score_mammo_models.py
models/
    models.py
config/
    config_data.py
    constants.py
utils/
    generic_utils.py
    < data util functions organized by data source name >    
```

### Installation
```
$ < cd to med_img repo's root > 
$ virtualenv venv -p python3  # Invoke a virtual env (using 3.6)
$ . venv/bin/activate 
$ pip install -r requirements.txt

$ . scripts/setenv.sh  # Set a few env variables, mostly useful for Spark and importing)
$ . scropts/launch_spyder.sh  # For spyder user 
```

### Train a model
```
Command line arguments are explained inside the script
$ ﻿python train_mammo_model.py --dataset_name=mnist --model_name=cnn --optimizer=adam --loss=categorical_crossentropy
```
