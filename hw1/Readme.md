# Usage

## (1) python train.py [train.csv path]

This will output a trained model named 'hw1.npy' for later use. As 'hw1.npy' is already pre-trained and provided above, it is not necessary to execute this command. ('train.py' is uploaded only for reproducing 'hw1.npy'.)

## (2a) bash hw1.sh [test.csv path] [prediction path]

This will load the 'hw1.npy' model and use it to make predictions for test.csv

## (2b) bash hw1_best.sh [test.csv path] [prediction path]

This will instead load the 'hw1_best.npy' model which is pre-trained and provided above to make predictions. Training source code for 'hw1_best.npy' is not provided, and the model generated by 'train.py' is only compatible with 'hw1.sh'.