# beepiping
Identification of Beehive Piping Signals

D. Fourer and A. Orlorwska, DETECTION AND IDENTIFICATION OF BEEHIVE PIPING AUDIO SIGNALS, Proc. DCASE 2022\
https://fourer.fr/dcase22

Requirements:\
Dataset : https://ieee-dataport.org/documents/identification-beehive-piping-audio-signals  (DOI: 10.21227/53mq-g936)

Matlab:\
MSVMpack : https://members.loria.fr/FLauer/files/MSVMpack/MSVMpack.html 

MFCC Implementation : https://www.jyu.fi/hytk/fi/laitokset/mutku/en/research/materials/mirtoolbox

Python:\
Keras 2.2.4 \
Tensorflow 1.12 \
numpy 1.16 \
h5py 2.9.0

----------
Usage:

Fig1 : generate the fig1 of the paper \
start_prepare_dataset : prepare the chunks and precompute the descriptors \
dataset_desc : paratition the dataset for for 3-fold cross validation 

start_classif : Main script for evaluating the 4 methods 1D-CNN, TTB+SVM, MFCC+CNN and STFT+CNN through 3-fold cross-validation
