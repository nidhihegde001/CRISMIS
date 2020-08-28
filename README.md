# CRISMIS - Google Summer of Code 2020 (CERN)

CRISMIS is an open-source machine learning based tool to identify cosmic-ray artifacts in imaging data.
Its purpose is to cater to the needs of planetary scientists and space science teams and help them to automatically filter, and search for the presence of cosmic ray artefacts in their data.

In addition to this, it also provides the flexibility to train on your own data. 
## Installation 
---
1. Clone this repo
2. Install the following packages in your environment:
   * Pytorch 1.6.0, cudatoolkit 10.1, torchvision (refer : https://pytorch.org/)
   * planetaryimage 0.5.0 ```pip install planetaryimage```
   * requests 2.7 ```pip install requests```
   * Astropy 4.0 ```pip install astropy```, Scipy 1.4 ```pip install scipy```
   * Matplotlib ```python -m pip install -U matplotlib```
   * Tqdm ```pip install tqdm```
  
3. To use the tool, download the 2 pre-trained models ```rpn_99.pth``` and ```classifier_final.pth``` and place it in the main directory of this repo. The link to the models is:
   * https://drive.google.com/drive/folders/18ciTnhn0-aSIqPd5NrWT6mfCpEI0Nat4?usp=sharing
  
## Usage:
---
### Tool:
1. To use the tool, first follow the Installation steps above.
1. Type ```python tool.py``` with the required arguments
    *  ```--directory``` name of the directory
    *  ``` --name``` name of the image in directory
1.  A typical example looks like this:
``` 
    python tool.py --directory 2014_215 --name EN1049375684M.IMG
 ```
 The result looks like this:
 
 <img src="README_samples/found.PNG" width="256" height="256">
 
Every tested image is stored in the ```images/``` subdirectory and its result is stored in the directory```prediction/```, if a cosmic ray is found.
If the cosmic ray is not found, the message ``` Cosmic Ray not Found``` is displayed.
 
 ### Training Script:
 1. In case you want to train the RPN model yourself on a different dataset, you may use the training script provided in the directory ```rpn/```
 2. Go to the rpn subfolder
 3. Type ```python train.py``` with the required arguments
    *  ```--e``` number of epochs to train
    * ``` --model``` Backbone : resnet50 or resnet18 are provided as of now
    * ``` --exp``` Experiment No if you want to perform any experiments
 4.  A typical example for training looks like this:
 ``` cd rpn/
      python train.py --e 10 --model resnet50 --exp 1
 ```
 5. The training script saves the model after every 20 epochs in the directory ```saved_models/exp_no``` where 'exp_no' is the experiment number.
 6. Along with the saved models, 3 graphs are generated corresponding to classification, regression and total losses during training and validation.
 7. For visualization of the trained model, the script ```visualize.py``` is provided.
 
### Notebooks
The work has also been distributed to the following notebooks for easy use:

```crismis_tool.ipynb``` This is a colab version of tool, provided you do not want to go through the hassle of creating environments on your local machine.
   (Colab comes pre-installed with most of the libraries)
   
```scanner.ipynb``` Contains utilities for easy retrieval of data from the MESSENGER archive, a GUI for sorting the images based on visual inspection, and a scanner to scan the  images for cosmic ray artefacts from a list of dates from the archive.

```artefacts_library.ipynb``` Pre-processing of predictions at the rpn stage to help in creation of different classes of data for the classifier stage

```RPN_R50_notebook.ipynb```  The training script contained in the ```rpn/``` folder has been converted into the form of a notebook, for step by step training and visualisation.

```classifier_Net.ipynb```  A notebook containing the implementation of the classifier trained on crops of different artifacts
