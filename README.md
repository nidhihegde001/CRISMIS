# CRISMIS
This project was made as a part of Google Summer of Code 2020
There are 2 parts of this project:
## Installation
1. Clone this repo
2. Install the following packages:
  1. For tool:
  2. For Training:
  
3. To use the tool, download the 2 pre-trained models ```rpn_99.pth``` and ```classifier_final.pth``` and place it in the main directory of this repository. The link to the models is :


  
## Usage

### Tool:
1. To use the tool, first follow the Installation steps for the tool
2. Within the main directory, ```cd``` to the  tool directory
3. Type ```python tool.py``` with the required arguments
    *  ```--directory``` name of the directory
    *  ``` --name``` name of the image in directory
4.  A typical example looks like this:
``` cd tool 
    python tool.py --directory 2014_215 --name EN1049375684M.IMG
 ```
 The result looks like this:
 ![Test Image 1](/README_samples/found.PNG = 256x256)
 
This result will be stored in a sub-directory called ```predictions/``` if a cosmic ray is found. If the cosmic ray is not found as in an example below, the following message is displayed:
 ![Test Image 2](/README_samples/not_found.PNG = 300x50)
 
 In all the cases, for every image tested, the corresponding image gets stored in a directory ```images/``` 
 
 
 ### Training Script:
 1. In case you want to train the RPN model yourself for a different dataset, you may use the training script provided in the directory ```rpn/```
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
 6. Along with the saved models, 3 graphs are generated corresponding to classification, regression and total losses obtained during training and validation. These graphs are also saved in the directory mentioned above.
 7. For visualization of the trained model, the script ```visualize.py``` is provided, to visually assess the predictions.

## Step-by step Use
To help with different aspects of the problem, the work has been divided into the following interactive notebooks as follows:



