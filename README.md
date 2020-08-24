# CRISMIS
This project was made as a part of Google Summer of Code 2020
There are 2 parts of this project:
## Installation
1. Clone this repo
2.Install the following packages:
  1. For tool:
  2. For Training:
  
## Usage

### Tool:
1.To use the tool, first follow the Installation steps for the tool
2. Within the main directory, ```cd``` to the  tool directory
3. Type ```python tool.py``` with the required arguments
    * ```--directory``` name of the directory
    *``` --name``` name of the image in the directory
4.  A typical example looks like this:
``` cd tool 
    python tool.py --directory 2011_207 --name EN0220155320M.IMG
 ```
 The result looks like this:
 
This result will be stored in a sub-directory called ```predictions/``` if a cosmic ray is found. If the cosmic ray is not found as in an example below, the following message is displayed:
 
 In all the cases, for every image tested, the corresponding image gets stored in a directory ```images/``` 
 
 
 ### Training Script:
 1. In case you want to train the model yourself for a different dataset, you may use the training script provided in the directory ```rpn/```
 2. Go to the rpn subfolder
 3. Type ```python train.py``` with the required arguments
    * ```--directory``` name of the directory
    *``` --name``` name of the image in the directory
 ``` cd rpn/
      python train.py
 ```

## Step-by step Use
To help with different aspects of the problem, the work has been divided into the following interactive notebooks as follows:



