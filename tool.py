
from rpn.model import *
from predict import Show_Final_Predictions
from classifier.Net import Net
from preprocess import preprocess_image
import torch
from torchvision import transforms
from PIL import Image
import argparse


def tool(args=None):
        parser = argparse.ArgumentParser(description='Simple script for usage of tool')
        parser.add_argument('--directory', help='Path to the Directory containing the image')
        # parser.add_argument('--all', help='')
        parser.add_argument('--name', help='Image to be used')
        parser_args = parser.parse_args(args)
        # print(parser)
        name = parser_args.name
        directory = parser_args.directory

        if parser_args.directory == None:
                raise ValueError('Must provide the directory name')

        if parser_args.name == None:
                raise ValueError('Must provide the image name in the directory')

        print("Loading Models ...")
        rpn_model = torch.load('rpn_99.pth', map_location=torch.device('cpu'))
        rpn_model.eval()
        classifier_model = Net()
        classifier_model.load_state_dict(torch.load('classifier_final.pth', map_location=torch.device('cpu')))
        classifier_model.eval()
        print("Models Loaded")
        print()
        transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        print("Obtaining image and checking for cosmic ray ...")
        arr = preprocess_image(directory,name)

        if arr is not None:
                img = Image.fromarray(arr)
                img_PIL = img.resize((1024,1024)).convert('RGB')
                img_tensor = transform(img_PIL)
                Show_Final_Predictions(img_tensor.view(1,3,1024,1024), img_PIL, name, rpn_model, classifier_model)
                print()
                print("Saved to directory- \'predictions/\'")


if __name__ == '__main__':
    tool()






