from tqdm import tqdm
import copy
import numpy as np
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from focal_loss import FocalLoss
from model import *
from save_graph import save_loss_graphs
from dataloaders import dataloaders, dataset_sizes
import argparse

parser = argparse.ArgumentParser(description='Simple script for usage of tool')
parser.add_argument('--e',default=100, help='Number of Epochs', type=int)
parser.add_argument('--model',default='resnet50', help='resnet50 head or resnet18 head')
parser.add_argument('--exp',default=1, help='Experiment number', type = int)
parser = parser.parse_args()

if parser.model == 'resnet50':
    rpn = resnet50(num_classes=2, pretrained=True)
elif parser.model == 'resnet18':
    rpn = resnet18(num_classes=2, pretrained=True)
else:
    raise ValueError('Must provide a correct Model type')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


""" Training Function:
    Saves the model in directory 'saved_models' after every 20 epochs
"""

def train_model(model, focal_loss, start_epoch = 0, num_epochs=25, exp = 1, use_GPU = True):
  model = model.to(device)
  optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
  # scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
  since = time.time()
  model_wts = copy.deepcopy(model.state_dict())
  best_loss = 100.0

  if not os.path.exists('saved_models/'+str(exp)):
    os.makedirs(os.path.join('saved_models/')+str(exp))

  e_count = start_epoch%20
  rpn_lambda = ((start_epoch-e_count)/20)*0.1

  cls_loss_values = []
  reg_loss_values = []
  tot_loss_values = []

  val_cls_loss_values = []
  val_reg_loss_values = []
  val_tot_loss_values = []

  for epoch in range(num_epochs): 
      e_count+=1  
      if (e_count==20):
          e_count = 0
          rpn_lambda+=0.1
          torch.save(model,'saved_models/'+str(exp)+'/rpn_'+str(start_epoch+epoch)+'.pth')
      print('Epoch {}/{}'.format(epoch + start_epoch, start_epoch+ num_epochs-1))
      print('-' * 10)

      # Each epoch has a training and validation phase
      for phase in ['train', 'valid']:
          if phase == 'train':
              model.train()  # Set model to training mode
          else:
              model.eval()   # Set model to evaluate mode
                
          cls_loss = 0.0
          reg_loss = 0.0  
          running_loss = 0.0            
          total_cls_loss = 0.0
          total_reg_loss = 0.0
          total_epoch_loss = 0.0
          index = 0

          tk0 = tqdm(dataloaders[phase], total=int(len(dataloaders[phase])))
          # Iterate over data.
          for inputs, labels, locations in tk0:
            # Batch size 1
              index+=1
              inputs = inputs[0].to(device)
              target_labels_np = labels[0].data.numpy()
              pos_arr = np.where(target_labels_np>0 )[0]
              if (use_GPU):
                target_labels = labels[0].to(device)
                target_locs = locations[0].to(device)
              pos = (target_labels == 1)
              # zero the parameter gradients
              optimizer.zero_grad()
              # forward
              # track history if only in train
              with torch.set_grad_enabled(phase == 'train'):
                  pred_labels, locs = model(inputs)
                  pred_locs = locs[0]
                  if torch.isnan(pred_locs).any():
                      print("Nan in output prediction:", index)
                      return None
                  # rpn_cls_loss = F.cross_entropy(pred_labels, target_labels.long(), ignore_index = -1,weight = class_weights)
                  rpn_cls_loss = focal_loss(pred_labels[0],target_labels.long())
                
                  if not(pos_arr.size == 0):
                      mask = pos.unsqueeze(1).expand_as(pred_locs)
                      mask_loc_preds = pred_locs[mask].view(-1,4)
                      mask_loc_targets = target_locs[mask].view(-1, 4)
                      # print("No of positives:",mask_loc_preds.shape)
                      x = torch.abs(mask_loc_targets - mask_loc_preds)
                      rpn_loc_loss = (((x < 1).float() * 0.5 * x*x) + ((x >= 1).float() * (x-0.5))).sum()
#                       print(rpn_loc_loss)
                      rpn_loc_loss_item = rpn_loc_loss.item()
                      # N_reg = (pos).float().sum()
                      # rpn_loc_loss = rpn_loc_loss/ N_reg
                      rpn_loss = rpn_cls_loss + rpn_lambda*rpn_loc_loss
                  else:
                      rpn_loc_loss_item = 0
                      rpn_loss = rpn_cls_loss
                  
                
                  # backward + optimize only if in training phase
                  if phase == 'train':
                      rpn_loss.backward()
                      optimizer.step()

              # statistics
              cls_loss += rpn_cls_loss.item()
              reg_loss += rpn_loc_loss_item
              running_loss += rpn_loss.item()
              tk0.set_postfix(loss=(running_loss / (index)))
          # if phase == 'train':
          #     scheduler.step()

          total_cls_loss = cls_loss / dataset_sizes[phase]
          total_reg_loss = reg_loss / dataset_sizes[phase]
          total_epoch_loss = running_loss / dataset_sizes[phase]

          if phase == 'train':
            cls_loss_values.append(total_cls_loss)
            reg_loss_values.append(total_reg_loss)
            tot_loss_values.append(total_epoch_loss)
          if phase == 'valid':
            val_cls_loss_values.append(total_cls_loss)
            val_reg_loss_values.append(total_reg_loss)
            val_tot_loss_values.append(total_epoch_loss)  
          
          print('{} rpn_cls Loss: {:.4f}       {} rpn_reg Loss: {:.4f}       {} Total Loss: {:.4f}'.format(phase, total_cls_loss,phase, total_reg_loss,phase, total_epoch_loss))

          # deep copy the model
          # if phase == 'train' and total_epoch_loss < best_loss:
          #     best_loss = total_epoch_loss
          #     model_wts = copy.deepcopy(model.state_dict())

          
      print()
  save_loss_graphs(cls_loss_values,reg_loss_values,tot_loss_values,val_cls_loss_values,val_reg_loss_values,val_tot_loss_values,start_epoch,num_epochs, exp)
  time_elapsed = time.time() - since
  print('Training complete in {:.0f}m {:.0f}s'.format(
      time_elapsed // 60, time_elapsed % 60))
  # print('Best Loss: {:4f}'.format(best_loss))
  torch.save(model,'saved_models/'+str(exp)+'/rpn_'+str(start_epoch+num_epochs-1)+'.pth')


fl = FocalLoss()
train_model(rpn, fl, 0, parser.e, parser.exp, torch.cuda.is_available())
