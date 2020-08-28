from rpn.Anchors import Anchors
from torchvision.ops import nms
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import torch.nn as nn
import numpy as np
import torch
import os

classes = ('Crater' , 'Mixed', 'Short_Streak', 'Space', 'Spot','Long_Streak') # Labels 0-5

transforms_cls = transforms.Compose(
    [transforms.Resize((32,32)),
    #  torchvision.transforms.RandomHorizontalFlip(p=0.5),
    #  torchvision.transforms.RandomVerticalFlip(p=0.5)
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


def Show_Final_Predictions(img_tensor,img_PIL, name, rpn_model, classifier_model):
  with torch.no_grad():
    rois, scores = Final_Proposals(img_tensor, name, rpn_model)
# Loop through all the boxes marked in the image
    artefact = []
    for index, box in enumerate(rois,1):
#       print(name,index, box, width, height)
        artefact.append(img_PIL.crop(box))
        # Save the artefact and its index number for any image
        # artefact.save(save_dir + '/' + str(index) + '__' + str(scores[index-1]) + '__train' + str(ind)  + '.jpg')

    labels = []
    for cropped_image in artefact:
        input_artefact = transforms_cls(cropped_image)
        # print(input_artefact.shape)
        output = classifier_model(input_artefact.view(1,3,32,32))
        # print(output)
        _, predicted = torch.max(output, 1)
        labels.append(predicted)

    show_predictions(img_tensor.view(3,1024,1024),rois, scores, labels, name)


def Final_Proposals(input_image, name, model_cpu):
  cls_output, reg_output = model_cpu(input_image)
  soft= nn.Softmax(dim=1)
  objectness_score = soft(cls_output[0])[:,1]
  score_numpy = objectness_score.data.numpy()

  # anchors = np.zeros((196416, 4),dtype=np.float32)
  a = Anchors()
  anchors  = a.generate_all()
  pred_anchor_locs_numpy = reg_output[0].data.numpy()
  anc_height = anchors[:, 2] - anchors[:, 0]
  anc_width = anchors[:, 3] - anchors[:, 1]
  anc_ctr_y = anchors[:, 0] + 0.5 * anc_height
  anc_ctr_x = anchors[:, 1] + 0.5 * anc_width

  dy = pred_anchor_locs_numpy[:, 0::4]
  dx = pred_anchor_locs_numpy[:, 1::4]
  dh = pred_anchor_locs_numpy[:, 2::4]
  dw = pred_anchor_locs_numpy[:, 3::4]

  # get the predicted box centers, height and width
  ctr_y = dy * anc_height[:, np.newaxis] + anc_ctr_y[:, np.newaxis]
  ctr_x = dx * anc_width[:, np.newaxis] + anc_ctr_x[:, np.newaxis]
  h = np.exp(dh) * anc_height[:, np.newaxis]
  w = np.exp(dw) * anc_width[:, np.newaxis]
  # print('*')
  # Convert loc to y1,x1,y2,x2 format
  roi = np.zeros(pred_anchor_locs_numpy.shape, dtype=anchors.dtype)
  roi[:, 0::4] = ctr_y - 0.5 * h*2
  roi[:, 1::4] = ctr_x - 0.5 * w*2
  roi[:, 2::4] = ctr_y + 0.5 * h*2
  roi[:, 3::4] = ctr_x + 0.5 * w*2
  img_size = (1024, 1024) #Image size
  roi[:, slice(0, 4, 2)] = np.clip(
              roi[:, slice(0, 4, 2)], 0, img_size[0])
  roi[:, slice(1, 4, 2)] = np.clip(
      roi[:, slice(1, 4, 2)], 0, img_size[1])
  # print("Valid ROI's")
  # print(roi)

# Decreasing order of confidence
  order = score_numpy.argsort()[::-1]
  score_numpy = score_numpy[order]
  roi = roi[order,:]
  anchors = anchors[order,:]

#   print("Top 20 scores before nms", score_numpy[:20])
  # Keep only the regions larger than minimum size
  min_size = 16
  hs = roi[:, 2] - roi[:, 0]
  ws = roi[:, 3] - roi[:, 1]
  keep = np.where((hs >= min_size) & (ws >= min_size))[0]
  roi = roi[keep,:]
  anchors = anchors[keep,:]
  score_numpy = score_numpy[keep]
  roi_tensor = torch.from_numpy(roi)
  score_tensor = torch.from_numpy(score_numpy)
  final_order_indices = nms( roi_tensor , score_tensor, 0.05)

  indices = np.where(score_numpy[final_order_indices]>0.5)[0]
  if (len(indices)>20):
    indices = indices[:20]
  print("No of proposals:", len(indices))
  print('Final scores', score_numpy[final_order_indices[indices]])
#   Final regions of interest boxes and scores as output by the proposal network 
  anchors = np.expand_dims(anchors[final_order_indices[indices],:], axis=0) if (len(indices)==1) else anchors[final_order_indices[indices],:]
  roi = np.expand_dims(roi[final_order_indices[indices],:], axis=0) if (len(indices)==1) else roi[final_order_indices[indices],:]
  score_numpy = np.expand_dims(score_numpy[final_order_indices[indices]], axis=0) if (len(indices)==1) else score_numpy[final_order_indices[indices]]
  return roi, score_numpy



def show_predictions(inp, rois, scores, labels, name):
  """Imshow for Tensor."""
  inp = inp.numpy().transpose((1, 2, 0))
  mean = np.array([0.5, 0.5, 0.5])
  std = np.array([0.5, 0.5, 0.5])
  inp = std * inp + mean
  inp = np.clip(inp, 0, 1)

  fig, axs = plt.subplots()
  axs.imshow(inp)
  axs.set_title('FINAL BOXES')
  for i,box in enumerate(rois):
    # get coordinates
    y1, x1, y2, x2 = box
    y1-=10
    x1-=10
    y2+=10
    x2+=10
    # calculate width and height of the box
    width, height = x2 - x1, y2 - y1
    # create the shape
    rect = Rectangle((x1, y1), width, height, fill=False, color='red')
    # draw the box
    axs.add_patch(rect)

    axs.text(x1 + 10, y1 - 10, classes[labels[i]] + " {:.2f}".format(scores[i]),
                color='g', size=8, backgroundcolor="none")
    
  if not os.path.exists('prediction/'):
      os.makedirs('prediction/')
  fig.savefig('prediction/' + name + '.png' )
  plt.close()
