
from Anchors import Anchors
from torchvision.datasets.voc import VisionDataset
from torchvision import datasets, models, transforms
import os
import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET
#'''
#   Generates target outputs for the data.
#     Input:
#         target: Ground Truth Boxes
#         level: A value between 3 to 7
#     Output:
#         labels: Classification Output corresponding to anchors at a level (feature_map_size*9*2)
#         locs: Regression Output corresponding to anchors at a level (feature_map_size*9*4)
#'''
def generate_targets(target,level): # data is the training set
  gt_bbox = np.array(target[0],dtype=np.float32)
  gt_labels = np.array(target[1])
  a = Anchors()
  valid_anchors = a.generate_at_level(level)
  # Create labels and anchors for valid anchor boxes
  label = np.empty((len(valid_anchors), ), dtype=np.int32)
  # default initialisation
  label.fill(-1)
  if (len(gt_labels)==0):
    anchor_locations = np.empty((len(valid_anchors),) + valid_anchors.shape[1:], dtype=valid_anchors.dtype)
    anchor_locations.fill(0)
    indices = np.where(label == -1)[0]
    neg_index = np.random.choice(indices, size=(500), replace = False)
    label[neg_index] = 0
    return label,anchor_locations

  ious = np.empty((len(valid_anchors), len(gt_labels)), dtype=np.float32)
  ious.fill(0)
# Calculating iou
  for num1, i in enumerate(valid_anchors):
      ya1, xa1, ya2, xa2 = i  
      anchor_area = (ya2 - ya1) * (xa2 - xa1)
      for num2, j in enumerate(gt_bbox):
          yb1, xb1, yb2, xb2 = j
          box_area = (yb2- yb1) * (xb2 - xb1)
          inter_x1 = max([xb1, xa1])
          inter_y1 = max([yb1, ya1])
          inter_x2 = min([xb2, xa2])
          inter_y2 = min([yb2, ya2])
          if (inter_x1 < inter_x2) and (inter_y1 < inter_y2):
              iter_area = (inter_y2 - inter_y1)*(inter_x2 - inter_x1)
              iou = iter_area/(anchor_area+ box_area - iter_area)            
          else:
              iou = 0
          ious[num1, num2] = iou

  gt_argmax_ious = ious.argmax(axis=0)
  gt_max_ious = ious[gt_argmax_ious, np.arange(ious.shape[1])]
  argmax_ious = ious.argmax(axis=1)
  max_ious = ious[np.arange(len(label)), argmax_ious]
  pos_iou_threshold  = 0.7
  neg_iou_threshold = 0.05
  label[max_ious < neg_iou_threshold] = 0
  label[gt_argmax_ious] = 1
  label[max_ious >= pos_iou_threshold] = 1

  n_neg = 1024
  pos_index = np.where(label==1)[0]
  neg_index = np.where(label == 0)[0]
  if len(neg_index) > n_neg:
      disable_index = np.random.choice(neg_index, size=(len(neg_index) - n_neg), replace = False)
      label[disable_index] = -1

  final_neg_anchor_indices = np.where(label==0)[0]
  final_neg_anchors = valid_anchors[final_neg_anchor_indices]
  max_iou_bbox = gt_bbox[argmax_ious]

# Get the correct regression outputs
  height = valid_anchors[:, 2] - valid_anchors[:, 0]
  width = valid_anchors[:, 3] - valid_anchors[:, 1]
  ctr_y = valid_anchors[:, 0] + 0.5 * height
  ctr_x = valid_anchors[:, 1] + 0.5 * width
  base_height = max_iou_bbox[:, 2] - max_iou_bbox[:, 0]
  base_width = max_iou_bbox[:, 3] - max_iou_bbox[:, 1]
  base_ctr_y = max_iou_bbox[:, 0] + 0.5 * base_height
  base_ctr_x = max_iou_bbox[:, 1] + 0.5 * base_width
  eps = np.finfo(height.dtype).eps
  height = np.maximum(height, eps)
  width = np.maximum(width, eps)
  dy = (base_ctr_y - ctr_y) / height
  dx = (base_ctr_x - ctr_x) / width
  dh = np.log(base_height / height)
  dw = np.log(base_width / width)
  anchor_locs = np.vstack((dy, dx, dh, dw)).transpose()
  return label,anchor_locs
  

"""Dataset Class 
       
"""
class Cosmic_Detection(VisionDataset):
  def __init__(self,
                root,                 
                mode='train',
                transform=None,
                target_transform=None,
                transforms=None):
      super(Cosmic_Detection, self).__init__(root, transforms, transform, target_transform)

      voc_root = 'roboflow_350_voc'
      image_dir = os.path.join(voc_root, mode)
      annotation_dir = os.path.join(voc_root, mode +'_annots')

      if not os.path.isdir(voc_root):
          raise RuntimeError('Dataset not found or corrupted')

      file_names = []
      for f in os.listdir(image_dir):
        print(f)
        file_names.append(f[:-4])
      self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
      self.annotations = [os.path.join(annotation_dir, x + ".xml") for x in file_names]
      self.anchor_lab = [os.path.join(mode,"lab", x + ".npy") for x in file_names]
      self.anchor_loc = [os.path.join(mode,"loc",x + ".npy") for x in file_names]
      if not os.path.exists(os.path.join(mode,"lab")):
        os.makedirs(os.path.join(mode,"lab"))
      if not os.path.exists(os.path.join(mode,"loc")):
        os.makedirs(os.path.join(mode,"loc"))
      assert (len(self.images) == len(self.annotations))

  def __getitem__(self, index):
      img = Image.open(self.images[index]).convert('RGB')
      target = self.parse_voc_xml(ET.parse(self.annotations[index]).getroot())

      if self.transforms is not None:
          img, target = self.transforms(img, target)
      if not (os.path.exists(self.anchor_lab[index]) or os.path.exists(self.anchor_loc[index])):  
        labels = []
        locations=[]
        for level in range(5):
          lab,loc = generate_targets(target,level+3)
          labels.append(lab)
          locations.append(loc)
        anchor_labels = np.concatenate((labels[0],labels[1],labels[2],labels[3],labels[4]),axis = 0)
        anchor_locations = np.concatenate((locations[0],locations[1],locations[2],locations[3],locations[4]),axis = 0)
        np.save(self.anchor_lab[index],anchor_labels)
        np.save(self.anchor_loc[index],anchor_locations)
      else:
        anchor_labels = np.load(self.anchor_lab[index])
        anchor_locations = np.load(self.anchor_loc[index])
      return img.view(1,3,1024,1024), anchor_labels, anchor_locations


  def __len__(self):
      return len(self.images)

  def parse_voc_xml(self, root):
        # extract each bounding box
        boxes = list()
        for box in root.findall('.//bndbox'):
            xmin = int(box.find('xmin').text)
            ymin = int(box.find('ymin').text)
            xmax = int(box.find('xmax').text)
            ymax = int(box.find('ymax').text)
            coors = [ymin, xmin, ymax, xmax]
            boxes.append(coors)
        # extract image dimensions
        width = int(root.find('.//size/width').text)
        height = int(root.find('.//size/height').text)
        # extract all the classes
        labels = list()
        for category in root.findall('.//name'):
          labels.append(category.text)
        return boxes, labels