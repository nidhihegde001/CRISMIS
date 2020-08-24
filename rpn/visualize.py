
import Anchors

""" Plots predictions ** Can be used only in Ipython Notebook
    Input:
        inp : Image tensor after appyling transformation
        gt_bbox : A list containing gt boxes
        anchors : A list of predictions
    """

def imshow(inp, gt_bbox, anchors):
  """Imshow for Tensor."""
  inp = inp.numpy().transpose((1, 2, 0))
  mean = np.array([0.5, 0.5, 0.5])
  std = np.array([0.5, 0.5, 0.5])
  inp = std * inp + mean
  inp = np.clip(inp, 0, 1)
  plt.imshow(inp)
  ax = plt.gca()
  # plot each box
  for box in gt_bbox:
    # get coordinates
    y1, x1, y2, x2 = box
    # calculate width and height of the box
    width, height = x2 - x1, y2 - y1
    # create the shape
    rect = Rectangle((x1, y1), width, height, fill=False, color='green')
    # draw the box
    ax.add_patch(rect)
  for box in anchors:
    # get coordinates
    y1, x1, y2, x2 = box
    # calculate width and height of the box
    width, height = x2 - x1, y2 - y1
    # create the shape
    rect = Rectangle((x1, y1), width, height, fill=False, color='red')
    # draw the box
    ax.add_patch(rect)
  plt.show()
    
""" Saves comparative predictions in a directory 'visual_results/'
    Input:
        inp : Image tensor after appyling transformation
        gt_bbox : A list containing gt boxes
        roi : A list of final predicted boxes
        ind : index of the image in the dataset
    """

def show_visual_results(inp, gt_bbox,roi, ind):
  """Imshow for Tensor."""
  inp = inp.numpy().transpose((1, 2, 0))
  mean = np.array([0.5, 0.5, 0.5])
  std = np.array([0.5, 0.5, 0.5])
  inp = std * inp + mean
  inp = np.clip(inp, 0, 1)

  fig, axs = plt.subplots(1, 2)
  # print(axs.shape)
  axs[0].imshow(inp)
  axs[0].set_title('GROUND TRUTH')
  # fig.set_title('subplot 1')
  # plt.imshow(inp)
  # plot each box
  for box in gt_bbox:
    # get coordinates
    y1, x1, y2, x2 = box
    y1-=15
    x1-=15
    y2+=15
    x2+=15
    # calculate width and height of the box
    width, height = x2 - x1, y2 - y1
    # create the shape
    rect = Rectangle((x1, y1), width, height, fill=False, color='green')
    # draw the box
    axs[0].add_patch(rect)

  axs[1].imshow(inp)
  axs[1].set_title('FINAL BOXES')
  for box in roi:
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
    axs[1].add_patch(rect)
    
  # plt.show()

  if not os.path.exists('visual_results/'):
      os.makedirs('visual_results/')
  fig.savefig('visual_results/' + str(ind) + '.png' )
  plt.close()


""" Compares the impovement in the scores, for predicted boxes
    Input:
        data : dataset (object of Cosmic_Detection class)
        index : index of the image in the dataset
        model_cpu :  Model after training
        rpn_cpu : Model before training
        fl : focal_loss object

    Prints:
      Indices predicted by the trained model
      Initial objectness scores, Final scores
      Label values predicted (preferably 1 for Positive, 0 for negative, -1 ignored)
    """
def Compare_with_untrained_model(data,index,model_cpu,rpn_cpu,fl):
    ind = index
    labels_np = data[ind][1]
    loc_np = data[ind][2]
    labels = torch.from_numpy(labels_np)
    input_image = data[ind][0].view(1,3,1024,1024)
    # gt_bbox = np.array(data[index][1][0],dtype=np.float32)# y1,x1,y2,x2
    gt_bbox,lab = data.parse_voc_xml(ET.parse(data.annotations[ind]).getroot())
    print("No of GT Boxes: ", len(gt_bbox))
    cls_output, reg_output = model_cpu(input_image)
    # outputs = cls_output.permute(0, 2, 3, 1).contiguous()
    # outputs = outputs.view(1, 32, 32, 9, 2).contiguous().view(-1, 2)
    # loss = fl(pred_labels,target_labels, class_weights)
    loss = fl(cls_output[0],labels.long())
    print("Loss function value for this image=",loss)
    
    soft= nn.Softmax(dim=1)
    objectness_score = soft(cls_output[0])[:,1]
    score_numpy = objectness_score.data.numpy()
    
    output_untrained, _ = rpn_cpu(input_image)
    # output_untrained = output_untrained.permute(0, 2, 3, 1).contiguous()
    # output_untrained = output_untrained.view(1, 32, 32, 9, 2).contiguous().view(-1, 2)
    output_untrained = soft(output_untrained[0])[:,1]
    score_numpy_untrained = output_untrained.data.numpy()
    
    nms_thresh = 0.3
    n_train_pre_nms = 1200
    # n_train_post_nms = 2000
    # n_test_pre_nms = 6000
    # n_test_post_nms = 300

    anchors = np.zeros((196416, 4),dtype=np.float32)
    a = Anchors()
    anchors  = a.generate_all()

    indices_inside = np.where(
      (anchors[:, 0] >= 0) &
      (anchors[:, 1] >= 0) &
      (anchors[:, 2] < 1024) &
      (anchors[:, 3] < 1024)
    )[0]
    roi = anchors
    y1 = roi[:, 0]
    x1 = roi[:, 1]
    y2 = roi[:, 2]
    x2 = roi[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = score_numpy.argsort()[::-1]
    order = order[:n_train_pre_nms]
    keep = []
    # Remove overlapping predictions
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= nms_thresh)[0]
        order = order[inds + 1]
    # final_order_indices = np.intersect1d(keep[:20], indices_inside, assume_unique = True)
    # final_order_indices = np.where(labels_np==1)[0]
    final_order_indices = keep[:20]
    print("Indices obtained")
    print(final_order_indices)
    print("Initial objectness scores at these indices")
    print(score_numpy_untrained[final_order_indices]) 
    print("Final Objectness scores:")
    print(score_numpy[final_order_indices])
    print("Original Label values at these indices")
    print(labels_np[final_order_indices])
    # Final regions of interest boxes and scores as output by the proposal network 
    print("No of proposals:", len(roi[final_order_indices]))
    imshow(data[ind][0].view(3,1024,1024),gt_bbox,roi[final_order_indices])




""" Calculates Final Proposals from the model
    Input:
        data : dataset (object of Cosmic_Detection class)
        index : index of the image in the dataset
        model_cpu :  Model after training
    """
def Show_Final_Proposals(data,index,model_cpu):
  ind = index
  labels_np = data[ind][1]
  loc_np = data[ind][2]
  labels = torch.from_numpy(labels_np)
  input_image = data[ind][0]
  gt_bbox,lab = data.parse_voc_xml(ET.parse(data.annotations[ind]).getroot())
  # print("No of GT Boxes: ", len(gt_bbox))
  cls_output, reg_output = model_cpu(input_image)
  
  soft= nn.Softmax(dim=1)
  objectness_score = soft(cls_output[0])[:,1]
  score_numpy = objectness_score.data.numpy()

  # print('*')
  # nms_thresh = 0.1

  # anchors = np.zeros((196416, 4),dtype=np.float32)
  a = Anchors()
  anchors  = a.generate_all()
  # print('*')
  pred_anchor_locs_numpy = reg_output[0].data.numpy()
  # print('*')
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
  final_order_indices = torchvision.ops.nms( roi_tensor , score_tensor, 0.05)
  indices = np.where(score_numpy[final_order_indices]>0.5)[0]
  if (len(indices)>20):
    indices = indices[:20]
  print("No of proposals:", len(indices))
  print('Final scores', score_numpy[final_order_indices[indices]])
#   Final regions of interest boxes and scores as output by the proposal network 
  anchors = np.expand_dims(anchors[final_order_indices[indices],:], axis=0) if (len(indices)==1) else anchors[final_order_indices[indices],:]
  roi = np.expand_dims(roi[final_order_indices[indices],:], axis=0) if (len(indices)==1) else roi[final_order_indices[indices],:]
  score_numpy = np.expand_dims(score_numpy[final_order_indices[indices]], axis=0) if (len(indices)==1) else score_numpy[final_order_indices[indices]]
  show_visual_results(data[ind][0].view(3,1024,1024),gt_bbox,roi, ind)