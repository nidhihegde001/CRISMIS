import numpy as np
'''
  Generates anchors on different feature maps obtained from FPN:
    level3 -> Size of Feature Map: (8,8)      area [16.0, 32.0, 64.0]  ratios = [0.5, 1, 2]
    level4 -> Size of Feature Map: (16,16)    area [16.0, 32.0, 64.0]  ratios = [0.5, 1, 2]
    level5 -> Size of Feature Map: (32,32)    area [16.0, 32.0, 64.0]  ratios = [0.5, 1, 2]
    level6 -> Size of Feature Map: (64,64)    area [16.0, 32.0, 64.0]  ratios = [0.5, 1, 2]
    level7 -> Size of Feature Map: (128,128)  area [64.0,128.0,256.0]  ratios = [0.5, 1, 2]
'''

class Anchors():
      def __init__(self):
        super(Anchors, self).__init__()

      def generate_at_level(self,level):
        if level==7:
          anchor_areas = [ x**2 for x in [64.0,128.0,256.0]]
        else:
          anchor_areas = [ x**2 for x in [16.0, 32.0, 64.0]]
        ratios = [0.5, 1, 2]
        # anchor_scales = [2 ** x for x in [0, 1 / 3, 2 / 3]]
        anchor_scales = [1]
        fe_size = 2**(10 - level)
        anchors = np.zeros((len(anchor_areas)*len(ratios)*len(anchor_scales)*fe_size*fe_size, 4),dtype=np.float32)
        ctr =  np.zeros((fe_size*fe_size, 2),dtype=np.float32)
        # print("Total no of anchors:", anchors.shape)
        # Generate centers for each feature map pixel
        pts = np.arange(1024/fe_size-1, 1024, 1024/fe_size)
        index = 0
        for x in range(len(pts)):
            for y in range(len(pts)):
                ctr[index, 0] = pts[x] - 1024/(fe_size*2)
                ctr[index, 1] = pts[y] - 1024/(fe_size*2)
                index +=1
        # Code to generate anchors at a location in Fmap
        index = 0
        for c in ctr:
          ctr_y, ctr_x = c
          for i in range(len(ratios)):
            for j in range(len(anchor_scales)):
              for k in range(len(anchor_areas)):
                h = anchor_scales[j] * np.sqrt(anchor_areas[k]/ratios[i])
                w = anchor_scales[j] * (anchor_areas[k]/h)
                anchors[index, 0] = ctr_y - h / 2.
                anchors[index, 1] = ctr_x - w / 2.
                anchors[index, 2] = ctr_y + h / 2.
                anchors[index, 3] = ctr_x + w / 2.
                index += 1
        # print("Initial Anchors:")
        # print(anchors)
        valid_anchors = anchors
        valid_anchors[:, slice(0, 4, 2)] = np.clip(
                    anchors[:, slice(0, 4, 2)], 0, 1024)
        valid_anchors[:, slice(1, 4, 2)] = np.clip(
            anchors[:, slice(1, 4, 2)], 0, 1024)
        return valid_anchors

      def generate_all(self):
        locations=[]
        for level in range(3,8):
          loc = self.generate_at_level(level)
          locations.append(loc)
        anchor_locations = np.concatenate((locations[0],locations[1],locations[2],locations[3],locations[4]),axis = 0)
        return anchor_locations