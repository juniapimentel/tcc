#https://stackoverflow.com/questions/11627362/how-to-straighten-a-rotated-rectangle-area-of-an-image-using-opencv-in-python/48553593#48553593
#https://jdhao.github.io/2019/02/23/crop_rotated_rectangle_opencv/#conclusion

import json
import os
from pathlib import Path
from collections import defaultdict
import cv2
import numpy as np


directory_in_str = r'/home/junia/Documents/PG/Dataset_COCO'
pathlist = Path(directory_in_str).glob('**/*.json')
points = defaultdict(dict)

for name in pathlist:
  with open(name) as json_file:
    name = str(name)
    name_image = os.path.splitext(os.path.basename(name))[0]
    name_image = name_image.replace(" - COCO", "")
    data = json.load(json_file)
    for p in data['objects']:
      for k in p['keypoints']:
        if k['id'] == '4':
          points[name_image]["right_shoulder"] = (k["position"]["x"], k["position"]["y"])
          
        if k['id'] == '7':
          points[name_image]["left_shoulder"] = (k["position"]["x"], k["position"]["y"])

        if k['id'] == '10':
          points[name_image]["right_hip"] = (k["position"]["x"], k["position"]["y"])
               
        if k['id'] == '13':
          points[name_image]["left_hip"] = (k["position"]["x"], k["position"]["y"])
         
        
directory = r'/home/junia/Documents/PG/Dataset_MPI'
pathlist = Path(directory).glob('**/*.json')

for name in pathlist:
  with open(name) as json_file:
    name = str(name)
    name_image = os.path.splitext(os.path.basename(name))[0]
    name_image = name_image.replace(" - MPI", "")
    data = json.load(json_file)
    for p in data['objects']:
      for k in p['keypoints']:
        if k['id'] == '3':
          points[name_image]["neck"] = (k["position"]["x"], k["position"]["y"])

print(json.dumps(points, indent=4))


directory_in_str = r'/home/junia/Documents/PG/Dataset_COCO/dataset_coco'
pathlist = Path(directory_in_str).glob('**/*.jpg')

def crop_rect(img, rect):
  # get the parameter of the small rectangle
  center = rect[0]
  size = rect[1]
  angle = rect[2]
  center, size = tuple(map(int, center)), tuple(map(int, size))

  # get row and col num in img
  height, width = img.shape[0], img.shape[1]
  print("width: {}, height: {}".format(width, height))
  M = cv2.getRotationMatrix2D(center, angle, 1)
  img_rot = cv2.warpAffine(img, M, (width, height))
  img_crop = cv2.getRectSubPix(img_rot, size, center)
  return img_crop, img_rot

  
for name in pathlist:
  name = str(name)
  if 'output' in name:
   continue
  name_image = os.path.splitext(os.path.basename(name))[0]
  name_image = name_image.replace(" - COCO", "")
  img = cv2.imread(name)
  
  cnt = np.array([
          [[round(points[name_image]["left_hip"][0]), round(points[name_image]["left_hip"][1])]],
          [[round(points[name_image]["left_shoulder"][0]+200), round(points[name_image]["neck"][1])]],
          [[round(points[name_image]["right_shoulder"][0]-200), round(points[name_image]["neck"][1])]],
          [[round(points[name_image]["right_hip"][0]), round(points[name_image]["right_hip"][1])]]
  ])


  rect = cv2.minAreaRect(cnt)
  box = cv2.boxPoints(rect)
  box = np.int0(box)
  print(box)

  print("bounding box: {}".format(box))
  cv2.drawContours(img, [box], 0, (0, 0, 255), 2)

  # get width and height of the detected rectangle
  width = int(rect[1][0])
  height = int(rect[1][1])

  src_pts = box.astype("float32")
  # corrdinate of the points in box points after the rectangle has been
  # straightened
  dst_pts = np.array([[0, height-1],
                      [0, 0],
                      [width-1, 0],
                      [width-1, height-1]], dtype="float32")

  # the perspective transformation matrix
  M = cv2.getPerspectiveTransform(src_pts, dst_pts)

  # directly warp the rotated rectangle to get the straightened rectangle
  warped = cv2.warpPerspective(img, M, (width, height))
  #cv2.imwrite("crop_img.jpg", warped)
  img_crop, img_rot = crop_rect(img, rect)

  new_size = (int(img_rot.shape[1]/2), int(img_rot.shape[0]/2))
  img_rot_resized = cv2.resize(img_rot, new_size)
  new_size = (int(img.shape[1]/2)), int(img.shape[0]/2)
  img_resized = cv2.resize(img, new_size)

  directory_in_str = r'/home/junia/Documents/PG/Dataset_Cropped'
  name_output = directory_in_str + '/' + name_image + '.jpg'
  cv2.imwrite(name_output, img_crop)



  