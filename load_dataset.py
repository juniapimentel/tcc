from pathlib import Path
from PIL import Image
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold


def load_dataset(directory):
    pathlist = Path(directory).glob('**/*.jpg')
    images = []
    labels = []

    for filename in pathlist:
        images.append(np.asarray(Image.open(filename)))
        filename = str(filename)
        if '/MP/' in filename:
            labels.append(0)
            
        if '/MMP/' in filename:
            labels.append(1)
    return np.array(images), np.array(labels)



def kfold_dataset(images, labels, n_partitions):
  data_folds = []
  
  #kf = KFold(n_splits=n_partitions, shuffle=True)
  kf = StratifiedKFold(n_splits=n_partitions, shuffle=False)
  
  
  for train_index, val_index in kf.split(images, labels):
    data_fold_k = {}
    data_fold_k['train/x'] = images[train_index]
    data_fold_k['train/y'] = labels[train_index]
    data_fold_k['val/x'] = images[val_index]
    data_fold_k['val/y'] = labels[val_index]
    data_folds.append(data_fold_k)

  return data_folds








