import matplotlib
# https://stackoverflow.com/questions/37604289/tkinter-tclerror-no-display-name-and-no-display-environment-variable
matplotlib.use('Agg')
from matplotlib import pyplot
from scipy import interp
import matplotlib.pylab as plt
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras import regularizers

from keras.applications.densenet import DenseNet201, preprocess_input
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Dropout
from keras.optimizers import SGD
from sklearn.model_selection import KFold, StratifiedKFold
from keras.callbacks import ModelCheckpoint, EarlyStopping
from pathlib import Path
from PIL import Image
import Augmentor
import numpy as np
import os
import tensorflow as tf
from keras.utils import to_categorical
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix, roc_auc_score, precision_recall_curve




from augmentor import createAugmentationPipeline
from load_dataset import load_dataset
from load_dataset import kfold_dataset

description = '256 neurônios (Fine Tuning)'
base_model_name = 'DenseNet201'
input_shape = (224,224,3)
num_classes = 2
learning_rate = 1e-4
batch_size = 20
steps_per_epoch =50 
epochs = 200
n_partitions = 3
directory = './dataset/train'
with_gap = False
finetuning_epochs = 100
finetuning_from = 483


def createModel():
  if with_gap:
    base_model = DenseNet201(weights='imagenet', include_top=False, input_shape=input_shape, pooling='avg')
    x = base_model.output
  else:
    base_model = DenseNet201(weights='imagenet', include_top=False, input_shape=input_shape)
    x = Flatten()(base_model.output)

 
 
  for layer in base_model.layers:
    layer.trainable = False

  x = Dense(256, activation='relu')(x)
  x = Dropout(0.5)(x)







  predictions = Dense(num_classes, activation = 'softmax')(x)

  model = Model(input = base_model.input, output = predictions)

  optimizer = SGD(lr=learning_rate)
  model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', 'mae'])

  return model, base_model



images, labels = load_dataset(directory)
data_folds = kfold_dataset(images, labels, n_partitions)


tprs = []
mean_fpr = np.linspace(0,1,100)

precisions = []
recalls = []

y_trues = []
y_true_probs = []

all_mae_histories = []
for idx, fold in enumerate(data_folds):

  fold['train/y'] = to_categorical(fold['train/y'], num_classes)
  fold['val/y'] = to_categorical(fold['val/y'], num_classes)

  model, base_model = createModel()

  train_pipeline = createAugmentationPipeline()
  
  augmentation_func = train_pipeline.keras_preprocess_func()
  
  def preprocessing_func(image):
    augmented_image = augmentation_func(image)  
    numpy_image = tf.keras.preprocessing.image.img_to_array(augmented_image)
    input_image = np.expand_dims(numpy_image, axis=0)
    normalized_image = preprocess_input(input_image) 
    return normalized_image

  datagen = ImageDataGenerator(preprocessing_function=preprocessing_func)
  generator = datagen.flow(fold['train/x'], fold['train/y'], batch_size=batch_size)
  checkpoint_path = "training_"+str(idx)+"/cp.ckpt"
  checkpoint_dir = os.path.dirname(checkpoint_path)

  if not os.path.exists(checkpoint_dir):
    os.mkdir(checkpoint_dir)

  # Create checkpoint callback
  cp_callback = [ModelCheckpoint(checkpoint_path,
                                monitor='val_acc',
                                save_weights_only=True,
                                save_best_only=True,
                                verbose=1),
                 ModelCheckpoint(checkpoint_path,
                                monitor='val_loss',
                                save_weights_only=True,
                                save_best_only=True,
                                verbose=1)]



  history = model.fit_generator(
    generator, 
    steps_per_epoch=steps_per_epoch,
    epochs=epochs, 
    verbose=1, 
    class_weight='balanced',
    validation_data=(preprocess_input(fold['val/x']),fold['val/y']),
    callbacks = cp_callback
    )

#  base_model.trainable = True
#  for layer in base_model.layers[:finetuning_from]:
#    layer.trainable = False
#
#  optimizer = SGD(lr=learning_rate/10)
#  model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', 'mae'])
#  model.load_weights(checkpoint_path)
#
#  history = model.fit_generator(
#    generator, 
#    steps_per_epoch=steps_per_epoch,
#    epochs=finetuning_epochs, 
#    verbose=1, 
#    class_weight='balanced',
#    validation_data=(preprocess_input(fold['val/x']),fold['val/y']),
#    callbacks = cp_callback
#    )

  model.load_weights(checkpoint_path)
  y_true = np.argmax(fold['val/y'],1)
  y_predicted_prob = model.predict(fold['val/x'])
  y_true_prob = y_predicted_prob[:,1]

  fpr, tpr, _ = roc_curve(y_true, y_true_prob)
  interpolated_tpr = interp(mean_fpr, fpr, tpr)
  tprs.append(interpolated_tpr)

  precision, recall, _ = precision_recall_curve(y_true, y_true_prob)
  precisions.append(precision)
  recalls.append(recall)

  y_trues.append(y_true)
  y_true_probs.append(y_true_prob)


  mae_history = history.history['val_mean_absolute_error']
  all_mae_histories.append(mae_history[-1])


precision_recall_auc = []
for precision, recall in zip(precisions, recalls):
  pr_auc = auc(recall, precision)
  precision_recall_auc.append(pr_auc)
  
y_trues = np.concatenate(y_trues)
y_true_probs = np.concatenate(y_true_probs)
mean_precision, mean_recall, _ = precision_recall_curve(y_trues, y_true_probs)
mean_pr_auc = auc(mean_recall, mean_precision)

pyplot.figure(0)
pyplot.xlim((0,1))
pyplot.ylim((0,1))
pyplot.plot(mean_recall, mean_precision, color='blue',
         label=r'Média - AUC = %0.2f' % (mean_pr_auc),lw=2, alpha=1)

for i, (precision, recall, pr_auc) in enumerate(zip(precisions, recalls, precision_recall_auc)):
  pyplot.plot(recall, precision, label=r' Fold %d - AUC = %0.2f' % (i, pr_auc), alpha=1, linestyle='--')
plt.title('Curva Precision-Recall - {}'.format(description))
pyplot.xlabel('Recall')
pyplot.ylabel('Precision')
pyplot.legend()
pyplot.savefig('Mean_PR_{}.png'.format(base_model_name))





#############################
mean_tpr = np.mean(tprs, axis=0)
mean_auc = auc(mean_fpr, mean_tpr)

pyplot.figure(1)
pyplot.plot([0, 1], [0, 1], linestyle='--')
pyplot.plot(mean_fpr, mean_tpr, color='blue',
         label=r'Mean ROC (AUC = %0.2f )' % (mean_auc),lw=2, alpha=1)


std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                 label=r'$\pm$ 1 std. dev.')         
pyplot.xlabel('Taxa de Falsos Positivos')
pyplot.ylabel('Taxa de Verdadeiros Positivos')
plt.title('Curva ROC - {}'.format(description))
pyplot.legend()
pyplot.savefig('Mean_ROC_{}.png'.format(base_model_name))


mean_mae = np.mean(all_mae_histories)
print("MAE = "+str(mean_mae))
print("Mean AUC = "+str(mean_auc))








