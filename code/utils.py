import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import tensorflow as tf
import itertools

plt.rcParams['figure.constrained_layout.use'] = True

DOWNLOAD_PATH = 'https://github.com/swghosh/DeepFace/releases/download/weights-vggface2-2d-aligned/VGGFace2_DeepFace_weights_val-0.9034.h5.zip'
MD5_HASH = '0b21fb70cd6901c96c19ac14c9ea8b89'

def get_weights():
    filename = 'deepface.zip'
    downloaded_file_path = tf.keras.utils.get_file(filename, DOWNLOAD_PATH,
        md5_hash=MD5_HASH, extract=True)
    downloaded_h5_file = os.path.join(os.path.dirname(downloaded_file_path),
        os.path.basename(DOWNLOAD_PATH).rstrip('.zip'))
    return downloaded_h5_file

def preprocess_faces(faces, target_size=(152,152)):
  nfaces = faces.shape[0]
  out = np.zeros((nfaces, target_size[0], target_size[1], 3))
  for i in range(nfaces):
    f = cv2.resize(faces[i], target_size, interpolation=cv2.INTER_AREA).T
    f = np.stack((f, f, f), axis=2)
    out[i] = f
  return out.astype('uint8')

def title(y_pred, y_test, target_names, i):
    pred_name = target_names[int(y_pred[i])]
    true_name = target_names[int(y_test[i])]
    return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)

def plot_gallery(images, titles, h, w, n_row=3, n_col=4, rgb=False):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        if rgb == False:
          plt.imshow(images[i].reshape((h, w)).T, cmap=plt.cm.gray)
        else:
          plt.imshow(images[i])
        plt.title(titles[i], size=12)
        plt.axis('off')

def plot_features(data, save=False, filename='features'):
    """Helper function to plot the distribution of features"""
    n_samples = data.shape[0]
    n_features = data.shape[1]
    labels = np.arange(1, n_features+1)
    width = 0.1 # the width of the bars

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot()
    x = np.arange(labels.size)
    
    for i in range(n_samples):
        ax.bar((x-(width*n_samples)/2)+i*width, data[i], width)

    ax.set_ylabel("Values")
    ax.set_xlabel("Succedding diagnostic features")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    plt.grid()

    if save == True:
      plt.savefig(f'Paper/figures/{filename}.pdf', format='pdf')

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
