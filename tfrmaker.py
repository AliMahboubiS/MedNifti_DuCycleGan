# Loading necessary libraries
import tensorflow as tf
from tensorflow import keras
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

nii_img_CT  = nib.load('sample_data/T_CHB-001.nii.gz')
nii_img_PT  = nib.load('sample_data/PT_CHB-001.nii.gz')
nii_data_CT = nii_img_CT.get_fdata()
nii_data_PT = nii_img_PT.get_fdata()

nii_aff  = nii_img_CT.affine
nii_hdr  = nii_img_CT.header

print(nii_aff ,'\n',nii_hdr)
print(nii_data_CT.shape)
print(nii_data_PT.shape)

# if(len(nii_data_CT.shape)==3):
#    for slice_Number in range(nii_data_CT.shape[2]):
#        plt.imshow(nii_data_CT[:,:,slice_Number ])
#        plt.show()


# Load MNIST data
# Preprocessing
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = nii_data_CT
y_train = nii_data_PT

x_train = x_train / 255.0
x_test = x_test / 255.0
# Track the data type
dataType = x_train.dtype
print(f"Data type: {dataType}")
labelType = y_test.dtype
print(f"Data type: {labelType}")

im_list = []
n_samples_to_show = 16
c = 0
for i in range(n_samples_to_show):
  im_list.append(x_train[i])
# Visualization
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
fig = plt.figure(figsize=(4., 4.))
# Ref: https://matplotlib.org/3.1.1/gallery/axes_grid1/simple_axesgrid.html
grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(4, 4),  # creates 2x2 grid of axes
                 axes_pad=0.1,  # pad between axes in inch.
                 )
# Show image grid
for ax, im in zip(grid, im_list):
    # Iterating over the grid returns the Axes.
    ax.imshow(im, 'gray')
plt.show()

# Convert values to compatible tf.Example types.
def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    # Create the features dictionary.
def image_example(image, label, dimension):
    feature = {
        'dimension': _int64_feature(dimension),
        'label': _bytes_feature(label.tobytes()),
        'image_raw': _bytes_feature(image.tobytes()),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


record_file = 'CHB.tfrecords'
n_samples = x_train.shape[0]
dimension = x_train.shape[1]
with tf.io.TFRecordWriter(record_file) as writer:
   for i in range(n_samples):
      image = x_train[i]
      label = y_train[i]
      tf_example = image_example(image, label, dimension)
      writer.write(tf_example.SerializeToString())