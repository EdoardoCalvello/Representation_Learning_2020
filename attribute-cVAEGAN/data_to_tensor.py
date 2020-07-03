from scikit-image.io import imread
from os.path import join
import numpy as np
from PIL.ImageOps import fit
from PIL import Image

inDirAtt ='/Users/edoardocalvello/Documents/Representation_Learning_2020/celebA/list_attr_celeba.txt'
inDirIm='/Users/edoardocalvello/Documents/Representation_Learning_2020/celebA/img_align_celeba'
IMSIZE=64

f = open(inDirAtt)
noSamples = int(f.readline())
print('There are %d samples' % noSamples)
labels = f.readline().split(' ')
print(labels, type(labels))
dataX = []
dataY = []
for i, line in enumerate(f):
    imName, labels = line.split(' ')[0], line.split(' ')[1:]
    label = np.loadtxt(labels)
    print(imName, label)

    print(i)
    im = imread(join(inDirIm, imName))
    im = Image.fromarray(im)
    im = fit(im, size=(IMSIZE, IMSIZE))
    label = label.astype('int')
    im = np.transpose(im, (2, 0, 1))
    dataX.append(im)
    dataY.append(label)

print(np.shape(dataX))
print(np.shape(dataY))

np.save('/Users/edoardocalvello/Documents/Representation_Learning_2020/xTrain.npy', np.asarray(dataX))
np.save('/Users/edoardocalvello/Documents/Representation_Learning_2020/yAllTrain.npy', np.asarray(dataY))