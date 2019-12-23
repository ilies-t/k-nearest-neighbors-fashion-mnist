"""
see : https://github.com/zalandoresearch/fashion-mnist#loading-data-with-python-requires-numpy
"""
def load_mnist(path, kind='train'):
    import os
    import gzip
    import numpy as np

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels

"""
see : https://github.com/zalandoresearch/fashion-mnist#labels
"""
def label_to_word(number):
    data = {
        0:	'T-shirt/top',
        1:	'Trouser',
        2:	'Pullover',
        3:	'Dress',
        4:	'Coat',
        5:	'Sandal',
        6:	'Shirt',
        7:	'Sneaker',
        8:	'Bag',
        9:	'Ankle boot'
    }
    return data[number]