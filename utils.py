import numpy as np
import os
from PIL import Image
from PIL import ImageOps


def generate_images():
    '''
    Loads NumPy bitmap files and generates 28x28 images
    '''
    for file in os.listdir('./numpy_files'):
        image_class = str(file)[18:-4]  # trim off prefix and suffix
        print('Loading :: ', file)
        img_array = np.load('./numpy_files/' + str(file))
        sq_array = []

        for img in img_array:
            sq_array.append(np.array(img).reshape(28, 28))

        sq_array = sq_array[:5000]    # trim to 5,000 images of each class
        for i in range(len(sq_array)):
            img2 = Image.fromarray(sq_array[i])
            img3 = ImageOps.invert(img2)
            if i % 5 == 1:
                prefix = 'validation'
            else:
                prefix = 'train'
            filename = './images/%s/%s/%s_%d.png' % (prefix, image_class,
                                                     image_class, i)
            img3.save(filename)

# generate_images()
