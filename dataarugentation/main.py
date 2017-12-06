import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np
import cv2

ia.seed(1)

images = np.empty(shape=[2]);
# Example batch of images.
# The array has shape (32, 64, 64, 3) and dtype uint8.

import os

def seq_setting():
    seq = iaa.Sequential([
        iaa.Fliplr(0.5), # horizontal flips
        iaa.Crop(percent=(0, 0.3)), # random crops
        # Small gaussian blur with random sigma between 0 and 0.5.
        # But we only blur about 50% of all images.
        iaa.Sometimes(0.5,
            iaa.GaussianBlur(sigma=(0, 0.5))
        ),
        # Strengthen or weaken the contrast in each image.
        iaa.ContrastNormalization((0.75, 1.5)),
        # Add gaussian noise.
        # For 50% of all images, we sample the noise once per pixel.
        # For the other 50% of all images, we sample the noise per pixel AND
        # channel. This can change the color (not only brightness) of the
        # pixels.
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
        # Make some images brighter and some darker.
        # In 20% of all cases, we sample the multiplier once per channel,
        # which can end up changing the color of the images.
        iaa.Multiply((0.8, 1.2), per_channel=0.2),
        # Apply affine transformations to each image.
        # Scale/zoom them, translate/move them, rotate them and shear them.
        iaa.Affine(
            scale={"x": (0.6, 1.2), "y": (0.6, 1.2)},
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            rotate=(-25, 25),
            shear=(-8, 8)
        )
    ], random_order=False) # apply augmenters in random order
    return seq
def load_image_dir(image_txt):
    image_dir_list = []
    image_label_list = []
    with open(image_txt,'r') as file:
        while 1:
            line = file.readline()
            if (line):
                line_spilt = line.split()
                line_spilt[0] = line_spilt[0].replace('.jpg','')
                image_dir_list.append(line_spilt[0])
                image_label_list.append(line_spilt[1])
            else:
                break
        image_dir = np.array(image_dir_list)
        image_label = np.array(image_label_list)
    return image_dir,image_label

def load_batch_image(image_batch_dir):
    images = cv2.imread(image_batch_dir)
    return images

def main():
    #batch_size = 32
    f1 = '/home/frelam/ai_challenger_scene_train_20170904/caffe_train.txt'
    output_txt = '/media/frelam/683cd494-d120-4dbd-81a4-eb3a90330106/ai_challange_scene_dataset_lmdb/aichallenger_arug20171116/dataarymentation/train_arug1118.txt'
    duplicate_num_per_image = 10
    seq = seq_setting()
    image_dir_array,image_label_array = load_image_dir(f1)
    image_new_dir_list = []
    label_new_list = []
    #image_dir_array = image_dir_array[43000:len(image_dir_array)]
    for i,image_dir in enumerate(image_dir_array):
        #images = cv2.imread('/home/frelam/ai_challenger_scene_train_20170904/train/scene_train_images_20170904/' + image_dir + '.jpg')
        for j in range(duplicate_num_per_image):
            #images_aug = seq.augment_image(images)
            #image_dir_temp  = '/media/frelam/683cd494-d120-4dbd-81a4-eb3a90330106/ai_challange_scene_dataset_lmdb/aichallenger_arug20171116/dataarymentation/scene_train_images_arug20171116/' + image_dir + '_' + str(j) + '.jpg'

            #cv2.imwrite(image_dir_temp,images_aug)

            aaa = image_dir + '_' + str(j) + '.jpg'
            image_new_dir_list.append(aaa)
            label_new_list.append(image_label_array[i])

        print i
    with open(output_txt, 'w') as output:
        for i in range(len(image_new_dir_list)):
            output.write(image_new_dir_list[i] + ' ' + label_new_list[i] + '\n')

if __name__ == '__main__':
    main()
