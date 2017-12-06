import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np
import cv2

ia.seed(1)

images = np.empty(shape=[2]);
# Example batch of images.
# The array has shape (32, 64, 64, 3) and dtype uint8.
f1 = '/home/frelam/ai_challenger_scene_train_20170904/caffe_train.txt'
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
                #line_spilt[0] = '/home/frelam/Desktop/data argumentation/scene_train_images_20170904' + line_spilt[0]
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


def process_batch_image(dataset_root,dataarumation_root,
                        duplicate_num_per_image,image_dir_batch_array,
                        image_label_batch_array,seq):
    #label_for_save_txt
    label_for_save = []
    for label in image_label_batch_array:
        for j in range(duplicate_num_per_image):
            label_for_save.append(label)
    label_for_save_array = np.array(label_for_save)

    #dir_for_save_txt
    dir_for_save = []
    for dir in image_dir_batch_array:
        for j in range(duplicate_num_per_image):
            dir_for_save.append(dir + '_' + str(j)+'.jpg')

    #dir for read_image
    dir_for_read_image = []
    for dir in image_dir_batch_array:
        for j in range(duplicate_num_per_image):
            dir_for_read_image.append(dataset_root + dir +'.jpg')
    #read image
    images = [cv2.imread(dir) for dir in dir_for_read_image]

    #dir_for_write
    dir_for_write_image = []
    for dir in image_dir_batch_array:
        for j in range(duplicate_num_per_image):
            dir_for_write_image.append(dataarumation_root + dir + '_' + str(j) + '.jpg')

    #process the images
    images_aug = seq.augment_images(images)
    #save the images
    for i in range(len(dir_for_write_image)):
        cv2.imwrite(dir_for_write_image[i],images_aug[i])

    return dir_for_save,label_for_save

def main():
    batch_size = 100
    output_txt = '/media/frelam/683cd494-d120-4dbd-81a4-eb3a90330106/ai_challange_scene_dataset_lmdb/aichallenger_arug20171116/dataarymentation/train_arug2.txt'
    dataset_root = '/home/frelam/ai_challenger_scene_train_20170904/train/scene_train_images_20170904/'
    dataarumation_root = '/media/frelam/683cd494-d120-4dbd-81a4-eb3a90330106/ai_challange_scene_dataset_lmdb/aichallenger_arug20171116/dataarymentation/scene_train_images_arug20171117/'
    duplicate_num_per_image = 10
    seq = seq_setting()
    #input:
    #   image_dir_array
    #   image_label_array
    image_dir_array,image_label_array = load_image_dir(f1)
    dir_chunk_array = [image_dir_array[x:x+batch_size] for x in xrange(0,len(image_dir_array),batch_size)]
    label_chunk_array = [image_label_array[x:x+batch_size] for x in xrange(0,len(image_label_array),batch_size)]

    dir_for_save = []
    label_for_save = []
    #batch process
    for i in range(len(dir_chunk_array)):
        dirs,labels = process_batch_image(dataset_root,dataarumation_root,
                        duplicate_num_per_image,dir_chunk_array[i],
                        label_chunk_array[i],seq)
        dir_for_save.append(dirs)
        label_for_save.append(labels)
        print i
    with open(output_txt, 'w') as output:
        for i in range(len(dir_for_save)):
            for j in range(len(dir_for_save[i])):
                output.write(dir_for_save[i][j] + ' ' + label_for_save[i][j] + '\n')
if __name__ == '__main__':
    main()
