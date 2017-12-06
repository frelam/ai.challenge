import os
import numpy as np
path = '/media/frelam/683cd494-d120-4dbd-81a4-eb3a90330106/ai_challange_scene_dataset_lmdb/ai_challenger_scene_test_b_20170922/scene_test_b_images_20170922'
files = os.listdir(path)
files_array = np.array(files)
np.save('files_name.npy',files_array)