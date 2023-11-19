import h5py
import cv2
import os
import datetime
import sys
import zipfile
from os.path import normpath, basename


def read_test_images(file_path):
    with h5py.File(file_path,'r') as f:
        os.mkdir('./test_set')
        for folder in f.keys():
            os.mkdir('./test_set/' + folder)
            for imagename in f[folder].keys():
                image_arr = f[folder][imagename]['rgb'][()]
                image_file_name = f[folder][imagename]['rgb_filename'][()]
                image_path = f'./test_set/{folder}/{image_file_name}'
                
                cv2.imwrite(image_path, image_arr)    

def write_test_images(folder_paths):
    if not os.path.exists("submissions"): 
        os.mkdir("submissions")
    output_file =  'submissions/submission_{:%Y%m%dT%H%M}'.format(datetime.datetime.now())
    with h5py.File(output_file + ".h5",'w') as f:
        for folder_path in folder_paths:
            print(f'folder_path {folder_path}')
            folder_group = f.create_group(basename(normpath(folder_path)))
            for img_name in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img_name)
                print(f'image path: {img_path}')
                img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                img_group_name = img_name.split('_')[0]
                img_group = folder_group.create_group(img_group_name)
                img_group.create_dataset('label', shape=img.shape, data=img)
                img_group.create_dataset('label_filename', data=f'{img_group_name}_label.png')

def main(argv):
    if argv[0] == 'write':
        write_test_images(argv[1:])
    elif argv[0] == 'read':
        read_test_images(argv[1])
    else:
        print('Invalid command')

if __name__ == '__main__':
    main(sys.argv[1:])