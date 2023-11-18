import h5py
import cv2
import os
import datetime
import sys
import zipfile


def read_test_images(file_path):
    with h5py.File(file_path,'r') as f:
        os.mkdir('./test_set')
        for folder in f.keys():
            os.mkdir('./test_set/' + folder)
            for imagename in f[folder].keys():
                image_arr = f[folder][imagename]['rgb'][()]
                image_file_name = f[folder][imagename]['rgb_filename'][()].decode('utf-8')
                image_path = f'./test_set/{folder}/{image_file_name}'
                
                cv2.imwrite(image_path, image_arr)    

def write_test_images(folder_path):
    output_file =  'submission_{:%Y%m%dT%H%M}'.format(datetime.datetime.now())
    with h5py.File(output_file + ".h5",'w') as f:
        for dirpath, subdirs, _ in os.walk(folder_path):
            for subdir in subdirs:
                _ = f.create_group(subdir)

            if dirpath == folder_path: continue

            for image in os.listdir(dirpath):
                img_path = os.path.join(dirpath, image)
                img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

                group_name = os.path.basename(dirpath)
                img_group_name = image.split('_')[0]
                img_group = f[group_name].create_group(img_group_name)
                img_group.create_dataset('label', shape=img.shape, data=img)
                img_group.create_dataset('label_filename', data=f'{img_group_name}_label.png')

def main(argv):
    if argv[0] == 'write':
        write_test_images(argv[1])
    elif argv[0] == 'read':
        read_test_images(argv[1])
    else:
        print('Invalid command')

if __name__ == '__main__':
    main(sys.argv[1:])