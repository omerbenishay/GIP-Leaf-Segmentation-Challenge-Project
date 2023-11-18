from shutil import copyfile
import requests
import getpass
import json
import os
import sys


def download(args):
    """
    example of phenomics API usage in python
    """
    task_id = args.task_id
    location = args.location
    usr, pw = input("username: "), getpass.getpass("password: ")

    # obtain api key
    # usr = input("Username:")
    # pw = getpass.getpass('Password:')
    api_key = auth(usr, pw)

    # get all the images within an annotation task
    # annotation_task_id = input("Please enter annotation task id:")
    annotation_task_id = task_id  # IR: 35 59 61 62   RGB polygon:  30 37 38 45  RGB L/W ratio '48' '70' 88 89  RGB growth-rate 42 64
    images = get_images_in_annotation_task(api_key, annotation_task_id)

    # destination_folder = os.getcwd() + '/tmp/task_' + str(annotation_task_id)
    destination_folder = os.path.join(location, 'task_' + str(annotation_task_id))
    os.makedirs(destination_folder, exist_ok=True)

    sys.stdout.write('Copying files')
    counter = 0
    for image in images:
        sys.stdout.write('.')
        sys.stdout.flush()
        source_filename = image['image_uri']
        #   import pdb; pdb.set_trace()
        destination_file_name = destination_folder + '/' + str(image['frame_id']) + '_' + image['image_uri'].split('/')[
            -1]

        annotations = get_annotations_by_image_id(image['image_id'], api_key)
        annotations_file_name = str(image['frame_id']) + '_' + (image['image_uri'].split('/')[-1]).split('.')[
            -2] + '_annnotations.json'
        if bool(annotations):
            copy_image_from_remote(source_filename, destination_file_name)
            save_annotation_to_disk(annotations, destination_folder, annotations_file_name)
            counter += 1
        if counter == 20:
            continue
            break

    print("Done.")


# get api key
def auth(username, password):
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    data = json.dumps({"username": username, "password": password})
    res = requests.post('https://www.agri-net.org.il/api/auth/', headers=headers, data=data)
    token = json.loads(res.text)['access_token']
    return ('JWT ' + token)


# get all the images in an annotation task
def get_images_in_annotation_task(api_key, annotation_task_id):
    headers = {"Content-Type": "application/json", "Accept": "application/json", "Authorization": api_key}
    res = requests.get(
        'https://www.agri-net.org.il/api/get_annotation_tasks/?annotation_task_id=' + str(annotation_task_id),
        headers=headers)
    json_response = json.loads(res.text)[0]
    images = json_response.get('images', [])
    return (images)


# copy a file to /tmp
def copy_image_to_tmp(source_filename, destination_filename):
    print()
    print(source_filename)
    print(destination_filename)
    copyfile(source_filename, destination_filename)


def copy_image_from_remote(source_file_name, destination_filename):
    # from paramiko import SSHClient
    # from scp import SCPClient
    # ssh = SSHClient()
    # ssh.load_system_host_keys()
    # ssh.connect('172.16.0.3', username=usr, password=pw)
    from shutil import copyfile
    # with SCPClient(ssh.get_transport()) as scp:
    #     scp.get(source_file_name, destination_filename)
    copyfile(source_file_name, destination_filename)


def get_annotations_by_image_id(image_id, token):
    headers = {"Content-Type": "application/json", "Accept": "application/json", "Authorization": f"{str(token)}"}
    res = requests.get(f'https://www.agri-net.org.il/api/get_annotations/?image_id={str(image_id)}', headers=headers)
    json_response = json.loads(res.text)
    return json_response


def save_annotation_to_disk(annotation, location, file_name):
    annotation_file = location + '/' + file_name
    with open(annotation_file, 'w') as outfile:
        json.dump(annotation, outfile)
