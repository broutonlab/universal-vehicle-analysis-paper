import requests
import base64
import json
import time
from tqdm import tqdm
import os
import glob
import cv2

# Sample image file is available at http://plates.openalpr.com/ea7the.jpg
# IMAGE_PATH = '/tmp/sample.jpg' sk_c3e3fad6074ecc25f9adf424
def analyze_img(path):
    SECRET_KEY = 'sk_c3e3fad6074ecc25f9adf424'  # 300 available sk_c6f76b2baa827db20f0043c8
    # data = 'benchmark_alpr'  # benchmark_alpr 2017-IWT4S-HDR_LP-dataset/images   aolp/Subset_AC/AC/test/images
    #
    std_test = path  # '/media/artem/opt/Downloads/broutonlab_alpr-master-0/data/' + data  # +'/*.jpg'

    ext = '.jpg'
    alpr_path = std_test + '/result/'
    # print(alpr_path)
    os.makedirs(alpr_path, exist_ok=True)
    os.makedirs(alpr_path + 'json', exist_ok=True)
    file_count = 0
    correct_count = 0
    null_count = 0

    start_time = time.time()
    print(std_test)
    for file_name in tqdm(glob.glob(std_test + '/*' + ext)):
        correct_name = file_name.split('/')[-1]
        correct_name = correct_name.split('.')[0]
        # if data == '2017-IWT4S-HDR_LP-dataset/images':
        #     correct_name = correct_name.split('_')[0]
        # print(correct_name)

        img = cv2.imread(file_name)
        file_count += 1
        # if file_count == 10:
        #     break;
        #     # print(file_name)
        with open(file_name, 'rb') as image_file:
            img_base64 = base64.b64encode(image_file.read())
            # print(img_base64)
        url = 'https://api.openalp' \
              'r.com/v2/recognize_bytes?recognize_vehicle=1&country=us&secret_key=%s' % (SECRET_KEY)
        r = requests.post(url, data=img_base64)
        file_object = open(alpr_path + 'json/' + correct_name + '.json', 'w')
        json.dump(r.json(), file_object)

        # read and write image
        file_object = open(alpr_path + 'json/' + correct_name + '.json', 'r')
        # print(file_object)
        dict_object = json.load(file_object)
        try:
            results = (dict_object['results'][0])
            letters = (results['plate'])
            if letters == correct_name:
                correct_count += 1
            # print(dict_object)
            # vehicle = (results['vehicle_region'])
            plate = (results['coordinates'])
            # print(plate)
            topLeft = plate[0]
            bottomRight = plate[2]
            plate1 = (topLeft['x'], topLeft['y'])
            plate2 = (bottomRight['x'], bottomRight['y'])

            # point1 = (vehicle['x'], vehicle['y'])
            # point2 = (vehicle['x'] + vehicle['width'], vehicle['y'] + vehicle['height'])
            # cv2.rectangle(img, point1, point2, (0, 0, 255))
            # cv2.putText(img, "openalpr commercial", (5, 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, lineType=cv2.LINE_AA)
            cv2.rectangle(img, plate1, plate2, (255, 0, 0))
            cv2.putText(img, letters, plate1, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, lineType=cv2.LINE_AA)
            # cv2.imshow('alpr', img)
            cv2.imwrite(alpr_path + letters + '_' + correct_name + '.jpg', img)
        except:
            null_count += 1
            # cv2.putText(img, "EMPTY", plate1, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, lineType=cv2.LINE_AA)
            cv2.imwrite(alpr_path + "EMPTY" + '_' + correct_name + '.jpg', img)

        # cv2.waitKey(0)
    print(time.time() - start_time)
    print('{:.2f} ({} / {})'.format(correct_count / file_count, correct_count, file_count, null_count))


if __name__ == '__main__':
    path = '/media/artem/opt/Downloads/broutonlab_alpr-master-0/data/aolp/Subset_LE/LE/test/images'
    analyze_img(path)

