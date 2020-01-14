import os
import cv2
import pandas as pd
import xml.etree.ElementTree as et
import numpy
import json
# import time
from tqdm import tqdm
import os
import glob
import cv2
import sys
# the mock-0.3.1 dir contains testcase.py, testutils.py & mock.py

import lp_detector_csv as lp
import alprcom


def rect_intersection(a,b):
    x = max(a[0], b[0])
    y = max(a[1], b[1])
    w = min(a[0]+a[2], b[0]+b[2]) - x
    h = min(a[1]+a[3], b[1]+b[3]) - y
    # print(a, b)
    # print(w,h,x,y)
    if w < 0 or h < 0:
        return None
    return (x, y, w, h)


def rect_area(a):
    return a[2]*a[3]


def aolp_converter(xml_path):
    xtree = et.parse(xml_path)
    xroot = xtree.getroot()
    object = xroot.find('object')
    plate = object.find('platetext')
    platetext = plate.text
    # print(plate.text)
    bndbox = object.find('bndbox')
    xmin = int(bndbox.find('xmin').text)
    ymin = int(bndbox.find('ymin').text)
    xmax = int(bndbox.find('xmax').text)
    ymax = int(bndbox.find('ymax').text)
    # print(xmin.text, xmax.text, ymin.text, ymax.text)
    gt_rect = [xmin, ymin, xmax, ymax]

    return platetext, gt_rect


def extract_rename(src, dst_path, platetext, no_src, ext):
    # path = img.split('/')
    # dst_folder = '/'.join(path[:-2])
    # dst_path = dst_folder + '/images/'

    # src = img
    dst = dst_path + platetext + '.' + ext

    # print(src, dst)
    try:
        os.rename(src, dst)
    except:
        no_src += 1
        return no_src
    return dst


def draw_IOU(dst_image, rect, gt_rect, ratio, iou_path):
    img = cv2.imread(dst_image)

    cv2.rectangle(img, (rect[0], rect[1]), (rect[2], rect[3]), (255, 0, 0), 1)
    cv2.rectangle(img, (gt_rect[0], gt_rect[1]), (gt_rect[2], gt_rect[3]), (0, 0, 255), 1)
    cv2.putText(img, str(ratio)[:6], (gt_rect[0], gt_rect[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1, lineType=cv2.LINE_AA)

    img_name = dst_image.split('/')[-1]
    # print(iou_path + img_name)
    cv2.imwrite(iou_path + img_name, img)


def broutonlab_lp(path, prefix):
    # prefix = "aolp_AC_test"


    img_path = path + 'jpeg/'
    xml_path = path + 'xml/'
    plates_path = path + 'jpeg/' # jpegaolp_AC_test

    dst_path = path + 'images/'
    new_plates_path = path + 'plates/'
    iou_path = path + 'iou_results/'
    dst_xml_path = path + 'plate_xml/'
    os.makedirs(dst_path, exist_ok=True)
    os.makedirs(dst_xml_path, exist_ok=True)
    os.makedirs(iou_path, exist_ok=True)
    os.makedirs(new_plates_path, exist_ok=True)

    # image extension
    extension = "jpg"

    df = pd.read_csv('lp_' + prefix + '.csv')  # +'_img'
    # print(df[:100])

    # ratio summary
    ratio_sum = 0.0
    no_src_img = 0
    wrong_counter = 0
    for i in range(1, df.shape[0]):  #
        try:
            each = df[i:i + 1]
            xmin = each.xmin.values[0]
            xmax = each.xmax.values[0]
            ymin = each.ymin.values[0]
            ymax = each.ymax.values[0]
            rect = [xmin, ymin, xmax, ymax]

            # print(each)
            each_name = str(each.name.values[0])
            each_xml = xml_path + each_name + '.xml'
            each_img = img_path + each_name + '.jpg'
            each_plate = plates_path + each_name + '.jpg'

            gt_platetext, gt_rect = aolp_converter(each_xml)
            intersection = rect_intersection(rect, gt_rect)
            if intersection is not None:
                ratio = rect_area(intersection) / max(rect_area(gt_rect), rect_area(rect))
                ratio_sum += ratio
                # print('ratio: ', ratio)

            dst_img = extract_rename(each_img, dst_path, gt_platetext, no_src_img, 'jpg')
            dst_plate = extract_rename(each_plate, new_plates_path, gt_platetext, no_src_img, 'jpg')
            dst_xml = extract_rename(each_xml, dst_xml_path, gt_platetext, no_src_img, 'xml')

            if type(dst_img) == int:
                no_src_img = dst_img
                continue
            draw_IOU(dst_img, rect, gt_rect, ratio, iou_path)
        except:
            wrong_counter += 1
            print('went wrong! ')

        # point1 = (rect[0], rect[1])

    print(no_src_img)
    print(ratio_sum / (df.shape[0] - no_src_img - wrong_counter))


def alprcom_lp(path):
    # path = '/media/artem/opt/Downloads/broutonlab_alpr-master-0/data/aolp/Subset_AC/AC/test/'

    img_path = path + 'images/'
    xml_path = path + 'plate_xml/'
    json_path = path + '/images/result/json/'
    iou_path = path + 'alprcom_iou_results/'

    os.makedirs(iou_path, exist_ok=True)

    # image extension
    ext = ".jpg"

    # ratio summary
    ratio_sum = 0.0
    no_src_img = 0
    file_count = 0
    for file_name in tqdm(glob.glob(img_path + '/*' + ext)):
        file_count += 1
        correct_name = file_name.split('/')[-1]
        correct_name = correct_name.split('.')[0]
        file_object = open(json_path + correct_name + '.json', 'r')
        # print(file_object)
        each_xml = xml_path + correct_name + '.xml'
        dict_object = json.load(file_object)
        try:
            results = (dict_object['results'][0])
            plate = (results['coordinates'])
            # print(plate)
            topLeft = plate[0]
            bottomRight = plate[2]
            alpr_rect = [topLeft['x'], topLeft['y'], bottomRight['x'], bottomRight['y']]

            gt_platetext, gt_rect = aolp_converter(each_xml)

            intersection = rect_intersection(alpr_rect, gt_rect)
            if intersection is not None:
                ratio = rect_area(intersection) / max(rect_area(gt_rect), rect_area(alpr_rect))
                ratio_sum += ratio
                # print('ratio: ', ratio)

            draw_IOU(file_name, alpr_rect, gt_rect, ratio, iou_path)
        except:
            print('exception on: ', correct_name)
    # print(no_src_img)
    # print(ratio_sum / file_count)
    

# working directory

if __name__ == '__main__':
    path = '/media/artem/opt/Downloads/broutonlab_alpr-master-0/data/aolp/Subset_RP/RP/test/'
    prefix = 'aolp_RP_test'
    # lp.lp_detector(path+'jpeg/', prefix)
    # broutonlab_lp(path, prefix)
    # alprcom.analyze_img(path+'images/')
    alprcom_lp(path)



