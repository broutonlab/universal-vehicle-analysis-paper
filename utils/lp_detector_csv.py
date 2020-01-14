import cv2
import glob
import numpy as np
import tensorflow as tf
import re
import pandas as pd
import os
from tqdm import tqdm


def rect_intersection(a, b):
    x = max(a[0], b[0])
    y = max(a[1], b[1])
    w = min(a[0] + a[2], b[0] + b[2]) - x
    h = min(a[1] + a[3], b[1] + b[3]) - y
    if w < 0 or h < 0:
        return None
    return (x, y, w, h)


def rect_area(a):
    return a[2] * a[3]

def lp_detector(path, prefix):
    model_path = '/media/artem/opt/code/alpr/application/broutonlab_alpr/data/best_detectors/plate_detector_12_08.pb'
    # letter_classes_all = ' 0123456789ABCDEFGHIJKLMNPQRSTUVWXYZ'
    letter_classes = ' 0123456789ABCDEFGH1JKLMNPQRSTUVWXYZ'  # indian number plates doesn't have 'i' character

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(model_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
            sess = tf.Session(graph=detection_graph)
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    platemania = True

    # path = '/mnt/mongo-data/mongodb/platemania/'

    os.makedirs(path + prefix, exist_ok=True)


    row_list = []
    file_count = 0
    correct_count = 0
    #os.makedirs(prefix+'_result', exist_ok=True)
    #os.makedirs(prefix+'_img', exist_ok=True)
    #
    #   ../LPRData/two_rows/eval/data*/*.jpg
    for file_name in tqdm(glob.glob(
            path + '/*.jpg')):  # eval_unique_demo/*/*.jpg
        # print(file_name)
        file_count += 1

        lp_class = 'plate'
        # print(file_count
        try:
            img = cv2.imread(file_name)
            name, ext = os.path.splitext(os.path.basename(file_name))
            new_file_name = '{}{}'.format(name, ext)
            if not platemania:
                cv2.imwrite(os.path.join(prefix + '_img', new_file_name), img)
            # cv2.imshow('img',img)
            # cv2.waitKey(0)
            rgbImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            image_expanded = np.expand_dims(rgbImg, axis=0)
            boxes, scores, classes, num = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: image_expanded})
            # img = cv2.resize(img, (img.shape[1]*4, img.shape[0]*4))

            license_number = ''
            correct_number = os.path.splitext(os.path.basename(file_name))[0]
            if num > 0:
                best_cls = []
                best_rects = []
                best_scores = []

                for i in range(len(boxes[0])):
                    box = boxes[0][i]
                    score = scores[0][i]
                    cls = classes[0][i]
                    curr_rect = (int(box[1] * img.shape[1]),
                                 int(box[0] * img.shape[0]),
                                 int(box[3] * img.shape[1]) - int(box[1] * img.shape[1]),
                                 int(box[2] * img.shape[0]) - int(box[0] * img.shape[0]))

                    if score > 0.5 and curr_rect[2] > 0 and curr_rect[3] > 0:
                        best_cls.append(cls)
                        best_rects.append(curr_rect)
                        best_scores.append(score)
                for i in range(len(best_cls)):
                    point1 = (best_rects[i][0], best_rects[i][1])
                    point2 = (best_rects[i][0] + best_rects[i][2], best_rects[i][1] + best_rects[i][3])
                    row_list.append([name, lp_class, point1[0], point2[0], point1[1], point2[1]])
                    # y_padding = 0.04
                    # x_padding = 0.07
                    ymin = best_rects[i][1]
                    ymax = best_rects[i][1] + best_rects[i][3]
                    xmin = best_rects[i][0]
                    xmax = best_rects[i][0] + best_rects[i][2]
                    # ymin = int((-y_padding * ymin) + ymin)
                    # ymax = int((y_padding * ymax) + ymax)
                    # xmin = int((-x_padding * xmin) + xmin)
                    # xmax = int((x_padding * xmax) + xmax)
                    crop_img = img[ymin:ymax, xmin:xmax]
                    cut_name = file_name.split('/')[-1]
                    #print(cut_name)
                    #cv2.imshow('img', crop_img)
                    #cv2.waitKey(0)
                    cv2.imwrite(os.path.join(path + prefix, cut_name), crop_img)
        except:
            print("file is bitten: ", file_name)
                #cv2.rectangle(img, point1, point2, (0, 0, 255))
                #cv2.putText(img, lp_class+': '+str(scores[0][i])[:4], point1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, lineType=cv2.LINE_AA)


        #if correct_number == license_number:
        #    correct_count += 1
            # FOR DEMO IMAGE PURPOSES
        # name, ext = os.path.splitext(os.path.basename(file_name))
        # new_file_name = '{}{}'.format(license_number, ext)
        # cv2.imwrite(os.path.join('correct_test_2', new_file_name), img)
        #else:

        #cv2.imwrite(os.path.join(prefix+'_result', new_file_name), img)

    csv_df = pd.DataFrame(row_list, columns=['name', 'plate', 'xmin', 'xmax', 'ymin', 'ymax'])
    csv_df.to_csv('lp_' + prefix + '.csv', index=None)
    print('Accuracy: {:.2f} ({} correct out of {})'.format(correct_count / file_count, correct_count, file_count))

if __name__ == '__main__':
    prefix = 'aolp_AC_test'
    path = '' # path for images folder
    lp_detector(path, prefix)

