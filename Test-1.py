from object_detection.utils import dataset_util, label_map_util
import tensorflow as tf
import os
import xml.etree.ElementTree as ET

def create_tf_example(xml_path, label_map):
    with tf.io.gfile.GFile(xml_path, 'r') as fid:
        xml_str = fid.read()
    xml = ET.fromstring(xml_str)

    filename = xml.find('filename').text
    img_path = os.path.join(os.path.dirname(xml_path), filename)
    with tf.io.gfile.GFile(img_path, 'rb') as fid:
        encoded_jpg = fid.read()

    width = int(xml.find('size')[0].text)
    height = int(xml.find('size')[1].text)

    class_names = []
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []

    for member in xml.findall('object'):
        class_name = member[0].text
        xmins.append(float(member[4][0].text) / width)
        ymins.append(float(member[4][1].text) / height)
        xmaxs.append(float(member[4][2].text) / width)
        ymaxs.append(float(member[4][3].text) / height)
        class_names.append(label_map[class_name])

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(class_names),
        'image/object/class/label': dataset_util.int64_list_feature(class_names),
    }))

    return tf_example

# Convert XMLs to TFRecords
label_map = label_map_util.get_label_map_dict('juice_dataset/annotations/label_map.pbtxt')

writer = tf.io.TFRecordWriter('juice_dataset/annotations/train.record')

for xml_file in os.listdir('juice_dataset/images'):
    if xml_file.endswith('.xml'):
        tf_example = create_tf_example(os.path.join('juice_dataset/images', xml_file), label_map)
        writer.write(tf_example.SerializeToString())

writer.close()
