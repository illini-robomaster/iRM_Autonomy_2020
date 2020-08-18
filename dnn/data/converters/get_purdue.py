import json
import xml.dom.minidom as minidom

import requests
from absl import app, flags
from absl.flags import FLAGS
from tqdm import tqdm
import os
import time

flags.DEFINE_string('input', '../Purdue Dataset/label.json',
                    'the path for input ROCO dataset (please unzip by yourself)')
flags.DEFINE_string('output', '../Purdue Dataset',
                    'the path for input ROCO dataset (please unzip by yourself)')


def main(_argv):
    json_file = json.loads(open(FLAGS.input, 'rb').read())['_via_img_metadata']
    if not os.path.exists(os.path.join(FLAGS.output, 'image')):
        os.mkdir(os.path.join(FLAGS.output, 'image'))
    if not os.path.exists(os.path.join(FLAGS.output, 'image_annotation')):
        os.mkdir(os.path.join(FLAGS.output, 'image_annotation'))
    for m, image_id in zip(tqdm(range(len(json_file)),
                                desc='Getting Purdue Dataset: ',
                                unit='pic',
                                ncols=150),
                           json_file):
        if m < 0:  # to resume download process, set 0 to be your latest download picture index
            continue
        url = json_file[image_id]['filename']
        try:
            r = requests.get(url, timeout=10)
        except requests.exceptions.RequestException:
            time.sleep(10)
            r = requests.get(url, timeout=30)
        image_target = r.content
        filename = url.split('/')[-1][:-4]
        output_image_path = os.path.join(FLAGS.output, 'image/{}.png'.format(filename))
        output_annot_path = os.path.join(FLAGS.output, 'image_annotation/{}.xml'.format(filename))
        if os.path.exists(output_image_path):
            os.remove(output_image_path)
        open(output_image_path, 'wb').write(image_target)

        doc = minidom.getDOMImplementation()
        dom = doc.createDocument(None, 'annotation', None)
        root_node = dom.documentElement
        filename_node = dom.createElement('filename')
        root_node.appendChild(filename_node)
        filename_text = dom.createTextNode(filename + '.png')
        filename_node.appendChild(filename_text)

        regions = json_file[image_id]['regions']
        for region in regions:
            cls = 'armor_' + region['region_attributes']['color']
            xmin = float(region['shape_attributes']['x'])
            ymin = float(region['shape_attributes']['y'])
            xmax = xmin + float(region['shape_attributes']['width'])
            ymax = ymin + float(region['shape_attributes']['height'])

            object_node = dom.createElement('object')
            root_node.appendChild(object_node)
            name_node = dom.createElement('name')
            object_node.appendChild(name_node)
            name_text = dom.createTextNode(cls)
            name_node.appendChild(name_text)

            bndbox_node = dom.createElement('bndbox')
            object_node.appendChild(bndbox_node)
            xmin_node = dom.createElement('xmin')
            ymin_node = dom.createElement('ymin')
            xmax_node = dom.createElement('xmax')
            ymax_node = dom.createElement('ymax')
            bndbox_node.appendChild(xmin_node)
            bndbox_node.appendChild(ymin_node)
            bndbox_node.appendChild(xmax_node)
            bndbox_node.appendChild(ymax_node)
            xmin_text = dom.createTextNode(str(xmin))
            ymin_text = dom.createTextNode(str(ymin))
            xmax_text = dom.createTextNode(str(xmax))
            ymax_text = dom.createTextNode(str(ymax))
            xmin_node.appendChild(xmin_text)
            ymin_node.appendChild(ymin_text)
            xmax_node.appendChild(xmax_text)
            ymax_node.appendChild(ymax_text)

        if os.path.exists(output_annot_path):
            os.remove(output_annot_path)
        output = open(output_annot_path, 'w')
        dom.writexml(output, addindent='\t', newl='\n', encoding='utf-8')
        output.close()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
