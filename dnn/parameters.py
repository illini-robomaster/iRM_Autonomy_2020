import numpy as np

PARAM = {
    # img_size
    'size': 416, 
    # yolo anchors
    'yolo_anchors': np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45), (59, 119), (116, 90), (156, 198), (373, 326)], np.float32),
    'yolo_anchor_masks': np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]], np.float32),
    # yolo v3 tiny Anchors
    "yolo_tiny_anchors": np.array([(10, 14), (23, 27), (37, 58), (81, 82), (135, 169), (344, 319)], np.float32),
    "yolo_tiny_anchor_masks": np.array([[3, 4, 5], [0, 1, 2]], np.float32),
    # Cutoff Params
    'max_boxes': 100,
    'iou_threshold': 0.5,
    'score_threshold': 0.5,
    'class_number': 5,
    # Training params
    'batch_size': 8,
    'epoch': 100,
    'learning_rate': 1e-3,
    'workers': 12,
    # Data
    'train': np.array([
        'F:/RM/iRM_Autonomy_2020/dnn/data/roco/ROCO_8.tfrecords',
        'F:/RM/iRM_Autonomy_2020/dnn/data/roco/ROCO_6.tfrecords',
        'F:/RM/iRM_Autonomy_2020/dnn/data/roco/ROCO_7.tfrecords',
        'F:/RM/iRM_Autonomy_2020/dnn/data/purdue/train/shard-0.tfrecord',
        'F:/RM/iRM_Autonomy_2020/dnn/data/purdue/train/shard-1.tfrecord',
        'F:/RM/iRM_Autonomy_2020/dnn/data/purdue/train/shard-2.tfrecord',
    ]),
    'val': [
        'F:/RM/iRM_Autonomy_2020/dnn/data/roco/ROCO_5.tfrecords',
        'F:/RM/iRM_Autonomy_2020/dnn/data/purdue/validation/shard-0.tfrecord',
    ],
    # save model dir
    'save_dir': 'out',
    # ckpt dir, be sure to include last '/'
    'ckpt_dir': 'checkpoints/',
    # tensorboard dir
    'tensorboard_dir': 'log'
    # Transfer learning
}