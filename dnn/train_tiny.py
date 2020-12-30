import tensorflow as tf
import numpy as np
from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
    EarlyStopping,
    ModelCheckpoint,
    TensorBoard
)
from dnn.model.yolov3_tiny import YoloV3Tiny, YoloLoss
from dnn.utils.dataLoader import dataLoader
from dnn.utils.yolo_encoder import yoloEncoder
from dnn.parameters import PARAM
from dnn.utils.mem import tf_set_memory_growth

def main():
    # Get params
    size = PARAM['size']
    anchors = PARAM['yolo_tiny_anchors']
    anchor_masks = PARAM['yolo_tiny_anchor_masks']
    num_cls = PARAM['class_number']
    train_dir = PARAM['train']
    val_dir = PARAM['val']
    lr = PARAM['learning_rate']
    epoch = PARAM['epoch']
    batch_size = PARAM['batch_size']
    # Load training and validation data
    loader_train = dataLoader(size, train = True)
    loader_test = dataLoader(size, train = False)
    train_data = loader_train(train_dir)
    val_data = loader_test(val_dir)
    # encode data
    encoder = yoloEncoder(size, anchors, anchor_masks)
    train_data = train_data.shuffle(buffer_size = 512)
    train = encoder(train_data)
    val = encoder(val_data)
    # batch and prefetch
    train = train.batch(batch_size)
    train = train.prefetch(buffer_size = tf.data.experimental.AUTOTUNE)
    val = val.batch(batch_size)
    val = val.prefetch(buffer_size = tf.data.experimental.AUTOTUNE)
    # initialize model, loss function and optimizer
    model = YoloV3Tiny(size, classes=num_cls, training=True)
    loss = [YoloLoss(anchors[mask], classes=num_cls) 
            for mask in np.array(anchor_masks, dtype=np.int8)] #cast to int 
    optimizer = tf.keras.optimizers.Adam(lr = lr)
    
    # compile
    model.compile(optimizer=optimizer, loss=loss, run_eagerly=True)

    # setup callbacks
    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1),
        EarlyStopping(monitor='val_loss', min_delta=0, patience=6, verbose=1),
        ModelCheckpoint('checkpoints/yolov3_train_{epoch}.tf',
                        verbose=1, save_weights_only=True),
    ]
    
    model.fit(
        train,
        epochs = epoch,
        callbacks = callbacks,
        validation_data = val,
        workers = 12,
        use_multiprocessing = True
    )

    model.save(PARAM['save_dir'])

if __name__ == '__main__':
    # set memory growth
    tf_set_memory_growth()
    main()

