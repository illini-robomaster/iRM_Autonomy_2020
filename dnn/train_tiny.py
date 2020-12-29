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
    
    ## Eager training, used for debugging, will remove in the future
    avg_loss = tf.keras.metrics.Mean('loss', dtype=tf.float32)
    avg_val_loss = tf.keras.metrics.Mean('val_loss', dtype=tf.float32)
    for epoch in range(1, epoch + 1):
        for batch, (images, labels) in enumerate(train):
            with tf.GradientTape() as tape:
                outputs = model(images, training=True)
                regularization_loss = tf.reduce_sum(model.losses)
                pred_loss = []
                for output, label, loss_fn in zip(outputs, labels, loss):
                    pred_loss.append(loss_fn(label, output))
                total_loss = tf.reduce_sum(pred_loss) + regularization_loss

            grads = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(
                zip(grads, model.trainable_variables))

            print("Epoch {}.{}, Train Loss: {}, {}".format(
                epoch, batch, total_loss.numpy(),
                list(map(lambda x: np.sum(x.numpy()), pred_loss))))
            avg_loss.update_state(total_loss)
            model.save(PARAM['save_dir'])

        for batch, (images, labels) in enumerate(val):
            outputs = model(images)
            regularization_loss = tf.reduce_sum(model.losses)
            pred_loss = []
            for output, label, loss_fn in zip(outputs, labels, loss):
                pred_loss.append(loss_fn(label, output))
            total_loss = tf.reduce_sum(pred_loss) + regularization_loss

            print("Epoch {}.{}, Validation Loss: {}, {}".format(
                epoch, batch, total_loss.numpy(),
                list(map(lambda x: np.sum(x.numpy()), pred_loss))))
            avg_val_loss.update_state(total_loss)

        print("Epoch {} Summary, train: {}, val: {}".format(
            epoch,
            avg_loss.result().numpy(),
            avg_val_loss.result().numpy()))

        avg_loss.reset_states()
        avg_val_loss.reset_states()
    
    # # compile
    # model.compile(optimizer=optimizer, loss=loss, run_eagerly=True)

    # # setup callbacks
    # callbacks = [
    #     ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1),
    #     EarlyStopping(monitor='val_loss', min_delta=0, patience=6, verbose=1),
    #     ModelCheckpoint('checkpoints/yolov3_train_{epoch}.tf',
    #                     verbose=1, save_weights_only=True),
    # ]
    
    # model.fit(
    #     train,
    #     epochs = epoch,
    #     callbacks = callbacks,
    #     validation_data = val
    #     #workers = 12,
    #     #use_multiprocessing = True
    # )
    # model.save(PARAM['save_dir'])

if __name__ == '__main__':
    # set memory growth
    tf_set_memory_growth()
    main()

