import matplotlib
import matplotlib.pyplot as plt
import math
import numpy as np
import tensorflow as tf
import time
from datasets import dataset_utils

slim = tf.contrib.slim

def regression_model(inputs, is_training=True, scope='deep_regression'):
    with tf.variable_scope(scope, 'deep_regression', [inputs]):
        end_points = {}
        with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.relu, weights_regularizer=slim.l2_regularizer(0.01)):
            net = slim.fully_connected(inputs, 32, scope='fc1')
            end_points['fc1'] = net

            net = slim.dropout(net, 0.8, is_training=is_training)

            net = slim.fully_connected(net, 16, scope='fc2')
            end_points['fc2'] = net

            predictions = slim.fully_connected(net, 1, activation_fn=None, scope='prediction')
            end_points['out'] = predictions

            return predictions, end_points

def produce_batch(batch_size, noise = 0.3):
    xs = np.random.random(size=[batch_size, 1]) * 10
    ys = np.sin(xs) + 5 + np.random.normal(size=[batch_size, 1], scale=noise)
    return [xs.astype(np.float32), ys.astype(np.float32)]

def test_layer():
    with tf.Graph().as_default():
        inputs = tf.placeholder(tf.float32, shape=(None, 1))
        outputs = tf.placeholder(tf.float32, shape=(None, 1))

        predictions, end_points = regression_model(inputs)

        print 'Layers'
        for k, v in end_points.iteritems():
            print 'name = {}, shape = {}'.format(v.name, v.get_shape())

        print '\n'
        print 'Parameters'
        for v in slim.get_model_variables():
            print 'name = {}, shape = {}'.format(v.name, v.get_shape())

def convert_data_to_tensor(x, y):
    inputs = tf.constant(x)
    inputs.set_shape([None, 1])
    outputs = tf.constant(y)
    outputs.set_shape([None, 1])
    return inputs, outputs

def regression_one_loss():
    cpkt_dir = '/tmp/regression_model/'

    with tf.Graph().as_default():
        tf.logging.set_verbosity(tf.logging.INFO)

        x_train, y_train = produce_batch(200)
        inputs, targets = convert_data_to_tensor(x_train, y_train)

        predictions, nodes = regression_model(inputs, is_training=True)

        loss = slim.losses.mean_squared_error(predictions, targets)
        total_loss = slim.losses.get_total_loss()

        optimizer = tf.train.AdamOptimizer(learning_rate=0.005)
        train_op = slim.learning.create_train_op(total_loss, optimizer)

        final_loss = slim.learning.train(train_op, logdir=cpkt_dir, number_of_steps=10000, save_summaries_secs=5, log_every_n_steps=500)

    print("Finished training, Last batch loss:", final_loss)
    print("Checkpoint saved in %s" % cpkt_dir)

def regression_one_eval():
    cpkt_dir = '/tmp/regression_model/'

    with tf.Graph().as_default():
        x_test, y_test = produce_batch(200)
        inputs, targets = convert_data_to_tensor(x_test, y_test)

        predictions, end_points = regression_model(inputs, is_training=False)

        sv = tf.train.Supervisor(logdir=cpkt_dir)
        with sv.managed_session() as sess:
            inputs, predictions, targets = sess.run([inputs, predictions, targets])

    plt.scatter(inputs, targets, c='r')
    plt.scatter(inputs, predictions, c='b')
    plt.title('red=true, blue=predicted')
    plt.show()

def regression_one_eval_printfinal():
    cpkt_dir = '/tmp/regression_model/'
    with tf.Graph().as_default():
        x_test, y_test = produce_batch(200)
        inputs, targets = convert_data_to_tensor(x_test, y_test)
        predictions, end_points = regression_model(inputs, is_training=False)
        names_to_value_nodes, names_to_update_nodes = slim.metrics.aggregate_metric_map({
            'Mean Squared Error': slim.metrics.streaming_mean_squared_error(predictions, targets),
            'Mean Absolute Error': slim.metrics.streaming_mean_absolute_error(predictions, targets)
        })

        sv = tf.train.Supervisor(logdir=cpkt_dir)
        with sv.managed_session() as sess:
            metric_values = slim.evaluation.evaluation(
                sess, num_evals=1, eval_op=names_to_update_nodes.values(), final_op=names_to_value_nodes.values())

        names_to_values = dict(zip(names_to_value_nodes.keys(), metric_values))
        for key, value in names_to_values.iteritems():
            print('%s : %f' % (key, value))

def regression_multiple_losses():
    cpkt_dir = '/tmp/regression_model_multi'
    with tf.Graph().as_default():
        tf.logging.set_verbosity(tf.logging.INFO)

        x_train, y_train = produce_batch(200)
        inputs, targets = convert_data_to_tensor(x_train, y_train)

        predictions, end_points = regression_model(inputs, is_training=True)

        mean_square_error = slim.losses.mean_squared_error(predictions, targets)
        absolute_difference_loss = slim.losses.absolute_difference(predictions, targets)

        regularization_loss = tf.add_n(slim.losses.get_regularization_losses())
        total_loss1 = mean_square_error + absolute_difference_loss + regularization_loss

        total_loss2 = slim.losses.get_total_loss(add_regularization_losses=True)

        init_op = tf.initialize_all_variables()

        optimizer = tf.train.AdamOptimizer(learning_rate=0.005)
        train_op = slim.learning.create_train_op(total_loss2, optimizer)

        final_loss = slim.learning.train(train_op, logdir=cpkt_dir, number_of_steps=10000, save_summaries_secs=5, log_every_n_steps=500)

    print('Finished training, Last batch loss:', final_loss)

flower_data_dir = '/tmp/flowers'
def makeDataSet_Flowers():
    from datasets import dataset_utils
    url = 'http://download.tensorflow.org/data/flowers.tar.gz'

    if not tf.gfile.Exists(flower_data_dir):
        tf.gfile.MakeDirs(flower_data_dir)

    dataset_utils.download_and_uncompress_tarball(url, flower_data_dir)


from datasets import flowers
def display_some_data():

    with tf.Graph().as_default():
        dataset = flowers.get_split('train', flower_data_dir)
        data_provider = slim.dataset_data_provider.DatasetDataProvider(dataset, common_queue_capacity=32, common_queue_min = 1)
        image, label = data_provider.get(['image', 'label'])
        with tf.Session() as sess:
            with slim.queues.QueueRunners(sess):
                for i in xrange(4):
                    np_image, np_label = sess.run([image, label])
                    height, width, _ = np_image.shape
                    class_name = name = dataset.labels_to_names[np_label]

                    plt.figure()
                    plt.imshow(np_image)
                    plt.title('%s, %d x %d' % (name, height, width))
                    plt.axis('off')
                    plt.show()

def my_cnn(images, num_classes, is_training):
    with slim.arg_scope([slim.max_pool2d], kernel_size=[3, 3], stride=2):
        net = slim.conv2d(images, 64, [5, 5])
        net = slim.max_pool2d(net)
        net = slim.conv2d(net, 64, [5, 5])
        net = slim.max_pool2d(net)
        net = slim.flatten(net)
        net = slim.fully_connected(net, 192)
        net = slim.fully_connected(net, num_classes, activation_fn = None)
        return net

def cnn_prediction():
    with tf.Graph().as_default():
        batch_size, height, width, channels = 3, 28, 28, 3
        images = tf.random_uniform([batch_size, height, width, channels], maxval = 1)

        num_classes = 10
        logits = my_cnn(images, num_classes, is_training=True)
        probabilities = tf.nn.softmax(logits)
        init_op = tf.initialize_all_variables()
        with tf.Session() as sess:
            sess.run(init_op)
            probabilities = sess.run(probabilities)
    print('Probabilities Shape: ')
    print(probabilities.shape)

    print('Probabilities: ')
    print(probabilities)

    print('Summing across all classes (shild equeal 1)"')
    print(np.sum(probabilities, 1))

from preprocessing import inception_preprocessing
def load_batch(dataset, batch_size=8, height=299, width=299, is_training=False):
    data_provider = slim.dataset_data_provider.DatasetDataProvider(dataset, common_queue_capacity=32, common_queue_min=8)
    image_raw, label = data_provider.get(['image', 'label'])

    image = inception_preprocessing.preprocess_image(image_raw, height, width, is_training=is_training)
    image_raw = tf.expand_dims(image_raw, 0)
    image_raw = tf.image.resize_images(image_raw, [height, width])
    image_raw = tf.squeeze(image_raw)

    images, images_raw, labels = tf.train.batch([image, image_raw, label], batch_size=batch_size, num_threads=1, capacity=2*batch_size)
    return images, images_raw, labels

train_dir = '/tmp/tfslim_model/'
def flower_train():
    with tf.Graph().as_default():
        tf.logging.set_verbosity(tf.logging.INFO)

        dataset = flowers.get_split('train', flower_data_dir)
        images, _, labels = load_batch(dataset)

        logits = my_cnn(images, num_classes=dataset.num_classes, is_training=True)

        one_hot_labels = slim.one_hot_encoding(labels, dataset.num_classes)
        slim.losses.softmax_cross_entropy(logits, one_hot_labels)
        total_loss = slim.losses.get_total_loss()

        tf.scalar_summary('losses/Total Loss', total_loss)

        optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
        train_op = slim.learning.create_train_op(total_loss, optimizer)

        final_loss = slim.learning.train(train_op, logdir=train_dir, number_of_steps=100, save_summaries_secs=1)

    print('Finished training... Final batch loss %d' % final_loss)

def flower_eval():
    tf.logging.set_verbosity(tf.logging.INFO)

    dataset = flowers.get_split('train', flower_data_dir)
    images, _, labels = load_batch(dataset)

    logits = my_cnn(images, num_classes=dataset.num_classes, is_training=False)
    predictions = tf.argmax(logits, 1)

    names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
        'eval/Accuray': slim.metrics.streaming_accuracy(predictions, labels),
        'eval/Recall@5': slim.metrics.streaming_recall_at_k(logits, labels, 5)
    })

    print('Running evaluation loop...')
    checkpoint_path = tf.train.latest_checkpoint(train_dir)
    metric_values = slim.evaluation.evaluate_once(master='', checkpoint_path=checkpoint_path, logdir=train_dir, eval_op=names_to_updates.values(), final_op=names_to_values.values())

    names_to_values = dict(zip(names_to_values.keys(), metric_values))
    for name in names_to_values:
        print('%s: %f' % (name, names_to_values[name]))

def pretrained_test():
    checkpoint_dir = '/tmp/checkpoints'

    def download_pretrain():
        url = 'http://download.tensorflow.org/models/inception_v1_2016_08_28.tar.gz'

        if not tf.gfile.Exists(checkpoint_dir):
            tf.gfile.MakeDirs(checkpoint_dir)

        dataset_utils.download_and_uncompress_tarball(url, checkpoint_dir)

    def apply_pretrained_model():
        import numpy as np
        import os
        import tensorflow as tf
        import urllib2
        from datasets import imagenet
        from nets import inception
        from preprocessing import inception_preprocessing

        slim = tf.contrib.slim

        batch_size = 3
        image_size = inception.inception_v1.default_image_size

        with tf.Graph().as_default():
            url = 'https://upload.wikimedia.org/wikipedia/commons/7/70/EnglishCockerSpaniel_simon.jpg'
            image_string = urllib2.urlopen(url).read()
            image = tf.image.decode_jpeg(image_string, channels=3)
            processed_image = inception_preprocessing.preprocess_image(image, image_size, image_size, is_training=False)
            processed_images = tf.expand_dims(processed_image, 0)

            with slim.arg_scope(inception.inception_v1_arg_scope()):
                logits, _ = inception.inception_v1(processed_images, num_classes=1001, is_training=False)
            probabilities = tf.nn.softmax(logits)

            init_fn = slim.assign_from_checkpoint_fn(os.path.join(checkpoint_dir, 'inception_v1.ckpt'), slim.get_model_variables('InceptionV1'))

            with tf.Session() as sess:
                init_fn(sess)
                np_image, probabilities = sess.run([image, probabilities])
                probabilities = probabilities[0, 0:]
                sorted_inds = [i[0] for i in sorted(enumerate(-probabilities), key=lambda x:x[1])]

            plt.figure()
            plt.imshow(np_image.astype(np.uint8))
            plt.axis('off')
            plt.show()

            names = imagenet.create_readable_names_for_imagenet_labels()
            for i in range(5):
                index = sorted_inds[i]
                print('Probabity %0.2f%% => [%s]' % (probabilities[index], names[index]))

    train_dir = '/tmp/inception_finetuned/'
    def adapt_pretrain_imagenet_to_flower():
        import os
        from datasets import flowers
        from nets import inception
        from preprocessing import inception_preprocessing

        slim = tf.contrib.slim
        image_size = inception.inception_v1.default_image_size

        def get_init_fn():
            checkpoint_exclude_scopes=['InceptionV1/Logits', 'InceptionV1/AuxLogits']
            exclusions = [scope.strip() for scope in checkpoint_exclude_scopes]
            variables_to_restore = []
            for var in slim.get_model_variables():
                excluded = False
                for exclusion in exclusions:
                    if var.op.name.startswith(exclusion):
                        excluded = True
                        break
                if not excluded:
                    variables_to_restore.append(var)

            return slim.assign_from_checkpoint_fn(os.path.join(checkpoint_dir, 'inception_v1.ckpt'), variables_to_restore)

        with tf.Graph().as_default():
            tf.logging.set_verbosity(tf.logging.INFO)
            dataset = flowers.get_split('train', flower_data_dir)
            images, _, labels = load_batch(dataset, height=image_size, width=image_size)

            with slim.arg_scope(inception.inception_v1_arg_scope()):
                logits, _ = inception.inception_v1(images, num_classes=dataset.num_classes, is_training=True)

                one_hot_labels = slim.one_hot_encoding(labels, dataset.num_classes)
                slim.losses.softmax_cross_entropy(logits, one_hot_labels)
                total_loss = slim.losses.get_total_loss()

                tf.scalar_summary('losses/Total Loss', total_loss)

                optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
                train_op = slim.learning.create_train_op(total_loss, optimizer)

                final_loss = slim.learning.train(train_op, logdir=train_dir, init_fn=get_init_fn(), number_of_steps=2)

        print("Finished training. Las batch loss %f" % final_loss)

    def eval_adapt_pretrained_to_flower():
        import numpy as np
        import tensorflow as tf
        from datasets import flowers
        from nets import inception

        slim = tf.contrib.slim

        image_size = inception.inception_v1.default_image_size
        batch_size = 8

        with tf.Graph().as_default():
            tf.logging.set_verbosity(tf.logging.INFO)
            dataset = flowers.get_split('train', flower_data_dir)
            images, images_raw, labels = load_batch(dataset, height=image_size, width=image_size)

            with slim.arg_scope(inception.inception_v1_arg_scope()):
                logits, _ = inception.inception_v1(images, num_classes=dataset.num_classes, is_training=True)
                probabilities = tf.nn.softmax(logits)

                checkpoint_path = tf.train.latest_checkpoint(train_dir)
                init_fn = slim.assign_from_checkpoint_fn(checkpoint_path, slim.get_variables_to_restore())
                with tf.Session() as sess:
                    with slim.queues.QueueRunners(sess):
                        sess.run(tf.initialize_local_variables())
                        init_fn(sess)
                        np_probabilities, np_images_raw, np_labels = sess.run([probabilities, images_raw, labels])

                        for i in xrange(batch_size):
                            image = np_images_raw[i, :, :, :]
                            true_label = np_labels[i]
                            predicted_label = np.argmax(np_probabilities[i, :])
                            predicted_name = dataset.labels_to_names[predicted_label]
                            true_name = dataset.labels_to_names[true_label]

                            plt.figure()
                            plt.imshow(image.astype(np.uint8))
                            plt.title('Ground Truth: [%s], Prediction [%s]' % (true_name, predicted_name))
                            plt.axis('off')
                            plt.show()


    #download_pretrain()
    #apply_pretrained_model()
    #adapt_pretrain_imagenet_to_flower()
    eval_adapt_pretrained_to_flower()

pretrained_test()
#makeDataSet_Flowers()
#display_some_data()
#cnn_prediction()

#flower_train()
#flower_eval()
exit()
x_train, y_train = produce_batch(200)
x_test, y_test = produce_batch(200)
plt.scatter(x_train, y_train)
plt.show()
