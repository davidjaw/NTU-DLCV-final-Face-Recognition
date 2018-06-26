import tensorflow as tf
import model
from data_reader import DataReader
import time
import argparse
import utils


def get_args():
    parser = argparse.ArgumentParser(description='Face recognition')
    parser.add_argument('--data_path', type=str, default='./dlcv_final_2_dataset/', help='path to dataset folder')
    parser.add_argument('--log_path', type=str, default='./log/', help='path to logfile folder')
    parser.add_argument('--weight_path', type=str, default='./log/', help='path to store/read weights')
    parser.add_argument('--batch_size', type=int, default=100, help='size of training batch')
    parser.add_argument('--target_epoch', type=int, default=500, help='size of training epoch')
    parser.add_argument('--load', action='store_true', help='Either to load pre-train weights or not')
    parser.add_argument('--optim_type', type=str, default='adam', help='the type of optimizer')
    parser.add_argument('--finetune_level', type=int, default=0,
                        help='0: without data augmentation(DA), 1: with DA, 2: with seaweed augmentation')

    return parser.parse_args()


def main(args):
    print(args)

    LOG_STEP = 250
    SAVE_STEP = 500
    LOG_ALL_TRAIN_PARAMS = False
    PRELOGIT_NORM_FACTOR = 0 if args.finetune_level < 2 else 1e-5
    CENTER_LOSS_FACTOR = 1e-5
    LEARNING_RATE = 1e-4 / args.finetune_level if args.finetune_level > 1 else 1e-3

    with tf.variable_scope('Data_Generator'):
        data_reader = DataReader(
            data_path=args.data_path
        )
        train_x, train_y = data_reader.get_instance(batch_size=args.batch_size, mode='train', augmentation_level=args.finetune_level)
        valid_x, valid_y = data_reader.get_instance(batch_size=args.batch_size * 2, mode='valid')

    network = model.TeacherNetwork()
    logits, net_dict = network.build_network(train_x, class_num=len(data_reader.dict_class.keys()), reuse=False, is_train=True)
    v_logits, v_net_dict = network.build_network(valid_x, class_num=len(data_reader.dict_class.keys()), reuse=True, is_train=True, dropout=1)

    with tf.variable_scope('compute_loss'):
        # Norm for the prelogits
        prelogits = net_dict['PreLogitsFlatten']
        eps = 1e-4
        prelogits_norm = tf.reduce_mean(tf.norm(tf.abs(prelogits) + eps, ord=1., axis=1)) * PRELOGIT_NORM_FACTOR

        # Center loss
        center_loss, _ = utils.center_loss(prelogits, train_y, .95, len(data_reader.dict_class.keys()))
        center_loss *= CENTER_LOSS_FACTOR

        # Cross Entropy loss
        train_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=train_y)
        valid_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=v_logits, labels=valid_y)
        if args.finetune_level == 2:
            train_loss = tf.reduce_mean(tf.where(tf.less(train_loss, tf.reduce_max(train_loss) * .7),
                                                 train_loss * .1,
                                                 train_loss))
            valid_loss = tf.reduce_mean(tf.where(tf.less(valid_loss, tf.reduce_max(valid_loss) * .7),
                                                 valid_loss * .1,
                                                 valid_loss))
        else:
            train_loss = tf.reduce_mean(train_loss)
            valid_loss = tf.reduce_mean(valid_loss)

        # Accuracy for tensorboard
        train_output = tf.argmax(tf.nn.softmax(logits, -1), -1, output_type=tf.int32)
        train_accu = tf.where(tf.equal(train_output, train_y), tf.ones_like(train_output), tf.zeros_like(train_output))
        train_accu = tf.reduce_sum(train_accu) / args.batch_size * 100

        valid_output = tf.argmax(tf.nn.softmax(v_logits, -1), -1, output_type=tf.int32)
        valid_accu = tf.where(tf.equal(valid_output, valid_y), tf.ones_like(valid_output), tf.zeros_like(valid_output))
        valid_accu = tf.reduce_sum(valid_accu) / args.batch_size * 100 / 2

    with tf.variable_scope('Summary'):
        tf.summary.histogram('logit_raw', logits)
        tf.summary.histogram('logit_softmax', train_output)

        tf.summary.scalar('train_loss', train_loss)
        tf.summary.scalar('train_accu', train_accu)
        tf.summary.scalar('valid_loss', valid_loss)
        tf.summary.scalar('valid_accu', valid_accu)

    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(LEARNING_RATE, global_step, 10000, 0.9, staircase=True)
    if args.optim_type == 'adam':
        optim = tf.train.AdamOptimizer(learning_rate)
    elif args.optim_type == 'adagrad':
        optim = tf.train.AdagradOptimizer(learning_rate)
    else:
        optim = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optim.minimize(train_loss + prelogits_norm + center_loss, global_step)

    train_params = list(filter(lambda x: 'Adam' not in x.op.name and 'Inception' in x.op.name,
                               tf.contrib.slim.get_variables()))
    saver = tf.train.Saver(var_list=train_params)

    if LOG_ALL_TRAIN_PARAMS:
        for i in train_params:
            tf.summary.histogram(i.op.name, i)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    if args.load:
        saver.restore(sess, args.weight_path + 'teacher.ckpt')

    train_writer = tf.summary.FileWriter(args.log_path, sess.graph)
    merged = tf.summary.merge_all(tf.GraphKeys.SUMMARIES)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    tf.Graph().finalize()
    start_time = time.time()
    step = 0
    while step * args.batch_size / len(data_reader.train_label) < args.target_epoch:
        _ = sess.run(train_op)

        if step % LOG_STEP == 0:
            time_cost = (time.time() - start_time) / LOG_STEP if step > 0 else 0
            loss, v_loss, accu, v_accu, s = sess.run([train_loss, valid_loss, train_accu, valid_accu, merged])
            train_writer.add_summary(s, step)
            print('======================= Step {} ====================='.format(step))
            print('[Log file saved] {:.2f} secs for one step'.format(time_cost))
            print('Current loss: {:.2f}, train accu: {:.2f}%, valid accu: {:.2f}%'.format(loss, accu, v_accu))
            start_time = time.time()

        if step % SAVE_STEP == 0:
            saver.save(sess, args.weight_path + 'teacher.ckpt', step)
            print('[Weights saved] weights saved at {}'.format(args.weight_path + 'teacher'))

        step += 1

    coord.request_stop()
    coord.join(threads)


if __name__ == '__main__':
    main(get_args())

