import tensorflow as tf
import model
from data_reader import DataReader
import time
import argparse


def get_args():
    parser = argparse.ArgumentParser(description='Face recognition')
    parser.add_argument('--data_path', type=str, default='./dlcv_final_2_dataset/', help='path to dataset folder')
    parser.add_argument('--log_path', type=str, default='./log/', help='path to logfile folder')
    parser.add_argument('--weight_path', type=str, default='./log/', help='path to store/read weights')
    parser.add_argument('--batch_size', type=int, default=100, help='size of training batch')
    parser.add_argument('--target_epoch', type=int, default=500, help='size of training epoch')

    return parser.parse_args()


def main(args):
    print(args)

    LOG_STEP = 50
    SAVE_STEP = 500

    with tf.variable_scope('Data_Generator'):
        data_reader = DataReader(
            data_path=args.data_path
        )
        train_x, train_y = data_reader.get_instance(batch_size=args.batch_size, mode='train')
        valid_x, valid_y = data_reader.get_instance(batch_size=args.batch_size, mode='valid')

    network = model.TeacherNetwork()
    logits, net_dict = network.build_network(train_x, class_num=len(data_reader.dict_class.keys()), reuse=False, is_train=True)
    v_logits, v_net_dict = network.build_network(valid_x, class_num=len(data_reader.dict_class.keys()), reuse=True, is_train=True)

    with tf.variable_scope('compute_loss'):
        train_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=train_y)
        valid_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=v_logits, labels=valid_y)
        train_loss = tf.reduce_mean(train_loss)
        valid_loss = tf.reduce_mean(valid_loss)

        train_output = tf.argmax(tf.nn.softmax(logits, -1), -1, output_type=tf.int32)
        train_accu = tf.where(tf.equal(train_output, train_y), tf.ones_like(train_output), tf.zeros_like(train_output))
        train_accu = tf.reduce_sum(train_accu) / args.batch_size * 100

        valid_output = tf.argmax(tf.nn.softmax(v_logits, -1), -1, output_type=tf.int32)
        valid_accu = tf.where(tf.equal(valid_output, valid_y), tf.ones_like(valid_output), tf.zeros_like(valid_output))
        valid_accu = tf.reduce_sum(valid_accu) / args.batch_size * 100

    with tf.variable_scope('Summary'):
        tf.summary.histogram('logit_raw', logits)
        tf.summary.histogram('logit_softmax', train_output)

        tf.summary.scalar('train_loss', train_loss)
        tf.summary.scalar('train_accu', train_accu)
        tf.summary.scalar('valid_loss', valid_loss)
        tf.summary.scalar('valid_accu', valid_accu)

    optim = tf.train.AdamOptimizer(1e-4)
    train_op = optim.minimize(train_loss)

    train_params = tf.contrib.slim.get_variables()
    saver = tf.train.Saver(var_list=train_params)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

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
            loss, accu, v_accu, s = sess.run([train_loss, train_accu, valid_accu, merged])
            train_writer.add_summary(s, step)
            print('======================= Step {} ====================='.format(step))
            print('[Log file saved] {:.2f} secs for one step'.format(time_cost))
            print('Current loss: {:.2f}, train accu: {:.2f}%, valid accu: {:.2f}%'.format(loss, accu, v_accu))
            start_time = time.time()

        if step % SAVE_STEP == 0:
            saver.save(sess, args.weight_path + 'teacher', step)
            print('[Weights saved] weights saved at {}'.format(args.weight_path + 'teacher'))

        step += 1

    coord.request_stop()
    coord.join(threads)


if __name__ == '__main__':
    main(get_args())

