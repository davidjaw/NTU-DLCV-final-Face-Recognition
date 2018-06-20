import tensorflow as tf
import model
from data_reader import DataReader
import argparse
import numpy as np


def get_args():
    parser = argparse.ArgumentParser(description='Face recognition')
    parser.add_argument('--data_path', type=str, default='./dlcv_final_2_dataset/', help='path to dataset folder')
    parser.add_argument('--weight_path', type=str, default='./weight/', help='path to store/read weights')
    parser.add_argument('--model_name', type=str, default='{}.ckpt', help='filename of trained model')
    parser.add_argument('--batch_size', type=int, default=100, help='size of training batch')
    parser.add_argument('--is_teacher', action='store_true', help='Either use the teacher network or student network')
    return parser.parse_args()


def main(args):
    if '{}' in args.model_name:
        args.model_name = args.model_name.format('teacher' if args.is_teacher else 'student')
    print(args)

    with tf.variable_scope('Data_Generator'):
        data_reader = DataReader(
            data_path=args.data_path
        )
        valid_x, valid_y = data_reader.get_instance(batch_size=args.batch_size, mode='valid')
        valid_num = len(data_reader.valid_img_path)

    if args.is_teacher:
        network = model.TeacherNetwork()
        v_logits, v_net_dict = network.build_network(valid_x, class_num=len(data_reader.dict_class.keys()), reuse=False,
                                                     is_train=False, dropout=1)
    else:
        network = model.StudentNetwork(len(data_reader.dict_class.keys()))
        v_logits, v_pre_logit = network.build_network(valid_x, False, False)
    v_pred = tf.nn.softmax(v_logits, -1)
    v_pred = tf.argmax(v_pred, -1, output_type=tf.int32)

    cnt = tf.equal(v_pred, valid_y)

    train_params = tf.contrib.slim.get_variables()
    saver = tf.train.Saver(var_list=train_params)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    saver.restore(sess, args.weight_path + args.model_name)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    tf.Graph().finalize()
    step = 0
    correct = []
    while step * args.batch_size < valid_num:
        np_cnt = sess.run(cnt)
        for i in range(np_cnt.shape[0]):
            if len(correct) < valid_num:
                correct.append(1 if np_cnt[i] else 0)
        step += 1

    accuracy = np.sum(correct) / valid_num
    print('================================================')
    print('[{}] Accuracy on validation set: {:.2f}'.format(args.model_name, accuracy * 100))
    print('================================================')

    coord.request_stop()
    coord.join(threads)


if __name__ == '__main__':
    main(get_args())

