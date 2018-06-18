import tensorflow as tf
import model
from data_reader import TestDataReader, DataReader
import argparse
import os


def get_args():
    parser = argparse.ArgumentParser(description='Face recognition')
    parser.add_argument('--data_path', type=str, default='./dlcv_final_2_dataset/', help='path to dataset folder')
    parser.add_argument('--test_data_path', type=str, default='./dlcv_final_2_dataset/test/', help='path to test folder')
    parser.add_argument('--weight_path', type=str, default='./weight/', help='path to store/read weights')
    parser.add_argument('--model_name', type=str, default='teacher.ckpt-224000', help='filename of trained model')
    parser.add_argument('--batch_size', type=int, default=100, help='size of training batch')
    parser.add_argument('--out_path', type=str, default='./out/', help='path to output file')
    return parser.parse_args()


def main(args):
    print(args)
    with tf.variable_scope('Data_Generator'):
        test_data_reader = TestDataReader(data_path=args.test_data_path)
        data_reader = DataReader(data_path=args.data_path)
        test_x, test_num = test_data_reader.get_instance(batch_size=args.batch_size)

    network = model.TeacherNetwork()
    logits, net_dict = network.build_network(test_x, class_num=len(data_reader.dict_class.keys()), reuse=False, is_train=False, dropout=1)
    pred = tf.nn.softmax(logits, -1)
    pred = tf.argmax(pred, -1, output_type=tf.int32)

    train_params = tf.contrib.slim.get_variables()
    saver = tf.train.Saver(var_list=train_params)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    saver.restore(sess, args.weight_path + args.model_name)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)
    tf.Graph().finalize()

    if not os.path.exists(args.out_path):
        os.mkdir(args.out_path)
    out_f = open(args.out_path + 'out.txt', 'w')
    out_f.write('id,ans\n')
    instance_cnt = 0
    step = 0
    while step * args.batch_size < test_num:
        step += 1

        np_pred = sess.run(pred)
        for i in range(np_pred.shape[0]):
            out_f.write('{},{}\n'.format(instance_cnt + 1, data_reader.dict_class[np_pred[i]]))
            instance_cnt += 1
            if instance_cnt >= test_num:
                break

    out_f.close()
    coord.request_stop()
    coord.join(threads)


if __name__ == '__main__':
    main(get_args())

