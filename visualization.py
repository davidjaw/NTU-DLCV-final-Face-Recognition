import tensorflow as tf
import model
from data_reader import DataReader
import argparse
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import cv2


def get_args():
    parser = argparse.ArgumentParser(description='Face recognition')
    parser.add_argument('--data_path', type=str, default='./dlcv_final_2_dataset/test/', help='path to the testset folder')
    parser.add_argument('--weight_path', type=str, default='./weight/', help='path to store/read weights')
    parser.add_argument('--model_name', type=str, default='teacher.ckpt',
                        help='filename of trained model\'s weightfile')
    parser.add_argument('--batch_size', type=int, default=100, help='Number of instance in each testing batch')
    parser.add_argument('--is_teacher', action='store_true', help='Either use the teacher network or student network')
    parser.add_argument('--light', action='store_true', help='Either to use light model or not')
    return parser.parse_args()


def main(args):
    print(args)
    with tf.variable_scope('Data_Generator'):
        data_reader = DataReader(data_path=None)
        test_x_p = tf.placeholder(tf.float32, [None, 86, 105, 3])
        test_x = tf.image.resize_bilinear(test_x_p, [218, 178])
        test_x = (test_x / 255. - .5) * 2

    if args.is_teacher:
        network = model.TeacherNetwork()
        logits, net_dict = network.build_network(test_x, class_num=len(data_reader.dict_class.keys()), reuse=False,
                                                 is_train=False, dropout=1)
        prelogits = net_dict['PreLogitsFlatten']
    else:
        network = model.StudentNetwork(len(data_reader.dict_class.keys()))
        logits, prelogits = network.build_network(test_x, False, False, light=args.light)

    train_params = tf.contrib.slim.get_variables()
    saver = tf.train.Saver(var_list=train_params)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    saver.restore(sess, args.weight_path + args.model_name)

    tf.Graph().finalize()

    class_num = 20
    np_train_img_path = np.asarray(data_reader.train_img_path)
    np_train_label = np.asarray(data_reader.train_label)
    np_total_feature = np.zeros([class_num, 16, 1792], np.float32)

    colors = ['aqua', 'azure', 'fuchsia', 'black', 'blue', 'brown', 'chartreuse', 'coral', 'cyan', 'darkgreen', 'indigo', 'lime', 'lightgreen', 'red', 'maroon', 'sienna', 'tan', 'teal', 'tomato', 'pink', 'olive']
    for i in range(class_num):
        np_class_n = np_train_img_path[np_train_label == i][:16]
        np_imgs = np.asarray([cv2.imread(x) + 0. for x in np_class_n.tolist()])
        h = int(218 / 5)
        w = int(178 / 5)
        np_imgs = np_imgs[:, h*2:h*4, w:w*4, :]
        np_class_features = sess.run(prelogits, feed_dict={test_x_p: np_imgs})
        np_total_feature[i] = np_class_features
    np_total_feature = np.reshape(np_total_feature, [16 * class_num, 1792])
    tsne_embed = TSNE(n_components=2, random_state=92).fit_transform(np_total_feature)
    tsne_embed = np.reshape(tsne_embed, [class_num, 16, 2])
    plt.style.use('default')
    for i in range(class_num):
        class_embbed_feature = tsne_embed[i, :, :]
        plt.plot(class_embbed_feature[:, 0], class_embbed_feature[:, 1], 'o', markersize=2.2, color=colors[i])
    plt.title('tSNE for pre-logits space')
    plt.savefig('out/out_{}.jpg'.format(args.model_name))


if __name__ == '__main__':
    main(get_args())
