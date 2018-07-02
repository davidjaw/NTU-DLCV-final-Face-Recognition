import tensorflow as tf
import tensorflow.contrib.slim as slim
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
    parser.add_argument('--batch_size', type=int, default=100, help='Number of instance in each training batch')
    parser.add_argument('--target_epoch', type=int, default=500, help='Target training epoch')
    parser.add_argument('--load', action='store_true', help='Either to load pre-train weights or not')
    parser.add_argument('--optim_type', type=str, default='adam', help='the type of optimizer')
    parser.add_argument('--finetune_level', type=int, default=2,
                        help='0: without data augmentation(DA), 1: with DA, 2: with seaweed augmentation')

    return parser.parse_args()


def main(args):
    print(args)

    LOG_STEP = 50
    SAVE_STEP = 500
    LOG_ALL_TRAIN_PARAMS = False
    LEARNING_RATE = 1e-4 / args.finetune_level if args.finetune_level > 1 else 1e-4

    with tf.variable_scope('Data_Generator'):
        data_reader = DataReader(
            data_path=args.data_path
        )
        train_x, train_y = data_reader.get_instance(batch_size=args.batch_size, mode='train', augmentation_level=args.finetune_level)
        valid_x, valid_y = data_reader.get_instance(batch_size=args.batch_size, mode='valid')
        class_num = len(data_reader.dict_class.keys())

    network = model.StudentNetwork(len(data_reader.dict_class.keys()))
    logits, prelogits = network.build_network(train_x, reuse=False, is_train=True)
    v_logits, v_prelogits = network.build_network(valid_x, reuse=True, is_train=True, dropout_keep_prob=1)

    use_center, use_pln, use_triplet, use_him = [False for _ in range(4)]
    pln_factor = center_factor = 0
    if args.finetune_level == 2:
        use_center, use_pln, use_triplet, use_him = [True for _ in range(4)]
        with tf.variable_scope('Output'):
            embed = slim.fully_connected(prelogits, 128, tf.identity, scope='Embedding')
            v_embed = slim.fully_connected(v_prelogits, 128, tf.identity, reuse=True, scope='Embedding')
        pln_factor = center_factor = 1e-4
    else:
        embed = v_embed = None
        if args.finetune_level == 1:
            use_center = use_pln = True
            pln_factor = center_factor = 1e-5

    loss_func = utils.LossFunctions(
        prelogit_norm_factor=pln_factor,
        center_loss_factor=center_factor,
    )
    loss, accu = loss_func.calculate_loss(logits, train_y, prelogits, class_num, use_center_loss=use_center,
                                          embed=embed, use_triplet_loss=use_triplet, use_prelogits_norm=use_pln,
                                          use_hard_instance_mining=use_him, scope_name='Training')
    _, v_accu = loss_func.calculate_loss(v_logits, valid_y, v_prelogits, class_num, use_center_loss=use_center,
                                         embed=v_embed, use_triplet_loss=use_triplet, use_prelogits_norm=use_pln,
                                         use_hard_instance_mining=use_him, scope_name='Validation')

    if args.optim_type == 'adam':
        optim = tf.train.AdamOptimizer(LEARNING_RATE)
    elif args.optim_type == 'adagrad':
        optim = tf.train.AdagradOptimizer(LEARNING_RATE)
    else:
        optim = tf.train.GradientDescentOptimizer(LEARNING_RATE)
    # get batch normalization parameters
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optim.minimize(loss)

    train_params = list(filter(lambda x: 'Adam' not in x.op.name and 'SqueezeNeXt' in x.op.name, tf.contrib.slim.get_variables()))
    saver = tf.train.Saver(var_list=train_params)

    if LOG_ALL_TRAIN_PARAMS:
        for i in train_params:
            tf.summary.histogram(i.op.name, i)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    if args.load:
        saver.restore(sess, args.weight_path + 'student.ckpt')

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
            np_loss, np_accu, np_v_accu, s = sess.run([loss, accu, v_accu, merged])
            train_writer.add_summary(s, step)
            print('======================= Step {} ====================='.format(step))
            print('[Log file saved] {:.2f} secs for one step'.format(time_cost))
            print('Current loss: {:.2f}, train accu: {:.2f}%, valid accu: {:.2f}%'.format(np_loss, np_accu, np_v_accu))
            start_time = time.time()

        if step % SAVE_STEP == 0:
            saver.save(sess, args.weight_path + 'student.ckpt', step)
            print('[Weights saved] weights saved at {}'.format(args.weight_path + 'student'))

        step += 1

    coord.request_stop()
    coord.join(threads)


if __name__ == '__main__':
    main(get_args())

