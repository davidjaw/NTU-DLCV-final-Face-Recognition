import tensorflow as tf
import tensorflow.contrib.slim as slim
import model
from data_reader import DataReader
import time
import argparse


def get_args():
    parser = argparse.ArgumentParser(description='Face recognition')
    parser.add_argument('--data_path', type=str, default='./dlcv_final_2_dataset/', help='path to dataset folder')
    parser.add_argument('--log_path', type=str, default='./log/', help='path to logfile folder')
    parser.add_argument('--weight_path', type=str, default='./log/', help='path to store/read trained weights')
    parser.add_argument('--t_weight_path', type=str, default='./weight/',
                        help='path to trained teacher network\'s weightfile')
    parser.add_argument('--t_model_name', type=str, default='teacher.ckpt',
                        help='filename of the teacher network\'s weightfile')
    parser.add_argument('--batch_size', type=int, default=100, help='Number of instance in each training batch')
    parser.add_argument('--target_epoch', type=int, default=500, help='Target training epoch')
    parser.add_argument('--load', action='store_true', help='Either to load pre-train weights or not')
    parser.add_argument('--light', action='store_true', help='Either to use light model or not')
    parser.add_argument('--optim_type', type=str, default='adam', help='the type of optimizer')
    parser.add_argument('--finetune_level', type=int, default=2,
                        help='0: without data augmentation(DA), 1: with DA, 2: with seaweed augmentation')

    return parser.parse_args()


def main(args):
    print(args)

    LOG_STEP = 250
    SAVE_STEP = 500
    LOG_ALL_TRAIN_PARAMS = False
    MODEL_NAME = 'TS.ckpt'
    LEARNING_RATE = 1e-4 / args.finetune_level if args.finetune_level > 1 else 1e-3

    with tf.variable_scope('Data_Generator'):
        data_reader = DataReader(
            data_path=args.data_path
        )
        train_x, train_y = data_reader.get_instance(batch_size=args.batch_size, mode='train', augmentation_level=args.finetune_level)
        valid_x, valid_y = data_reader.get_instance(batch_size=args.batch_size, mode='valid')
        class_num = len(data_reader.dict_class.keys())

    teacher_net = model.TeacherNetwork()
    teacher_logits, teacher_dict = teacher_net.build_network(train_x, class_num, reuse=False, is_train=False)
    teacher_logits = tf.stop_gradient(teacher_logits)

    network = model.StudentNetwork(len(data_reader.dict_class.keys()))
    logits, prelogits = network.build_network(train_x, reuse=False, is_train=True, light=args.light)
    v_logits, v_prelogits = network.build_network(valid_x, reuse=True, is_train=True, dropout_keep_prob=1, light=args.light)

    with tf.variable_scope('SqueezeNeXt/Embedding'):
        t_prelogits = teacher_dict['PreLogitsFlatten']
        s_embed = slim.fully_connected(prelogits, t_prelogits.get_shape().as_list()[-1], activation_fn=tf.identity)
        s_embed_pred = slim.fully_connected(s_embed, len(data_reader.dict_class.keys()), activation_fn=tf.identity)

    with tf.variable_scope('compute_loss'):
        # Euclidean embedding loss
        euclidean_loss = tf.squared_difference(s_embed, t_prelogits)
        euclidean_loss = tf.reduce_mean(euclidean_loss)

        # soft label loss
        with tf.variable_scope('soft_CE'):
            soft_CE = lambda x, y: tf.reduce_mean(tf.reduce_sum(-1 * y * tf.log(x + 1e-6), -1))
            s_embed_CE = soft_CE(tf.nn.softmax(s_embed_pred), tf.nn.softmax(teacher_logits))
            s_CE = soft_CE(tf.nn.softmax(logits), tf.nn.softmax(teacher_logits))

        # hard label loss
        hard_CE = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=train_y))
        train_loss = euclidean_loss + s_embed_CE + s_CE * .5 + hard_CE

        train_output = tf.argmax(tf.nn.softmax(logits, -1), -1, output_type=tf.int32)
        train_accu = tf.where(tf.equal(train_output, train_y), tf.ones_like(train_output), tf.zeros_like(train_output))
        train_accu = tf.reduce_sum(train_accu) / args.batch_size * 100

        valid_output = tf.argmax(tf.nn.softmax(v_logits, -1), -1, output_type=tf.int32)
        valid_accu = tf.where(tf.equal(valid_output, valid_y), tf.ones_like(valid_output), tf.zeros_like(valid_output))
        valid_accu = tf.reduce_sum(valid_accu) / args.batch_size * 100

    with tf.variable_scope('Summary'):
        tf.summary.histogram('logit_raw', logits)
        tf.summary.histogram('logit_softmax', train_output)

        tf.summary.scalar('s_embed_CE', s_embed_CE)
        tf.summary.scalar('s_CE', s_CE)
        tf.summary.scalar('euclidean_loss', euclidean_loss)
        tf.summary.scalar('hard_CE', hard_CE)

        tf.summary.scalar('train_loss', train_loss)
        tf.summary.scalar('train_accu', train_accu)
        tf.summary.scalar('valid_accu', valid_accu)

    train_params = list(filter(lambda x: 'Adam' not in x.op.name and 'SqueezeNeXt' in x.op.name,
                               tf.contrib.slim.get_variables_to_restore(exclude=['InceptionResnetV1'])))
    teacher_params = list(filter(lambda x: 'Adam' not in x.op.name and 'Inception' in x.op.name,
                                 tf.contrib.slim.get_variables_to_restore(exclude=['SqueezeNeXt'])))
    inference_param = train_params[:-4]

    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(LEARNING_RATE, global_step, 10000, 0.9, staircase=True)
    if args.optim_type == 'adam':
        optim = tf.train.AdamOptimizer(learning_rate)
    elif args.optim_type == 'adagrad':
        optim = tf.train.AdagradOptimizer(learning_rate)
    else:
        optim = tf.train.GradientDescentOptimizer(learning_rate)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optim.minimize(train_loss, global_step=global_step, var_list=train_params)

    saver = tf.train.Saver(var_list=train_params)

    if LOG_ALL_TRAIN_PARAMS:
        for i in train_params:
            tf.summary.histogram(i.op.name, i)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    if args.load:
        saver.restore(sess, args.weight_path + MODEL_NAME)

    teacher_saver = tf.train.Saver(var_list=teacher_params)
    teacher_saver.restore(sess, args.t_weight_path + args.t_model_name)

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
            saver.save(sess, args.weight_path + MODEL_NAME, step)
            print('[Weights saved] weights saved at {}'.format(args.weight_path + MODEL_NAME))

        step += 1

    coord.request_stop()
    coord.join(threads)


if __name__ == '__main__':
    main(get_args())

