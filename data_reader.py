import tensorflow as tf
import numpy as np
from scipy import misc
import os


def random_rotate_image(image):
    angle = np.random.uniform(low=-15.0, high=15.0)
    return misc.imrotate(image, angle, 'bicubic')


class TestDataReader(object):
    def __init__(self, data_path):
        self.data_path = data_path
        self.file_paths = sorted(os.listdir(self.data_path))

    def get_instance(self, batch_size):
        tf_file_path = tf.convert_to_tensor([self.data_path + x for x in self.file_paths], tf.string)
        tf_path = tf.train.slice_input_producer([tf_file_path], shuffle=False)[0]
        tf_img = tf.image.decode_jpeg(tf.read_file(tf_path), channels=3)

        h = int(218 / 5)
        w = int(178 / 5)
        tf_img.set_shape([218, 178, 3])
        tf_img = tf_img[h*2:h*4, w:w*4, :]

        tf_img = tf.cast(tf_img, tf.float32) / 255.
        tf_imgs = tf.train.batch([tf_img], batch_size)
        tf_imgs = (tf_imgs - .5) * 2
        tf_imgs = tf.image.resize_bilinear(tf_imgs, [218, 178])
        return tf_imgs, len(self.file_paths)


class DataReader(object):
    def __init__(
            self,
            data_path,
            random_seed=9527
    ):
        self.data_path = data_path
        self.seed = random_seed
        self.train_data_raw, self.valid_data_raw = [None, None]
        self.dict_instance_id = {'train': {}, 'valid': {}}
        self.train_img_path, self.train_label, self.valid_img_path, self.valid_label, self.dict_id, self.dict_class = \
            self._prepare_data()

    def _prepare_data(self):
        def read_ground_truth(file_path):
            # read ground file and split into list
            with open(file_path, 'r') as f:
                content = [x.split(' ') for x in f.read().split('\n')]
                while content[-1] == ['']:
                    content.pop()
            return content

        train_gt_file = self.data_path + 'train_id.txt'
        valid_gt_file = self.data_path + 'val_id.txt'

        train_data = read_ground_truth(train_gt_file)
        valid_data = read_ground_truth(valid_gt_file)
        self.train_data_raw = train_data
        self.valid_data_raw = valid_data

        # embed id into class
        dict_id_to_class, dict_class_to_id = [{}, {}]
        train_img_path, train_label = [[], []]
        embed_cnt = 0
        for instance in train_data:
            img_fn, celeb_id = instance
            if celeb_id not in dict_id_to_class:
                dict_id_to_class[celeb_id] = embed_cnt
                dict_class_to_id[embed_cnt] = celeb_id
                embed_cnt += 1
            train_img_path.append('{}train/{}'.format(self.data_path, img_fn))
            train_label.append(dict_id_to_class[celeb_id])

            # collect to dict
            if celeb_id not in self.dict_instance_id['train']:
                self.dict_instance_id['train'][celeb_id] = ['{}train/{}'.format(self.data_path, img_fn)]
            else:
                self.dict_instance_id['train'][celeb_id].append('{}train/{}'.format(self.data_path, img_fn))

        valid_img_path, valid_label = [[], []]
        for instance in valid_data:
            img_fn, celeb_id = instance
            if celeb_id not in dict_id_to_class:
                raise KeyError('Celeb id not found: {}'.format(celeb_id))
            valid_img_path.append('{}val/{}'.format(self.data_path, img_fn))
            valid_label.append(dict_id_to_class[celeb_id])

            # collect to dict
            if celeb_id not in self.dict_instance_id['valid']:
                self.dict_instance_id['valid'][celeb_id] = ['{}val/{}'.format(self.data_path, img_fn)]
            else:
                self.dict_instance_id['valid'][celeb_id].append('{}val/{}'.format(self.data_path, img_fn))

        return train_img_path, train_label, valid_img_path, valid_label, dict_id_to_class, dict_class_to_id

    def get_instance(self, batch_size, mode):
        img_path = self.train_img_path if mode == 'train' else self.valid_img_path
        label = self.train_label if mode == 'train' else self.valid_label

        tf_img_path, tf_label = [tf.convert_to_tensor(x) for x in [img_path, label]]
        tf_img_path, tf_label = tf.train.slice_input_producer([tf_img_path, tf_label], seed=self.seed)

        tf_img = tf.image.decode_jpeg(tf.read_file(tf_img_path), channels=3)
        h = int(218 / 5)
        w = int(178 / 5)
        tf_img.set_shape([218, 178, 3])
        tf_img = tf_img[h*2:h*4, w:w*4, :]
        tf_img_shape = tf_img.get_shape().as_list()

        if mode == 'train':
            # data augmentation
            tf_img = tf.image.random_flip_left_right(tf_img)
            tf_img = tf.py_func(random_rotate_image, [tf_img], tf.uint8)
            tf_img.set_shape(tf_img_shape)

        tf_img = tf.cast(tf_img, tf.float32) / 255.

        tf_imgs, tf_labels = tf.train.batch([tf_img, tf_label], batch_size)

        if mode == 'train':
            # data augmentation
            # random scale
            tf_imgs_scaled = tf.image.resize_bilinear(tf_imgs, [int(x * 1.2) for x in tf_img_shape[:-1]])
            tf_imgs_scaled = tf.image.resize_image_with_crop_or_pad(tf_imgs_scaled, tf_img_shape[0], tf_img_shape[1])
            random_scale = tf.random_uniform([100], 0., 1.)
            tf_imgs = tf.where(tf.greater(random_scale, .7), tf_imgs, tf_imgs_scaled)

            # gray-scale augmentation
            tf_imgs_gray = tf.reduce_mean(tf_imgs, -1, keepdims=True)
            tf_imgs_gray = tf.concat([tf_imgs_gray, tf_imgs_gray, tf_imgs_gray], -1)

            # build-in color-based augmentations
            tf_imgs = tf.image.random_brightness(tf_imgs, max_delta=32. / 255.)
            tf_imgs = tf.image.random_saturation(tf_imgs, lower=0.75, upper=1.25)
            tf_imgs = tf.image.random_hue(tf_imgs, max_delta=.05)
            tf_imgs = tf.clip_by_value(tf_imgs, 0., 1.)

            random_gray = tf.random_uniform([100], 1., 0.)
            tf_imgs = tf.where(tf.greater(random_gray, .8), tf_imgs_gray, tf_imgs)

            tf.summary.image('in_image', tf_imgs, max_outputs=5)

        # inception pre-processing
        tf_imgs = (tf_imgs - .5) * 2
        tf_imgs = tf.image.resize_bilinear(tf_imgs, [218, 178])

        return tf_imgs, tf_labels

    def debug_cv(self):
        import cv2
        for i in range(len(self.dict_class.keys())):
            class_id = self.dict_class[i]
            train_imgs = list(filter(lambda x: x[1] == class_id, self.train_data_raw))
            valid_imgs = list(filter(lambda x: x[1] == class_id, self.valid_data_raw))
            train_imgs = [cv2.imread(self.data_path + 'train/' + x[0]) for x in train_imgs]
            valid_imgs = [cv2.imread(self.data_path + 'val/' + x[0]) for x in valid_imgs]

            h, w = train_imgs[0].shape[:-1]
            h = int(h / 5)
            w = int(w / 5)

            # train_imgs = np.concatenate(train_imgs, 1)
            # valid_imgs = np.concatenate(valid_imgs, 1)
            train_imgs = np.concatenate([x[h*2:h*4, w:w*4] for x in train_imgs], 1)
            valid_imgs = np.concatenate([x[h*2:h*4, w:w*4] for x in valid_imgs], 1)

            cv2.imshow('train_img', train_imgs)
            cv2.imshow('valid_img', valid_imgs)
            cv2.waitKey()


# data_reader = DataReader('./dlcv_final_2_dataset/')
# data_reader.debug_cv()

