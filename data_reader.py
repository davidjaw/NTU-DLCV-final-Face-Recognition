import tensorflow as tf
import numpy as np


class DataReader(object):
    def __init__(
            self,
            data_path,
            random_seed=9527
    ):
        self.data_path = data_path
        self.seed = random_seed
        self.train_data_raw, self.valid_data_raw = [None, None]
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

        valid_img_path, valid_label = [[], []]
        for instance in valid_data:
            img_fn, celeb_id = instance
            if celeb_id not in dict_id_to_class:
                raise KeyError('Celeb id not found: {}'.format(celeb_id))
            valid_img_path.append('{}val/{}'.format(self.data_path, img_fn))
            valid_label.append(dict_id_to_class[celeb_id])

        return train_img_path, train_label, valid_img_path, valid_label, dict_id_to_class, dict_class_to_id

    def get_instance(self, batch_size, mode):
        img_path = self.train_img_path if mode == 'train' else self.valid_img_path
        label = self.train_label if mode == 'train' else self.valid_label

        tf_img_path, tf_label = [tf.convert_to_tensor(x) for x in [img_path, label]]
        tf_img_path, tf_label = tf.train.slice_input_producer([tf_img_path, tf_label], seed=self.seed)

        tf_img = tf.image.decode_jpeg(tf.read_file(tf_img_path), channels=3)
        tf_img = tf.cast(tf_img, tf.float32)
        # inception pre-processing
        tf_img = (tf_img - .5) * 2
        tf_img.set_shape([218, 178, 3])

        tf_imgs, tf_labels = tf.train.batch([tf_img, tf_label], batch_size)

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

