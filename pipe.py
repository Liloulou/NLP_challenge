import tensorflow as tf
import tensorflow.contrib as contrib
import numpy as np

text_vocab_list = np.array([
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a',
    'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p',
    'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', ' ', '!', '/', '?'
])

cim_vocab_list = np.array([
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z', ' ', '.'
])

is_upper_vocab_list = ['0', '1']

COL_NAMES = ['raw_text', 'ICD10', 'is_upper']
COL_TYPES = [''] * 3


def _parse_line(line):
    """
    takes in a line of a csv file and returns its data as a feature dictionary
    :param line: the csv file's loaded line
    :return: the associated feature dictionary
    """

    fields = tf.decode_csv(line, record_defaults=COL_TYPES)
    features = dict(zip(COL_NAMES, fields))

    return features


def _split(features):
    """

    :param features:
    :return:
    """

    for key in features.keys():
        features[key] = tf.sparse.reshape(
            tf.string_split([features[key]], delimiter=''),
            shape=[-1]
        )

    return features


def _pre_process(line):
    """
    Overheads all csv processing functions.
    :param line: a raw csv line
    :return:
    """
    features = _parse_line(line)
    features = _split(features)
    labels = {'labels': features.pop('ICD10')}

    return features, labels


def csv_input_fn(dataset_name, batch_size, num_epochs):
    """
    A predefined input function to feed an Estimator csv based cepidc files
    :param dataset_name: the file's ending type (either 'train, 'valid' or 'test')
    :param batch_size: the size of batches to feed the computational graph
    :param num_epochs: the number of time the entire dataset should be exposed to a gradient descent iteration
    :return: a BatchedDataset as a tuple of a feature dictionary and the labels
    """

    dataset = tf.data.TextLineDataset('data/NLP_full_challenge_' + dataset_name + '.csv').skip(1)
    dataset = dataset.map(_pre_process)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    # TODO put shuffle back
    # dataset = dataset.shuffle(buffer_size=10000).batch(batch_size).repeat(num_epochs)

    return dataset.prefetch(buffer_size=batch_size)


def make_columns():
    """
    Builds the feature_columns required by the estimator to link the Dataset and the model_fn
    :return:
    """

    columns_dict = {}

    with tf.name_scope('text_column'):
        columns_dict['raw_text'] = tf.feature_column.indicator_column(
            contrib.feature_column.sequence_categorical_column_with_vocabulary_list(
                'raw_text',
                text_vocab_list,
                default_value=-1
            )
        )

    with tf.name_scope('is_upper_column'):
        columns_dict['is_upper'] = tf.feature_column.indicator_column(
            contrib.feature_column.sequence_categorical_column_with_vocabulary_list(
                'is_upper',
                is_upper_vocab_list,
                default_value=0
            )
        )

    with tf.name_scope('labels_one_hot_encoding'):
        columns_dict['labels'] = tf.feature_column.indicator_column(
            contrib.feature_column.sequence_categorical_column_with_vocabulary_list(
                'labels',
                cim_vocab_list,
                default_value=37
            )
        )

    return columns_dict


def make_input_layers(dataset, feature_columns, batch_size):
    features = dataset[0]
    labels = dataset[1]

    with tf.name_scope('raw_text_input_layer'):
        text, text_len = contrib.feature_column.sequence_input_layer(
            features=features,
            feature_columns=feature_columns['raw_text']
        )

        is_upper = contrib.feature_column.sequence_input_layer(
            features=features,
            feature_columns=feature_columns['is_upper']
        )[0]

        is_upper = tf.expand_dims(
            tf.cast(
                tf.argmax(is_upper, axis=-1, output_type=tf.int32), dtype=tf.float32),
            axis=-1
        )

        features = tf.reshape(
            tf.concat([text, is_upper], axis=-1),
            shape=[batch_size, -1, len(text_vocab_list) + 1]
        )
        features = features[:, :tf.reduce_max(text_len)]

    with tf.name_scope('labels_input_layer'):
        labels, labels_len = contrib.feature_column.sequence_input_layer(
            features=labels,
            feature_columns=feature_columns['labels']
        )

        labels += (1 - tf.reduce_sum(
            labels,
            axis=-1,
            keepdims=True
        )) * tf.one_hot(len(cim_vocab_list) - 1, depth=len(cim_vocab_list))

        labels = tf.reshape(labels, shape=[batch_size, tf.reduce_max(labels_len), len(cim_vocab_list)])
        labels = labels[:, :tf.reduce_max(labels_len)]
        labels_len = tf.cast(labels_len, tf.int32)

    return {'text': features, 'sequence_length': text_len}, labels, labels_len


data = csv_input_fn('train', 10, 1).make_one_shot_iterator().get_next()
features, labels, labels_len = make_input_layers(data, make_columns(), batch_size=10)
init_tables = tf.tables_initializer()
init_var = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_tables)
    sess.run(init_var)
    a = sess.run(labels)
"""

data = csv_input_fn('train', 10, 1).make_one_shot_iterator().get_next()
features, labels, labels_len = make_input_layers(data, make_columns(), batch_size=10)
init_tables = tf.tables_initializer()
init_var = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_tables)
    sess.run(init_var)
    a = sess.run(labels)
    
data = csv_input_fn('train', hparams['batch_size'], 1).make_one_shot_iterator().get_next()

dat = tf.Session().run(data)
features, labels, labels_len = make_input_layers(data, make_columns(), batch_size=hparams['batch_size'])

encoding, _ = model_utils.tcn_encoder(
    features['text'],
    features['sequence_length'],
    params=hparams['inference']['encoder'],
    mode=tf.estimator.ModeKeys.TRAIN,
    name='coucou'
)

_, state = model_utils.tcn_encoder(
    features['text'],
    features['sequence_length'],
    params=hparams['inference']['encoder'],
    mode=tf.estimator.ModeKeys.TRAIN,
    name='coucou_2'
)

encoder_state = model_utils.from_tcn_encoder_to_rnn_decoder_states(
    state,
    params=hparams['inference'],
    mode=tf.estimator.ModeKeys.TRAIN
)

logits = model_utils.attention_beam_search_decoder(
            encoder_out=encoding,
            encoder_state=encoder_state,
            sequence_length=features['sequence_length'],
            labels=labels,
            labels_length=labels_len,
            params=hparams['inference']['decoder'],
            mode=tf.estimator.ModeKeys.EVAL
        )

init_tables = tf.tables_initializer()
init_var = tf.global_variables_initializer()
stop = False
i = 0
with tf.Session() as sess:
    sess.run(init_tables)
    sess.run(init_var)
    while not stop:
        i += 1
        print(i)
        a = sess.run(logits)

"""


