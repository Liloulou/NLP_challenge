import transformer
import pipe
import custom_metrics
import tensorflow as tf
import decoder_utils as utils


def greedy_transformer_decoder(features, labels, labels_length, params, mode):
    kernel_initializer = tf.contrib.layers.xavier_initializer(uniform=False)

    with tf.name_scope('encoder'):

        encoder = transformer.Encoder(
            params['encoder'],
            kernel_initializer=kernel_initializer,
            drop_out=lambda x: tf.layers.dropout(
                x,
                rate=params['drop_out'],
                training=mode == tf.estimator.ModeKeys.TRAIN
            ),
            num_layers=6,
            name='transformer_encoder'
        )

        encoder_outputs = encoder.apply(features)

    with tf.name_scope('decoder'):

        decoder = transformer.Decoder(
            params['decoder'],
            kernel_initializer=kernel_initializer,
            drop_out=lambda x: tf.layers.dropout(
                x,
                rate=params['drop_out'],
                training=mode == tf.estimator.ModeKeys.TRAIN
            ),
            num_layers=6,
            name='transformer_decoder'
        )

        soft_layer = tf.layers.Dense(
            units=labels.get_shape()[-1] + 1,
            kernel_initializer=kernel_initializer,
            name='softmax_output_layer'
        )

        decoded_tuple = utils.transformer_decoding(
            decoder=decoder,
            encoder_outputs=encoder_outputs,
            labels=labels,
            labels_length=labels_length,
            soft_layer=soft_layer,
            mode=mode
        )

    return decoded_tuple


def make_weights(labels, labels_length):

    weights = tf.ones_like(labels, dtype=tf.float32)

    weights -= tf.one_hot(
        tf.minimum(labels_length + 1, tf.reduce_max(labels_length)),
        axis=-1,
        depth=tf.shape(labels)[1],
        dtype=tf.float32)

    weights = tf.cumprod(weights, axis=-1)

    return weights


def get_loss(logits, labels, labels_length):

    weights = make_weights(labels, labels_length)

    loss = tf.contrib.seq2seq.sequence_loss(
        logits,
        labels,
        weights
    )

    return loss


def get_train_op(loss, params):
    global_step = tf.train.get_or_create_global_step()

    learning_rate = tf.train.exponential_decay(
        params['learning_rate'],
        global_step,
        params['decay_steps'],
        params['decay_rate'],
        staircase=False
    )
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    return optimizer.minimize(loss, global_step=global_step)


def greedy_model_fn(features, labels, params, mode):
    with tf.variable_scope('my_model', reuse=tf.AUTO_REUSE):

        features, labels, labels_length = pipe.make_input_layers(
            [features, labels],
            params['feature_columns'],
            params['batch_size']
        )
        features = features['text']

        decoded_tuple = greedy_transformer_decoder(
            features=features,
            labels=labels,
            labels_length=labels_length,
            params=params['inference'],
            mode=mode
        )

        if mode == tf.estimator.ModeKeys.TRAIN:
            logits, training_labels = decoded_tuple

            labels_ind = tf.argmax(training_labels, axis=-1)

            loss = get_loss(
                logits=logits,
                labels=labels_ind,
                labels_length=labels_length
            )

            train_op = get_train_op(loss, params['train'])

            return tf.estimator.EstimatorSpec(
                mode,
                loss=loss,
                train_op=train_op
            )

        if mode == tf.estimator.ModeKeys.EVAL:

            logits, predictions, training_labels = decoded_tuple

            labels_ind = tf.argmax(training_labels, axis=-1)

            loss = get_loss(
                logits=logits,
                labels=labels_ind,
                labels_length=labels_length
            )

            eval_metrics = {
                'char_accuracy': tf.metrics.accuracy(
                    labels_ind,
                    predictions,
                    name='char_accuracy'
                ),

                'cim_10_accuracy': custom_metrics.word_level_accuracy(
                    labels_ind,
                    predictions,
                    name='ICD_10_accuracy'
                )
            }

            return tf.estimator.EstimatorSpec(
                mode,
                predictions=predictions,
                loss=loss,
                eval_metric_ops=eval_metrics
            )

        if mode == tf.estimator.ModeKeys.PREDICT:
            logits, predictions = decoded_tuple

            predictions = {
                'logits': logits,
                'predictions': predictions
            }

            return tf.estimator.EstimatorSpec(
                mode,
                predictions=predictions
            )
