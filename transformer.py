import tensorflow as tf


def make_positional_encoding(tensor):
    basis = tf.ones_like(tensor[0])

    pos = tf.cumsum(basis, axis=0) - 1
    i = tf.cumsum(basis, axis=1) - 1

    pos_encoding = pos / tf.pow(10000., 4 * (i // 2) / (i[0, -1] + 1))

    even = tf.sin(pos_encoding[:, ::2])
    odd = tf.cos(pos_encoding[:, 1::2])

    encoding = tf.reshape(
        tf.stack([even, odd], axis=-1),
        shape=[pos[-1, -1] + 1, -1]
    )

    return encoding


class PositionalEmbedding(tf.layers.Dense):

    def __init__(
            self,
            units,
            kernel_initializer=None,
            drop_out=None,
            name=None,
    ):

        self.drop_out = drop_out
        super(PositionalEmbedding, self).__init__(
            units=units,
            kernel_initializer=kernel_initializer,
            use_bias=False,
            name=name,
        )

    def call(self, inputs):

        outputs = super(PositionalEmbedding, self).call(inputs)
        outputs += make_positional_encoding(outputs)
        if self.drop_out is not None:
            outputs = self.drop_out(outputs)

        return outputs


class MultiHeadAttention(tf.keras.layers.Layer):

    def __init__(
            self,
            dim_key,
            dim_value,
            dim_model,
            num_heads=1,
            masked=False,
            kernel_initializer=None,
            trainable=True,
            name=None,
    ):

        self.dim_key = dim_key
        self.dim_value = dim_value
        self.num_heads = num_heads
        self.masked = masked

        super(MultiHeadAttention, self).__init__(trainable=trainable, name=name, dtype=tf.float32)

        self.values_projector = [
            tf.layers.Dense(
                units=dim_value,
                kernel_initializer=kernel_initializer,
                name=self.name + '_values_projector_' + str(i),
            ) for i in range(num_heads)
        ]

        self.keys_projector = [
            tf.layers.Dense(
                units=dim_key,
                kernel_initializer=kernel_initializer,
                name=self.name + '_keys_projector_' + str(i),
            ) for i in range(num_heads)
        ]

        self.queries_projector = [
            tf.layers.Dense(
                units=dim_key,
                kernel_initializer=kernel_initializer,
                name=self.name + '_queries_projector_' + str(i),
            ) for i in range(num_heads)
        ]

        self.linear_out = tf.layers.Dense(
            units=dim_model,
            kernel_initializer=kernel_initializer,
            name=self.name + '_linear_out',
        )

    def set_masked(self, masked_value):
        self.masked = masked_value

    def build(self, input_shape):

        values_shape, queries_shape, keys_shape = input_shape

        for proj in self.values_projector:
            proj.build(values_shape)

        for proj in self.queries_projector:
            proj.build(queries_shape)

        for proj in self.keys_projector:
            proj.build(keys_shape)

        self.linear_out.build((queries_shape[0], queries_shape[1], self.dim_value * self.num_heads))

        self.built = True

    def _scaled_dot_prod(self, values, queries, keys):

        dot_prod = tf.matmul(
            queries,
            keys,
            transpose_b=True
        )
        dot_prod /= (self.dim_key ** 0.5)

        if self.masked:

            mask = tf.ones_like(dot_prod)
            mask = tf.linalg.LinearOperatorLowerTriangular(mask).to_dense()
            minus_inf = tf.ones_like(dot_prod) * -1e19
            dot_prod = tf.where(tf.equal(0., mask), minus_inf, dot_prod)

        scaled_dot_product = tf.matmul(
            tf.nn.softmax(dot_prod),
            values
        )

        return scaled_dot_product

    def call(self, inputs, **kwargs):

        values, queries, keys = inputs

        scaled_dot_prod_out = []
        for i in range(self.num_heads):
            scaled_dot_prod_out.append(self._scaled_dot_prod(
                values=self.values_projector[i].call(values),
                queries=self.queries_projector[i].call(queries),
                keys=self.keys_projector[i].call(keys),
            ))

        out = tf.concat(scaled_dot_prod_out, axis=-1)

        return self.linear_out.call(out)


class DecoderBlock(tf.keras.layers.Layer):

    def __init__(
            self,
            params,
            kernel_initializer=None,
            drop_out=None,
            trainable=True,
            name=None,
    ):

        # TODO tweak kernel initializer to match paper
        self.self_attention_block = MultiHeadAttention(
            params['dim_key'],
            params['dim_value'],
            params['dim_model'],
            num_heads=params['num_heads'],
            masked=True,
            kernel_initializer=kernel_initializer,
            name=name + '_masked_attention',
        )

        self.encoder_decoder_attention = MultiHeadAttention(
            params['dim_key'],
            params['dim_value'],
            params['dim_model'],
            num_heads=params['num_heads'],
            masked=False,
            kernel_initializer=kernel_initializer,
            name=name + '_encoder_decoder_attention',
        )

        self.feed_forward_internal = tf.layers.Dense(
            units=params['units'],
            activation=tf.nn.leaky_relu,
            kernel_initializer=kernel_initializer,
            name=name + '_internal_position_wide_feed_forward',
        )

        self.feed_forward_out = tf.layers.Dense(
            units=params['dim_model'],
            kernel_initializer=kernel_initializer,
            name=name + 'out_position_wide_feed_forward',
        )

        self._drop_out = drop_out
        self._units = params['units']
        super(DecoderBlock, self).__init__(trainable=trainable, name=name, dtype=tf.float32)

    def build(self, input_shape):
        decoder_in_shape, encoder_out_shape = input_shape

        self.self_attention_block.build(
            (
                decoder_in_shape,
                decoder_in_shape,
                decoder_in_shape
            )
        )

        self.encoder_decoder_attention.build(
            (
                encoder_out_shape,
                decoder_in_shape,
                encoder_out_shape
            )
        )

        self.feed_forward_internal.build(decoder_in_shape)
        self.feed_forward_out.build(
            (
                decoder_in_shape[0],
                decoder_in_shape[1],
                self._units
            )
        )
        self.built = True

    def call(self, inputs, **kwargs):
        decoder_in, encoder_out = inputs

        with tf.variable_scope(self.name + '_self_attention') as scope:

            self_out = self.self_attention_block.call(
                (
                    decoder_in,
                    decoder_in,
                    decoder_in
                )
            )

            if callable(self._drop_out):
                self_out = self._drop_out(self_out)

            self_out += decoder_in
            self_out = tf.contrib.layers.layer_norm(
                self_out,
                reuse=tf.AUTO_REUSE,
                begin_norm_axis=2,
                scope=scope
            )

        with tf.variable_scope(self.name + '_encoder_decoder_attention') as scope:
            enc_dec_out = self.encoder_decoder_attention.call(
                (
                    encoder_out,
                    self_out,
                    encoder_out
                )
            )

            if callable(self._drop_out):
                enc_dec_out = self._drop_out(enc_dec_out)

            enc_dec_out += self_out
            enc_dec_out = tf.contrib.layers.layer_norm(
                enc_dec_out,
                reuse=tf.AUTO_REUSE,
                begin_norm_axis=2,
                scope=scope
            )

        with tf.variable_scope(self.name + '_feed_forward') as scope:
            out = self.feed_forward_internal.call(
                enc_dec_out
            )

            out = self.feed_forward_out.call(
                out
            )

            if callable(self._drop_out):
                out = self._drop_out(out)

            out += enc_dec_out
            out = tf.contrib.layers.layer_norm(
                out,
                reuse=tf.AUTO_REUSE,
                begin_norm_axis=2,
                scope=scope
            )

        return out


class EncoderBlock(tf.keras.layers.Layer):

    def __init__(
            self,
            params,
            kernel_initializer=None,
            drop_out=None,
            trainable=True,
            name=None
    ):
        self.self_attention_block = MultiHeadAttention(
            params['dim_key'],
            params['dim_value'],
            params['dim_model'],
            num_heads=params['num_heads'],
            masked=False,
            kernel_initializer=kernel_initializer,
            name=name + '_masked_attention',
        )

        self.feed_forward_internal = tf.layers.Dense(
            units=params['units'],
            activation=tf.nn.leaky_relu,
            kernel_initializer=kernel_initializer,
            name=name + '_internal_position_wide_feed_forward',
        )

        self.feed_forward_out = tf.layers.Dense(
            units=params['dim_model'],
            kernel_initializer=kernel_initializer,
            name=name + 'out_position_wide_feed_forward',
        )

        self._drop_out = drop_out
        self._units = params['units']
        super(EncoderBlock, self).__init__(trainable=trainable, name=name, dtype=tf.float32)

    def build(self, input_shape):

        self.self_attention_block.build(
            (
                input_shape,
                input_shape,
                input_shape
            )
        )

        self.feed_forward_internal.build(input_shape)
        self.feed_forward_out.build(
            self.feed_forward_internal.compute_output_shape(input_shape)
        )

        self.built = True

    def call(self, inputs, **kwargs):

        with tf.variable_scope(self.name + '_self_attention') as scope:

            self_out = self.self_attention_block.call(
                (
                    inputs,
                    inputs,
                    inputs
                )
            )

            if callable(self._drop_out):
                self_out = self._drop_out(self_out)

            self_out += inputs
            self_out = tf.contrib.layers.layer_norm(
                self_out,
                reuse=tf.AUTO_REUSE,
                begin_norm_axis=2,
                scope=scope
            )

        with tf.variable_scope(self.name + '_feed_forward'):

            out = self.feed_forward_internal.call(
                self_out
            )

            out = self.feed_forward_out.call(
                out
            )

            if callable(self._drop_out):
                out = self._drop_out(out)

            out += self_out
            out = tf.contrib.layers.layer_norm(
                out,
                reuse=tf.AUTO_REUSE,
                begin_norm_axis=2,
                scope=scope
            )

        return out


class Decoder(tf.keras.layers.Layer):

    def __init__(self,
                 params,
                 kernel_initializer,
                 drop_out=None,
                 num_layers=6,
                 trainable=True,
                 name=None,):
        """

        :param params: a python Dictionary with entries:
                        - 'dim_key'
                        - 'dim_value'
                        - 'dim_model'
                        - 'num_heads'
                        - 'units'
        :param kernel_initializer:
        :param drop_out:
        :param num_layers:
        :param trainable:
        :param name:
        """

        self.linear_embedding = PositionalEmbedding(
            units=params['dim_model'],
            kernel_initializer=kernel_initializer,
            drop_out=drop_out,
            name=name + '_positional_embedding'
        )

        self.decoder_blocks = [
            DecoderBlock(
                params=params,
                kernel_initializer=kernel_initializer,
                drop_out=drop_out,
                name=name + '_decoder_block_' + str(i),
            ) for i in range(num_layers)
        ]

        super(Decoder, self).__init__(trainable=trainable, name=name, dtype=tf.float32)

    def build(self, input_shape):

        decoder_in_shape, encoder_out_shape = input_shape

        self.linear_embedding.build(decoder_in_shape)

        actual_input_shape = (self.linear_embedding.compute_output_shape(decoder_in_shape), encoder_out_shape)

        for block in self.decoder_blocks:
            block.build(actual_input_shape)

        self.built = True

    def call(self, inputs, **kwargs):

        decoder_in, encoder_out = inputs

        out = self.linear_embedding.call(decoder_in)

        for block in self.decoder_blocks:
            out = block.call(
                (
                    (out, encoder_out)
                )
            )

        return out


class Encoder(tf.keras.layers.Layer):

    def __init__(
            self,
            params,
            kernel_initializer,
            drop_out=None,
            num_layers=6,
            trainable=True,
            name=None,):
        """

        :param params: a python Dictionary with entries:
                        - 'dim_key'
                        - 'dim_value'
                        - 'dim_model'
                        - 'num_heads'
                        - 'units'
        :param kernel_initializer:
        :param drop_out:
        :param num_layers:
        :param trainable:
        :param name:
        :return:
        """

        self.linear_embedding = PositionalEmbedding(
            units=params['dim_model'],
            kernel_initializer=kernel_initializer,
            drop_out=drop_out,
            name=name + '_positional_embedding'
        )

        self.encoder_blocks = [
            EncoderBlock(
                params=params,
                kernel_initializer=kernel_initializer,
                drop_out=drop_out,
                name=name + '_encoder_block_' + str(i),
            ) for i in range(num_layers)
        ]

        super(Encoder, self).__init__(trainable=trainable, name=name, dtype=tf.float32)

    def build(self, input_shape):

        self.linear_embedding.build(input_shape)

        actual_input_shape = self.linear_embedding.compute_output_shape(input_shape)

        for block in self.encoder_blocks:
            block.build(actual_input_shape)

        self.built = True

    def call(self, inputs, **kwargs):

        out = self.linear_embedding.call(inputs)

        for block in self.encoder_blocks:
            out = block.call(
                out
            )

        return out


"""
test = tf.cast(tf.random.uniform([100, 250], maxval=36), tf.int32)
test = tf.one_hot(test, depth=36, dtype=tf.float32)

decoder = TransformerDecoder(
    params=hparams,
    kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
    num_layers=3,
    name='decoder_biatch'
)

outputs = decoder.apply((test, test))

init_var = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_var)
    bla = sess.run(outputs)

hparams = {
    'dim_key': 5,
    'dim_value': 6,
    'dim_model': 7,
    'num_heads': 8,
    'units': 10
}

my_decoder = TransformerDecoder(
    hparams,
    tf.contrib.layers.xavier_initializer(uniform=False),
    num_layers=3,
    name='decoder'
)

test_in_1 = tf.ones([50, 22, 3])
test_in_2 = tf.ones([50, 22, 3])

test_1 = my_decoder.call((test_in_1, test_in_1))
test_2 = my_decoder.call((test_in_2, test_in_2))

with tf.Session() as sess:

    x, y = sess.run([test_1, test_2])
)"""