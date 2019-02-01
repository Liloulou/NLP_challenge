import tensorflow as tf
import pickle
import pipe
import transformer_model as model
import os

model_dir = 'model_directory/'

hparams = {
        'batch_size': 10,
        'num_epochs': 500,
        'train': {
            'learning_rate': 0.001,
            'decay_rate': 0.98,
            'decay_steps': 2000
        },
        'inference': {
            'encoder': {
                'dim_key': 64,
                'dim_value': 64,
                'dim_model': 512,
                'num_heads': 8,
                'units': 2048

            },
            'decoder': {
                'dim_key': 64,
                'dim_value': 64,
                'dim_model': 512,
                'num_heads': 8,
                'units': 2048
            },
            'drop_out': 0.1
        }
    }

hparams = {
    'batch_size': 10,
    'num_epochs': 500,
    'train': {
        'learning_rate': 0.001,
        'decay_rate': 0.98,
        'decay_steps': 2000
    },
    'inference': {
        'encoder': {
            'dim_key': 12,
            'dim_value': 12,
            'dim_model': 12,
            'num_heads': 4,
            'units': 15

        },
        'decoder': {
            'dim_key': 12,
            'dim_value': 12,
            'dim_model': 12,
            'num_heads': 4,
            'units': 15
        },
        'drop_out': 0.1
    }
}


def get_next_model_dir():
    list_name = [int(name[name.find('_') + 1:]) for name in os.listdir('model_directory')]

    if len(list_name) is 0:
        last_model = 0
    else:
        last_model = max(list_name)

    return 'model_directory/model_' + str(last_model + 1)


tf.logging.set_verbosity(tf.logging.INFO)

while True:

    next_model_dir = get_next_model_dir()
    hparams['feature_columns'] = pipe.make_columns()

    run_config = tf.estimator.RunConfig(
        model_dir=get_next_model_dir(),
        save_checkpoints_steps=200,
        keep_checkpoint_max=1,
    )

    estimator = tf.estimator.Estimator(
        model_fn=model.greedy_model_fn,
        config=run_config,
        params=hparams
    )

    train_spec = tf.estimator.TrainSpec(
        input_fn=lambda: pipe.csv_input_fn('train', hparams['batch_size'], hparams['num_epochs']),
        max_steps=100000,
        hooks=[tf.contrib.estimator.stop_if_no_decrease_hook(
            estimator,
            metric_name='cim_10_accuracy',
            max_steps_without_decrease=10000,
            run_every_secs=None,
            run_every_steps=500
        )]
    )

    eval_spec = tf.estimator.EvalSpec(
        input_fn=lambda: pipe.csv_input_fn('valid', hparams['batch_size'], 20),
        steps=100,
        start_delay_secs=10,
        throttle_secs=10
    )

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
