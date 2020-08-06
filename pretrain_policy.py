# https://datamarket.com/data/set/22ox/monthly-milk-production-pounds-per-cow-jan-62-dec-75#!ds=22ox&display=line
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import numpy as np
from tf2rl.policies.wave_ft_actor import WaveFTActor
from tf2rl.misc.prepare_output_dir import prepare_output_dir

import sys
import signal
def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)

# Create model
action_dim = 25
state_shape = (6 + 6 + action_dim + (12*6),)
max_action = 1.
actor_units = [256,256]
actor = WaveFTActor(state_shape, action_dim, max_action, squash=True, units=actor_units)

# Load data
filename = '/root/dev/pretrain/processed_randerror_p14_4.npy'
data = np.load(filename, allow_pickle=True)
print(data.shape)

x = np.array(data[:,0].tolist()).astype(float)
y = np.array(data[:,1].tolist()).astype(float)
train_dataset = tf.data.Dataset.from_tensor_slices((x, y))
BATCH_SIZE = 1024
SHUFFLE_BUFFER_SIZE = 1024

train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
print(x.shape, y.shape)

lr=0.002
opt = tf.keras.optimizers.Adam(learning_rate=lr)
actor.compile(opt, loss='mean_squared_error')

loss_fn = tf.keras.losses.MeanSquaredError()

@tf.function
def training_step(x):
    with tf.GradientTape() as tape:
        sample_actions, logp, _ = actor(x[0], test=True)  # Compute input reconstruction.
        # Compute loss.
        # print(x[1].shape, sample_actions.shape)
        loss = loss_fn(x[1], sample_actions)
        # loss += sum(actor.losses)  # Add KLD term.
    # Update the weights of the VAE.
    grads = tape.gradient(loss, actor.trainable_weights)
    opt.apply_gradients(zip(grads, actor.trainable_weights))
    return loss

print('Train...')
losses = []  # Keep track of the losses over time.
for epoch in range(5):
    # Iterate over the batches of a dataset.
    for step, x in enumerate(train_dataset):
        loss = training_step(x)
        # Logging.
        losses.append(float(loss))
        if step % 100 == 0:
            print("Epoch", epoch, "Step:", step, "Loss:", sum(losses) / len(losses))

        # Stop after 1000 steps.
        # Training the model to convergence is left
        # as an exercise to the reader.
        # if step >= 10000:
        #     break

output_dir = prepare_output_dir(
            args=None, user_specified_dir=None,
            suffix="{}_{}".format(policy.policy_name, args.dir_suffix))
checkpoint = tf.train.Checkpoint(policy=policy)
checkpoint_manager = tf.train.CheckpointManager(
        _checkpoint, directory=output_dir, max_to_keep=5)
