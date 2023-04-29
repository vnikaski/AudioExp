from generators.TTSGenre import TTSGenre
from models.models import Kell2018
from utils.losses import Unconcerned_CCE
from keras.metrics import TopKCategoricalAccuracy, CategoricalAccuracy
import tensorflow as tf
import warnings, sys, argparse
from keras.optimizers import Adam
from keras.callbacks import LambdaCallback
from utils.metrics import UnconcernedAccuracy
import datetime

parser = argparse.ArgumentParser()

parser.add_argument('-l', '--libripath', help='Path to folder with all LibriTTS subsets')
parser.add_argument('-g', '--gtzanpath', help='Path to folder with GTZAN dataset')
parser.add_argument('-b', '--batchsize', default=64)
parser.add_argument('-e', '--epochs')
parser.add_argument('--lr', default=1e-3)
parser.add_argument('--cppath', default='./cp.ckpt')
parser.add_argument('--nmels', default=512)
parser.add_argument('--nfft', default=2048)
parser.add_argument('--hop', default=44)
parser.add_argument('--window', default=1)
parser.add_argument('--words', default=200)
parser.add_argument('--whichword', default=2)
parser.add_argument('--quiet', action='store_true', help='Silences the mid-training messages')
parser.add_argument('--logdir', default=None)


args = parser.parse_args()

train_gen = TTSGenre(
    libri_path=args.libripath,
    gtzan_path=args.gtzanpath,
    batch_size=int(args.batchsize),
    n_mels=args.nmels,
    n_fft=args.nfft,
    hop=args.hop,
    window_s=args.window,
    words=args.words,
    which_word=args.whichword,
    quiet=args.quiet,
    mode='train'
)

val_gen = TTSGenre(
    libri_path=args.libripath,
    gtzan_path=args.gtzanpath,
    batch_size=int(args.batchsize),
    n_mels=args.nmels,
    n_fft=args.nfft,
    hop=args.hop,
    window_s=args.window,
    words=train_gen.get_words(),
    quiet=args.quiet,
    mode='val'
)

xs, ys = train_gen.get_sample()
print(f'input shape: {xs.shape[1:]}')
input_shape = xs.shape[1:]
wout_shape = ys['wout'].shape[1:]
gout_shape = ys['gout'].shape[1:]


output_signature = (tf.TensorSpec(shape=(None, *xs.shape[1:])),
                    {'wout': tf.TensorSpec(shape=(None, *wout_shape)),
                     'gout': tf.TensorSpec(shape=(None, *gout_shape))})

train_gen.output_signature = output_signature
train_ds = tf.data.Dataset.from_generator(
    train_gen,
    output_signature=output_signature
)

val_gen.output_signature = output_signature
val_ds = tf.data.Dataset.from_generator(
    val_gen,
    output_signature=output_signature
)

model = Kell2018(input_shape, wout_shape[0], gout_shape[0])
model.compile(loss={'wout':Unconcerned_CCE(), 'gout':Unconcerned_CCE()},
              optimizer=Adam(learning_rate=float(args.lr)),
              metrics={
                  'wout': [UnconcernedAccuracy()],
                  'gout': [UnconcernedAccuracy()]},
              run_eagerly=True
              )
print(model.summary())

train_ds = train_ds.apply(tf.data.experimental.assert_cardinality(len(train_gen)))
val_ds = val_ds.apply(tf.data.experimental.assert_cardinality(len(val_gen)))

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=args.cppath,
                                                 save_weights_only=True,
                                                 verbose=0,
                                                 )

if args.logdir is None:
    log_dir = "logs/fit/" + f"l{args.lr}m{args.nmels}f{args.nfft}w{args.window}n{args.words}ww{args.whichword}_" + datetime.datetime.now().strftime("%m%d-%H%M")
else:
    log_dir = args.logdir

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
warnings.filterwarnings(action='ignore', category=FutureWarning)
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=int(args.epochs),
    batch_size=args.batchsize,
    callbacks=[LambdaCallback(on_epoch_end=lambda epoch, logs: train_gen.on_epoch_end()),
               LambdaCallback(on_epoch_end=lambda epoch, logs: val_gen.on_epoch_end()),
               cp_callback,
               tensorboard_callback])  # noqa

