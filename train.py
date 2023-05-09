from generators.TTSGenre import TTSGenre
from models.models import Kell2018, Kell2018small, ASTClassifierConnected
from utils.losses import Unconcerned_CCE
from keras.metrics import TopKCategoricalAccuracy, CategoricalAccuracy
import tensorflow as tf
import warnings, sys, argparse
from keras.optimizers import Adam
from keras.callbacks import LambdaCallback
from utils.metrics import UnconcernedAccuracy
import datetime
import tensorflow_addons as tfa

parser = argparse.ArgumentParser()

parser.add_argument('-l', '--libripath', help='Path to folder with all LibriTTS subsets')
parser.add_argument('-g', '--gtzanpath', help='Path to folder with GTZAN dataset')
parser.add_argument('-b', '--batchsize', default=64)
parser.add_argument('-e', '--epochs')
parser.add_argument('--lr', default=1e-3)
parser.add_argument('--glr', default=None)
parser.add_argument('--wlr', default=None)
parser.add_argument('--wbatch', default=None)
parser.add_argument('--gbatch', default=None)
parser.add_argument('--nmels', default=512)
parser.add_argument('--nfft', default=2048)
parser.add_argument('--hop', default=44)
parser.add_argument('--window', default=1)
parser.add_argument('--words', default=200)
parser.add_argument('--whichword', default=2)
parser.add_argument('--quiet', action='store_true', help='Silences the mid-training messages')
parser.add_argument('--logdir', default=None)
parser.add_argument('--model', choices=['kell', 'kellsmall', 'astcon'])
parser.add_argument('--patchf', default=16)
parser.add_argument('--patcht', default=16)
parser.add_argument('--overf', default=6)
parser.add_argument('--overt', default=6)
parser.add_argument('--projdim', default=256)
parser.add_argument('--numheads', default=12)
parser.add_argument('--hiddenu', default=256)
parser.add_argument('--ntlayers', default=12)
parser.add_argument('--mlpheadu', default=256)
parser.add_argument('--norm', choices=['smaple', 'batch', 'none'], default='sample')
parser.add_argument('--augment', action='store_true')
parser.add_argument('--urbanpath')
parser.add_argument('--pretrained', action='store_true')



args = parser.parse_args()

train_gen = TTSGenre(
    libri_path=args.libripath,
    gtzan_path=args.gtzanpath,
    batch_size=int(args.batchsize),
    n_mels=int(args.nmels),
    n_fft=int(args.nfft),
    hop=int(args.hop),
    window_s=float(args.window),
    words=int(args.words),
    which_word=int(args.whichword),
    quiet=args.quiet,
    mode='train',
    augment=args.augment,
    norm=args.norm,
    urbanpath=args.urbanpath,
    shuffle=True,
    wbatch = args.wbatch,
    gbatch = args.gbatch
)

val_gen = TTSGenre(
    libri_path=args.libripath,
    gtzan_path=args.gtzanpath,
    batch_size=int(args.batchsize),
    n_mels=int(args.nmels),
    n_fft=int(args.nfft),
    hop=int(args.hop),
    window_s=float(args.window),
    words=train_gen.get_words(),
    quiet=args.quiet,
    mode='val',
    augment=False,
    norm=args.norm,
    shuffle=True
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


if args.model == 'kell':
    model = Kell2018(input_shape, wout_shape[0], gout_shape[0], pretrained=args.pretrained)
elif args.model == 'kellsmall':
    model = Kell2018small(input_shape, wout_shape[0], gout_shape[0])
elif args.model == 'astcon':
    model = ASTClassifierConnected(
        input_shape=input_shape,
        patch_size=(int(args.patchf), int(args.patcht)),
        overlap=(int(args.overf), int(args.overt)),
        projection_dims=int(args.projdim),
        num_heads=int(args.numheads),
        hidden_units=int(args.hiddenu),
        n_transformer_layers=int(args.ntlayers),
        mlp_head_units=int(args.mlpheadu),
        wout_classes=wout_shape[0],
        gout_classes=gout_shape[0]
    )
else:
    raise ValueError(f'model {args.model} not supported')

glr = args.glr
wlr = args.wlr

if glr is None:
    glr = args.lr
if wlr is None:
    wlr = args.lr

if args.model=='kell' or args.model=='kellsmall':
    mutual = list(range(0, 8))
    word = [8, 10, 12, 14, 16, 18, 20, 22]
    genre = [9, 11, 13, 15, 17, 19, 21, 23]

    optimizers_and_layers = [
        (Adam(learning_rate=float(args.lr)), [model.layers[i] for i in mutual]),
        (Adam(learning_rate=float(wlr)), [model.layers[i] for i in word]),
        (Adam(learning_rate=float(glr)), [model.layers[i] for i in genre])
    ]
    optimizer = tfa.optimizers.MultiOptimizer(optimizers_and_layers)
else:
    optimizer = Adam(learning_rate=float(args.lr))

model.compile(loss={'wout': Unconcerned_CCE(), 'gout': Unconcerned_CCE()},
              optimizer=optimizer,
              metrics={
                  'wout': [UnconcernedAccuracy()],
                  'gout': [UnconcernedAccuracy()]
              },
              run_eagerly=True
              )
print(model.summary())

train_ds = train_ds.apply(tf.data.experimental.assert_cardinality(len(train_gen)))
val_ds = val_ds.apply(tf.data.experimental.assert_cardinality(len(val_gen)))

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=f"{args.model}l{args.lr}{args.wlr}{args.glr}m{args.nmels}f{args.nfft}w{args.window}n{args.words}ww{args.whichword}h{args.hop}_" + datetime.datetime.now().strftime("%m%d-%H%M") + '.ckpt',
                                                 save_weights_only=True,
                                                 verbose=0,
                                                 )

if args.logdir is None:
    log_dir = "logs/fit/" + f"{args.model}l{args.lr}{args.wlr}{args.glr}m{args.nmels}f{args.nfft}w{args.window}n{args.words}ww{args.whichword}h{args.hop}_" + datetime.datetime.now().strftime("%m%d-%H%M")
else:
    log_dir = args.logdir

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
warnings.filterwarnings(action='ignore', category=FutureWarning)
warnings.filterwarnings(action='ignore', category=UserWarning)
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=int(args.epochs),
    batch_size=args.batchsize,
    callbacks=[LambdaCallback(on_epoch_end=lambda epoch, logs: train_gen.on_epoch_end()),
               LambdaCallback(on_epoch_end=lambda epoch, logs: val_gen.on_epoch_end()),
               cp_callback,
               tensorboard_callback])  # noqa

