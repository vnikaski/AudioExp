from generators.TTSGenre import TTSGenre
from models.models import Kell2018
from utils.losses import Unconcerned_CCE
import tensorflow as tf
import warnings, sys, argparse
from keras.optimizers import Adam
from utils.metrics import UnconcernedAccuracy

parser = argparse.ArgumentParser()

parser.add_argument('-l', '--libripath', help='Path to folder with all LibriTTS subsets')
parser.add_argument('-g', '--gtzanpath', help='Path to folder with GTZAN dataset')
parser.add_argument('-b', '--batchsize')
parser.add_argument('-e', '--epochs')
parser.add_argument('--lr', default=1e-3)
parser.add_argument('--quiet', action='store_true', help='Silences the mid-training messages')


args = parser.parse_args()



gen = TTSGenre(libri_path=args.libripath,
               gtzan_path=args.gtzanpath,
               batch_size=int(args.batchsize),
               quiet=args.quiet
               )

xs, ys = gen.get_sample()
print(xs.shape)
input_shape = xs.shape[1:]
wout_shape = ys['wout'].shape[1:]
gout_shape = ys['gout'].shape[1:]


output_signature = (tf.TensorSpec(shape=(None, *xs.shape[1:])),
                    {'wout': tf.TensorSpec(shape=(None, *wout_shape)),
                     'gout': tf.TensorSpec(shape=(None, *gout_shape))})

gen.output_signature = output_signature


ds = tf.data.Dataset.from_generator(gen,
                                    output_signature=output_signature)
model = Kell2018(input_shape, wout_shape[0], gout_shape[0])
model.compile(loss={'wout':Unconcerned_CCE(), 'gout':Unconcerned_CCE()},
              optimizer=Adam(learning_rate=float(args.lr)),
              # metrics=[UnconcernedAccuracy()],
              )
print(model.summary())
ds = ds.apply(tf.data.experimental.assert_cardinality(len(gen)))
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath='./checkpoints',
                                                 save_weights_only=True,
                                                 verbose=1)
warnings.filterwarnings(action='ignore', category=FutureWarning)
model.fit(ds, epochs=int(args.epochs), batch_size=args.batchsize)  # noqa

