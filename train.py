from generators.FSD50K import FSD50K
from matplotlib import pyplot as plt
import librosa.display
import tensorflow as tf

"""

gen = FSD50K('/Volumes/Folder1/', batch_size=1, n_mels=256)
data = gen[0]
print(data[0].shape)
print(data[1].shape)

librosa.display.specshow(data[0][0,:,:,0], sr=22050, hop_length=int(0.002*22050), x_axis='time', y_axis='mel')
plt.colorbar()
plt.show()

"""

if tf.test.gpu_device_name():
    print(f'GPU: {tf.test.gpu_device_name()}')
else:
    print("no GPU used")

model = tf.keras.applications.MobileNetV3Small(include_top=True, weights=None, input_shape=(256, 501, 1), classes=200, include_preprocessing=False)
train_gen = FSD50K('/Volumes/Folder1/', batch_size=64, n_mels=256)
model.compile(loss=tf.keras.losses.categorical_crossentropy)
print("model compiled")
model.fit(x=train_gen, epochs=1, verbose=1)
