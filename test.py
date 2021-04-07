import os
import pandas as pd
import nibabel as nib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

demo = pd.read_csv("qit_recoveryrate.csv"); X = demo.iloc[:,11:-1]; y = demo.iloc[:,-1];

# Random Forest

clf = RandomForestClassifier()
scores = cross_val_score(clf, X, y, cv=2)
clf.fit(X, y)
feature_importances = pd.DataFrame(clf.feature_importances_, index = X.columns, columns=['importance']).sort_values('importance', ascending=False)


# Computer Vision CNN

preprocF = os.listdir("/ifs/loni/faculty/farshid/img/shared.data/TBI/raw/TRACKTBI/diffusion/preproc")
matching = []; nifty = []
for i in range(len(demo["RID"])):
	path = ["/ifs/loni/faculty/farshid/img/shared.data/TBI/raw/TRACKTBI/diffusion/preproc"+"/"+j for j in preprocF if df2["RID"][i] in j][0] + "/dmri/dwi.nii"
	matching.append(path) 
	nifty.append(nib.load(path).get_data())
def get_model(width=128, height=128, depth=64):
	inputs = keras.Input((width, height, depth, 1))
	x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(inputs)
	x = layers.MaxPool3D(pool_size=2)(x)
	x = layers.BatchNormalization()(x)
	x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(x)
	x = layers.MaxPool3D(pool_size=2)(x)
	x = layers.BatchNormalization()(x)
	x = layers.Conv3D(filters=128, kernel_size=3, activation="relu")(x)
	x = layers.MaxPool3D(pool_size=2)(x)
	x = layers.BatchNormalization()(x)
	x = layers.Conv3D(filters=256, kernel_size=3, activation="relu")(x)
	x = layers.MaxPool3D(pool_size=2)(x)
	x = layers.BatchNormalization()(x)
	x = layers.GlobalAveragePooling3D()(x)
	x = layers.Dense(units=512, activation="relu")(x)
	x = layers.Dropout(0.3)(x)
	outputs = layers.Dense(units=1, activation="sigmoid")(x)
	model = keras.Model(inputs, outputs, name="3dcnn")
	return model
model = get_model(width=128, height=128, depth=64)
model.summary()

# 1D GAN multilabel

input_length = 1680
batch_size = 1
def build_generator():
	generator = tf.keras.Sequential()
	generator.add(layers.Dense(input_length*batch_size, input_shape=[input_length, 1],dtype='float32'))
	generator.add(layers.Conv1D(filters = 32,kernel_size = 3, padding = 'same'))
	generator.add(layers.LeakyReLU(alpha=0.2))
	generator.add(layers.Conv1D(filters = 32,kernel_size = 3, padding = 'same'))
	generator.add(layers.LeakyReLU(alpha=0.2))
	generator.add(layers.Conv1D(filters = 32,kernel_size = 3, padding = 'same'))
	generator.add(layers.LeakyReLU(alpha=0.2))
	generator.add(layers.Conv1D(filters = 32,kernel_size = 3, padding = 'same'))
	generator.add(layers.LeakyReLU(alpha=0.2))
	generator.add(layers.Conv1D(filters = 32,kernel_size = 3, padding = 'same'))
	generator.add(layers.LeakyReLU(alpha=0.2))
	generator.add(layers.Conv1D(filters = 32,kernel_size = 3, padding = 'same'))
	generator.add(layers.LeakyReLU(alpha=0.2))
	generator.add(layers.Conv1D(filters = 32,kernel_size = 5, padding = 'same'))
	generator.add(layers.LeakyReLU(alpha=0.2))
	generator.add(layers.Conv1D(filters = 32,kernel_size = 5, padding = 'same'))
	generator.add(layers.LeakyReLU(alpha=0.2))
	generator.add(layers.Conv1D(filters = 1,kernel_size = 5, activation = 'tanh', padding = 'same'))
	return generator

def build_discriminator():
	discriminator = tf.keras.Sequential()
	discriminator.add(layers.Conv1D(filters = 32,kernel_size = 3, padding = 'same',input_shape=[input_length, 1]))
	discriminator.add(layers.LeakyReLU(alpha=0.2))
	discriminator.add(layers.Conv1D(filters = 32,kernel_size = 3, padding = 'same')) 
	discriminator.add(layers.LeakyReLU(alpha=0.2))
	discriminator.add(layers.MaxPooling1D(pool_size=2))
	discriminator.add(layers.Conv1D(filters = 32,kernel_size = 3, padding = 'same')) 
	discriminator.add(layers.LeakyReLU(alpha=0.2))
	discriminator.add(layers.Conv1D(filters = 32,kernel_size = 3, padding = 'same')) 
	discriminator.add(layers.LeakyReLU(alpha=0.2))
	discriminator.add(layers.MaxPooling1D(pool_size=2))
	discriminator.add(layers.Conv1D(filters = 32,kernel_size = 3, padding = 'same')) 
	discriminator.add(layers.LeakyReLU(alpha=0.2))
	discriminator.add(layers.Conv1D(filters = 32,kernel_size = 3, padding = 'same')) 
	discriminator.add(layers.LeakyReLU(alpha=0.2))
	discriminator.add(layers.MaxPooling1D(pool_size=2))
	discriminator.add(layers.Flatten(input_shape=(input_length,)))
	discriminator.add(layers.Dense(64,dtype='float32'))
	discriminator.add(layers.Dropout(0.4))
	discriminator.add(layers.LeakyReLU(alpha=0.2))
	discriminator.add(layers.Dense(1, activation='tanh',dtype='float32'))
	return discriminator
def discriminator_loss(real_output, fake_output, label_batch):
	real_loss = cross_entropy(-0.9*tf.ones_like(real_output), real_output)
	fake_loss = cross_entropy(0.9*tf.ones_like(fake_output), fake_output)
	total_loss = 0.4*real_loss+0.6*fake_loss
	return total_loss
def generator_loss(fake_output, label_batch):
	return cross_entropy(-0.9*tf.ones_like(fake_output), fake_output)
generator_optimizer = tf.keras.optimizers.Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
discriminator_optimizer = tf.keras.optimizers.Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
total_optimizer = tf.keras.optimizers.Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
global_image = None
def train_step(image_batch, label_batch):
	noise =
	print(noise)
	noise = tf.cos(4*noise)
	with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape,tf.GradientTape() as total_tape:
		print("ALOTLAOTLAOTLAOT")
		gen_image = generator(noise, training=True)
		print(gen_image)
		print("BALOTLAOTLAOTLAOT")
		real_output = discriminator(image_batch, training=True)
		print("CALOTLAOTLAOTLAOT")
		fake_output = discriminator(gen_image, training=True)
		print("DALOTLAOTLAOTLAOT")
		gen_loss = generator_loss(fake_output, label_batch)
		print("EALOTLAOTLAOTLAOT")
		disc_loss = discriminator_loss(real_output, fake_output)
		print("FALOTLAOTLAOTLAOT")
		total_loss = tf.tanh(tf.abs(gen_loss)-tf.abs(disc_loss))
		print("GALOTLAOTLAOTLAOT")
	gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
	gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
	gradients_of_total = total_tape.gradient(total_loss, generator.trainable_variables)
	generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
	discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
	total_optimizer.apply_gradients(zip(gradients_of_total,generator.trainable_variables))
	return gen_loss,disc_loss,total_loss

epochs = 20
generator = build_generator(); discriminator = build_discriminator();
for epoch in range(epochs):
	start = time.time()
	cunt = 0
	for (image_batch, label_batch) in zip(X.to_numpy(), y):
		cunt += 1
		gen_loss,disc_loss,total_loss = train_step(tf.constant(image_batch, dtype = 'float32'), tf.constant(label_batch, dtype = 'float32'))
