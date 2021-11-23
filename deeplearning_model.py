import warnings
from keras.layers.pooling import MaxPool1D
from keras.models import Model, Sequential, model_from_json
from keras.layers import Input, Embedding, LSTM, concatenate, Bidirectional, Dot
from keras import backend as K
from keras import optimizers
from keras.callbacks import EarlyStopping
import numpy as np
import transformers
from matplotlib import pyplot
from keras.models import load_model
transformers.logging.set_verbosity_error()
warnings.filterwarnings("ignore")


class SiameseLSTM:
    def __init__(self, vocab_size, seq_dim=100, embedding_dim=300, lstm_unit=50, batch_size=16, dense_unit=128, epoch=20, lr=0.01):
        self.embedding_dim = embedding_dim
        self.seq_dim = seq_dim
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.lstm_unit = lstm_unit
        self.dense_unit = dense_unit
        self.epoch = epoch
        self.lr = lr
        self.model = self.build_model()

    def build_model(self):
        s1 = Input(shape=(self.seq_dim,), dtype='int32')
        s2 = Input(shape=(self.seq_dim,), dtype='int32')
        m = Sequential()
        m.add(Embedding(self.vocab_size+1, self.embedding_dim,
              input_length=self.seq_dim))
        m.add(Bidirectional(LSTM(self.lstm_unit, dropout=0.2)))
        d = Dot(axes=1, normalize=True)([m(s1), m(s2)])
        # d = Dense(self.dense_unit, activation='sigmoid')(d)

        model = Model(inputs=[s1, s2], outputs=d)
        opt = optimizers.Adam(learning_rate=self.lr)
        model.compile(optimizer=opt,
                      loss="binary_crossentropy", metrics=['acc'])
        return model

    def train(self, x1, x2, train_label):
        train_label = np.array(train_label)

        es_callback = EarlyStopping(
            monitor='val_loss', patience=3, verbose=1)
        history = self.model.fit([x1, x2], train_label, batch_size=self.batch_size,
                                 epochs=self.epoch, verbose=1, shuffle=True, validation_split=0.3, callbacks=[es_callback])
        pyplot.plot(history.history['loss'])
        pyplot.plot(history.history['val_loss'])
        pyplot.title('model train vs validation loss')
        pyplot.ylabel('loss')
        pyplot.xlabel('epoch')
        pyplot.legend(['train', 'validation'], loc='upper right')
        pyplot.savefig('./train_val_loss.png')
        pyplot.close()

    def predict_proba(self, x1, x2):
        p = self.model.predict([x1, x2])
        return p.flatten()

    def predict(self, x1, x2):
        p = self.model.predict([x1, x2])
        return p.round().flatten()

    def save(self, filename='./SiameseLSTM.h5'):
        self.model.save(filename)

    def load(self, filename='./SiameseLSTM.h5'):
        try:
            self.model = load_model(filename)
        except Exception as e:
            print(filename[2:] + " model not found")

    def cosine_distance(vests):
        x, y = vests
        x = K.l2_normalize(x, axis=-1)
        y = K.l2_normalize(y, axis=-1)
        return -K.mean(x * y, axis=-1, keepdims=True)

    def cos_dist_output_shape(shapes):
        shape1, shape2 = shapes
        return (shape1[0], 1)
