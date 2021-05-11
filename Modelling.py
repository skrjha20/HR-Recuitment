import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

def build_model(X_train, y_train, X_test, y_test):
    model = Sequential()
    model.add(Dense(64, input_dim=X.shape[1], activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    es = EarlyStopping(monitor='val_loss', mode='min', patience=20, verbose=1)
    history = model.fit(X_train, y_train, epochs=200, batch_size=128, validation_data=(X_test, y_test), callbacks=[es],
                        verbose=0)
    return history, model

def model_loss_plot(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def model_accuracy_plot(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def predict_model(model, X_train, y_train, X_test, y_test):
    train_acc = model.evaluate(X_train, y_train, verbose=0)
    test_acc = model.evaluate(X_test, y_test, verbose=0)
    print('Train: %.3f, Test: %.3f' % (train_acc[1], test_acc[1]))

    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, np.rint(y_pred))
    print(cm)

def predict_test(model, df_test):
    df_test_pred = model.predict(df_test)
    df_test['target'] = np.rint(df_test_pred)
    submit = df_test['target'].reset_index()
    submit.to_csv("submit.csv", index=False)

if __name__ == '__main__':
    df_train = pd.read_csv("df_train.csv")
    df_test = pd.read_csv("df_test.csv")

    df_train = df_train.set_index('enrollee_id')
    df_test = df_test.set_index('enrollee_id')
    df_test = df_test.drop(['target'], axis=1)
    y = df_train['target']
    X = df_train.drop(['target'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    history, model = build_model(X_train, y_train, X_test, y_test)
    predict_model(model, X_train, y_train, X_test, y_test)
    predict_test(model, df_test)
