########################################################################################################################
# ML R&D
########################################################################################################################

import csv
import numpy as np
np.random.seed(2)
from tensorflow import keras
from tensorflow.keras import layers

from pdb import set_trace as debug

# Raise Failure when we produce no trades
class Failure(Exception):
    pass


def main(symbol_set, train_pct=0.9):
    # Read in data set
    with open('ml_research/datasets/{}.csv'.format(symbol_set), 'r', newline='') as file:
        ref_data = list(csv.reader(file, delimiter=','))
    data = list(map(lambda row: row[2:], ref_data[1:]))

    # Determine where to split into train and test
    mark = int(len(ref_data) * train_pct)

    # Split into train and test
    train, test = data[:mark], data[mark:]
    np.random.shuffle(train)

    # Reduce number of negative examples in training set as to balance +/- counts in training set
    # In application the ratio is 90-10, but this causes model during training to just predict -, and we need both
    num_pos = len(list(filter(lambda x: x[-1] == '1', train)))
    print(len(train), 'total samples\t', num_pos, 'from the positive class')

    # Number of total samples in trimmed data set (defined as function of num_pos)
    balance_controller = 0.5
    adj_len = num_pos / balance_controller

    while len(train) > adj_len:
        idx = int(np.random.uniform(0, len(train) - 1))
        if train[idx][-1] == '0':
            train.pop(idx)
    print('final length:', len(train))

    x_train, x_test = list(map(lambda x: x[:-1], train)), list(map(lambda x: x[:-1], test))
    y_train, y_test = np.array(list(map(lambda x: int(x[-1]), train))), np.array(list(map(lambda x: int(x[-1]), test)))
    x_train = np.array(list(map(lambda row: np.array(list(map(float, row))), x_train)))
    x_test = np.array(list(map(lambda row: np.array(list(map(float, row))), x_test)))
    print(len(list(filter(lambda x: x == 1, y_test))), 'samples from positive class in test data')

    # model = MLP(hidden_layer_sizes=(50, 1, 2), learning_rate_init=0.01, activation='relu', solver='sgd',
    #             momentum=0.5, verbose=0, max_iter=x_train.shape[0], tol=10**-6, warm_start=False, random_state=2)

    # Build the model
    drop_out = 0.1
    model = keras.Sequential([
        layers.Dense(25),
        layers.Activation('relu'),
        layers.Dropout(drop_out),
        layers.Dense(3),
        layers.Activation('sigmoid'),
        layers.Dropout(drop_out),
        layers.Dense(1, activation='sigmoid')
    ])
    sgd = keras.optimizers.SGD(learning_rate=0.025, momentum=0.1)

    model.compile(sgd, loss='binary_crossentropy', metrics=[keras.metrics.Precision()])
    model.fit(x_train, y_train, batch_size=2, epochs=50, validation_data=[x_test, y_test])

    # Compile list of trades
    trades, preds = [], model.predict(x_test)
    for i in range(len(preds)):
        ref_idx = len(ref_data) - 1 - i
        pred_idx = len(preds) - 1 - i
        predicted_class = 1 if preds[pred_idx][0] >= 0.5 else 0
        if predicted_class == 1:
            trades.append(ref_data[ref_idx][:2] + [predicted_class])
    trades.reverse()
    if not trades:
        print('No trades')
        raise Failure
    print(len(trades), 'predicted positive labels')

    # Save trades to file for analysis
    for symbol in set(list(map(lambda x: x[1], trades))):
        sub_trades = list(map(lambda x: [x[0], x[2]], list(filter(lambda y: y[1] == symbol, trades))))
        with open('ml_research/tradefiles/{}.csv'.format(symbol), 'w', newline='') as file:
            csv.writer(file, delimiter=',').writerows(sub_trades)


if __name__ == '__main__':
    raise NotImplemented
