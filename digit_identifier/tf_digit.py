import pickle, gzip, numpy, theano
import theano.tensor as T

def shared_dataset(data_xy):
    data_x, data_y = data_xy
    shared_x = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX))
    shared_y = theano.shared(numpy.asarray(data_y, dtype=theano.config.floatX))

    return shared_x, T.cast(shared_y, 'int32')

def main():
    f = gzip.open('mnist.pkl.gz' , 'rb')
    train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
    f.close()
    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)
    x = test_set_x.get_value()
    for y in range(0, 250):
        print (x.item(y))
    batch_size = 500

main()
