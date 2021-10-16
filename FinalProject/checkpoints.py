from init import *


class DiscriminativeEarlyStopping(Callback):
    """
    A custom callback for discriminative active learning, to stop the training a little bit before the classifier is
    able to get 100% accuracy on the training set. This makes sure examples which are similar to ones already in the
    labeled set won't have a very high confidence.
    """

    def __init__(self, monitor='acc', threshold=0.98, verbose=0):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.threshold = threshold
        self.verbose = verbose
        self.improved = 0

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)

        if current > self.threshold:
            if self.verbose > 0:
                print("Epoch {e}: early stopping at accuracy {a}".format(
                    e=epoch, a=current))
            self.model.stop_training = True


class Checkpoint(Callback):

    def __init__(self, filepath, monitor='val_acc', delay=0, verbose=0, weights=False):

        super(Checkpoint, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.delay = delay
        if self.monitor == 'val_acc':
            self.best = -np.Inf
        else:
            self.best = np.Inf
        self.weights = weights

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        if self.monitor == 'val_acc':
            current = logs.get(self.monitor)
            if current >= self.best and epoch > self.delay:
                if self.verbose > 0:
                    print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                          ' saving model to %s'
                          % (epoch, self.monitor, self.best,
                             current, self.filepath))
                self.best = current
                if self.weights:
                    self.model.save_weights(self.filepath, overwrite=True)
                else:
                    self.model.save(self.filepath, overwrite=True)
        else:
            current = logs.get(self.monitor)
            if current <= self.best and epoch > self.delay:
                if self.verbose > 0:
                    print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                          ' saving model to %s'
                          % (epoch, self.monitor, self.best,
                             current, self.filepath))
                self.best = current
                if self.weights:
                    self.model.save_weights(self.filepath, overwrite=True)
                else:
                    self.model.save(self.filepath, overwrite=True)
