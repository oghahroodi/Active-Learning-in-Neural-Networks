from init import *
from train import *
from models import *


def get_unlabeled(X_train, labeled_idx):
    return np.arange(X_train.shape[0])[np.logical_not(np.in1d(np.arange(X_train.shape[0]), labeled_idx))]


class Query:

    def __init__(self, model, input_shape=(28, 28), num_labels=10, gpu=1):
        self.model = model
        self.input_shape = input_shape
        self.num_labels = num_labels
        self.gpu = gpu

    def query(self, X_train, Y_train, labeled_idx, amount):
        return NotImplemented

    def update_model(self, new_model):
        del self.model
        gc.collect()
        self.model = new_model


class Random_Sampling(Query):

    def __init__(self, model, input_shape, num_labels, gpu):
        super().__init__(model, input_shape, num_labels, gpu)

    def query(self, X_train, Y_train, labeled_idx, amount):
        unlabeled_idx = get_unlabeled(X_train, labeled_idx)
        return np.hstack((labeled_idx, np.random.choice(unlabeled_idx, amount, replace=False)))


class Uncertainty(Query):

    def __init__(self, model, input_shape, num_labels, gpu):
        super().__init__(model, input_shape, num_labels, gpu)

    def query(self, X_train, Y_train, labeled_idx, amount):

        unlabeled_idx = get_unlabeled(X_train, labeled_idx)
        predictions = self.model.predict(X_train[unlabeled_idx, :])

        unlabeled_predictions = np.amax(predictions, axis=1)

        selected_indices = np.argpartition(
            unlabeled_predictions, amount)[:amount]
        return np.hstack((labeled_idx, unlabeled_idx[selected_indices]))


class Uncertainty_Entropy(Query):

    def __init__(self, model, input_shape, num_labels, gpu):
        super().__init__(model, input_shape, num_labels, gpu)

    def query(self, X_train, Y_train, labeled_idx, amount):

        unlabeled_idx = get_unlabeled(X_train, labeled_idx)
        predictions = self.model.predict(X_train[unlabeled_idx, :])

        unlabeled_predictions = np.sum(
            predictions * np.log(predictions + 1e-10), axis=1)

        selected_indices = np.argpartition(
            unlabeled_predictions, amount)[:amount]
        return np.hstack((labeled_idx, unlabeled_idx[selected_indices]))


class BayesianـUncertainty(Query):

    def __init__(self, model, input_shape, num_labels, gpu):
        super().__init__(model, input_shape, num_labels, gpu)

        self.T = 20

    def dropout_predict(self, data):

        f = K.function([self.model.layers[0].input, K.learning_phase()],
                       [self.model.layers[-1].output])
        predictions = np.zeros((self.T, data.shape[0], self.num_labels))
        for t in range(self.T):
            predictions[t, :, :] = f([data, 1])[0]

        final_prediction = np.mean(predictions, axis=0)
        prediction_uncertainty = np.std(predictions, axis=0)

        return final_prediction, prediction_uncertainty

    def query(self, X_train, Y_train, labeled_idx, amount):

        unlabeled_idx = get_unlabeled(X_train, labeled_idx)

        predictions = np.zeros((unlabeled_idx.shape[0], self.num_labels))
        uncertainties = np.zeros((unlabeled_idx.shape[0], self.num_labels))
        i = 0
        split = 128
        while i < unlabeled_idx.shape[0]:

            if i+split > unlabeled_idx.shape[0]:
                preds, unc = self.dropout_predict(
                    X_train[unlabeled_idx[i:], :])
                predictions[i:] = preds
                uncertainties[i:] = unc
            else:
                preds, unc = self.dropout_predict(
                    X_train[unlabeled_idx[i:i+split], :])
                predictions[i:i+split] = preds
                uncertainties[i:i+split] = unc
            i += split

        unlabeled_predictions = np.amax(predictions, axis=1)
        selected_indices = np.argpartition(
            unlabeled_predictions, amount)[:amount]
        return np.hstack((labeled_idx, unlabeled_idx[selected_indices]))


class BayesianـUncertaintyـEntropy(Query):

    def __init__(self, model, input_shape, num_labels, gpu):
        super().__init__(model, input_shape, num_labels, gpu)

        self.T = 100

    def dropout_predict(self, data):

        f = K.function([self.model.layers[0].input, K.learning_phase()],
                       [self.model.layers[-1].output])
        predictions = np.zeros((self.T, data.shape[0], self.num_labels))
        for t in range(self.T):
            predictions[t, :, :] = f([data, 1])[0]

        final_prediction = np.mean(predictions, axis=0)
        prediction_uncertainty = np.std(predictions, axis=0)

        return final_prediction, prediction_uncertainty

    def query(self, X_train, Y_train, labeled_idx, amount):

        unlabeled_idx = get_unlabeled(X_train, labeled_idx)

        predictions = np.zeros((unlabeled_idx.shape[0], self.num_labels))
        i = 0
        while i < unlabeled_idx.shape[0]:

            if i+1000 > unlabeled_idx.shape[0]:
                preds, _ = self.dropout_predict(X_train[unlabeled_idx[i:], :])
                predictions[i:] = preds
            else:
                preds, _ = self.dropout_predict(
                    X_train[unlabeled_idx[i:i+1000], :])
                predictions[i:i+1000] = preds

            i += 1000

        unlabeled_predictions = np.sum(
            predictions * np.log(predictions + 1e-10), axis=1)
        selected_indices = np.argpartition(
            unlabeled_predictions, amount)[:amount]
        return np.hstack((labeled_idx, unlabeled_idx[selected_indices]))


class Adversarial(Query):

    def __init__(self, model, input_shape, num_labels, gpu):
        super().__init__(model, input_shape, num_labels, gpu)

    def query(self, X_train, Y_train, labeled_idx, amount):

        unlabeled_idx = get_unlabeled(X_train, labeled_idx)
        unlabeled = X_train[unlabeled_idx]

        keras_wrapper = KerasModelWrapper(self.model)
        sess = K.get_session()
        deep_fool = DeepFool(keras_wrapper, sess=sess)
        deep_fool_params = {'over_shoot': 0.02,
                            'clip_min': 0.,
                            'clip_max': 1.,
                            'nb_candidate': Y_train.shape[1],
                            'max_iter': 10}
        true_predictions = np.argmax(
            self.model.predict(unlabeled, batch_size=256), axis=1)
        adversarial_predictions = np.copy(true_predictions)
        while np.sum(true_predictions != adversarial_predictions) < amount:
            adversarial_images = np.zeros(unlabeled.shape)
            for i in range(0, unlabeled.shape[0], 100):
                print("At {i} out of {n}".format(i=i, n=unlabeled.shape[0]))
                if i+100 > unlabeled.shape[0]:
                    adversarial_images[i:] = deep_fool.generate_np(
                        unlabeled[i:], **deep_fool_params)
                else:
                    adversarial_images[i:i+100] = deep_fool.generate_np(
                        unlabeled[i:i+100], **deep_fool_params)
            pertubations = adversarial_images - unlabeled
            norms = np.linalg.norm(np.reshape(
                pertubations, (unlabeled.shape[0], -1)), axis=1)
            adversarial_predictions = np.argmax(self.model.predict(
                adversarial_images, batch_size=256), axis=1)
            norms[true_predictions == adversarial_predictions] = np.inf
            deep_fool_params['max_iter'] *= 2

        selected_indices = np.argpartition(norms, amount)[:amount]

        del keras_wrapper
        del deep_fool
        gc.collect()

        return np.hstack((labeled_idx, unlabeled_idx[selected_indices]))


class DiscriminativeAutoencoderSampling(Query):
    """
    An implementation of DAL (discriminative active learning), using an autoencoder embedding as our representation.
    """

    def __init__(self, model, input_shape, num_labels, gpu):
        super().__init__(model, input_shape, num_labels, gpu)

        self.sub_batches = 10
        self.autoencoder = None
        self.embedding = None

    def query(self, X_train, Y_train, labeled_idx, amount):

        if self.autoencoder is None:
            self.autoencoder = get_autoencoder_model(input_shape=(28, 28, 1))
            self.autoencoder.compile(optimizer=optimizers.Adam(
                lr=0.0003), loss='binary_crossentropy')
            self.autoencoder.fit(X_train, X_train,
                                 epochs=30,
                                 batch_size=256,
                                 shuffle=True,
                                 verbose=2)
            encoder = Model(self.autoencoder.input,
                            self.autoencoder.get_layer('embedding').input)
            self.embedding = encoder.predict(
                X_train.reshape((-1, 28, 28, 1)), batch_size=1024)

        # subsample from the unlabeled set:
        unlabeled_idx = get_unlabeled(X_train, labeled_idx)
        unlabeled_idx = np.random.choice(unlabeled_idx, np.min(
            [labeled_idx.shape[0]*10, unlabeled_idx.size]), replace=False)

        # iteratively sub-sample using the discriminative sampling routine:
        labeled_so_far = 0
        sub_sample_size = int(amount / self.sub_batches)
        while labeled_so_far < amount:
            if labeled_so_far + sub_sample_size > amount:
                sub_sample_size = amount - labeled_so_far

            model = train_discriminative_model(
                self.embedding[labeled_idx], self.embedding[unlabeled_idx], self.embedding[0].shape, gpu=self.gpu)
            predictions = model.predict(self.embedding[unlabeled_idx])
            selected_indices = np.argpartition(
                predictions[:, 1], -sub_sample_size)[-sub_sample_size:]
            labeled_idx = np.hstack(
                (labeled_idx, unlabeled_idx[selected_indices]))
            labeled_so_far += sub_sample_size
            unlabeled_idx = get_unlabeled(X_train, labeled_idx)
            unlabeled_idx = np.random.choice(unlabeled_idx, np.min(
                [labeled_idx.shape[0]*10, unlabeled_idx.size]), replace=False)

            # delete the model to free GPU memory:
            del model
            gc.collect()

        return labeled_idx


class CoreSetSampling(Query):
    """
    An implementation of the greedy core set query strategy.
    """

    def __init__(self, model, input_shape, num_labels, gpu):
        super().__init__(model, input_shape, num_labels, gpu)

    def greedy_k_center(self, labeled, unlabeled, amount):

        greedy_indices = []

        # get the minimum distances between the labeled and unlabeled examples (iteratively, to avoid memory issues):
        min_dist = np.min(distance_matrix(labeled[0, :].reshape(
            (1, labeled.shape[1])), unlabeled), axis=0)
        min_dist = min_dist.reshape((1, min_dist.shape[0]))
        for j in range(1, labeled.shape[0], 100):
            if j + 100 < labeled.shape[0]:
                dist = distance_matrix(labeled[j:j+100, :], unlabeled)
            else:
                dist = distance_matrix(labeled[j:, :], unlabeled)
            min_dist = np.vstack(
                (min_dist, np.min(dist, axis=0).reshape((1, min_dist.shape[1]))))
            min_dist = np.min(min_dist, axis=0)
            min_dist = min_dist.reshape((1, min_dist.shape[0]))

        # iteratively insert the farthest index and recalculate the minimum distances:
        farthest = np.argmax(min_dist)
        greedy_indices.append(farthest)
        for i in range(amount-1):
            dist = distance_matrix(
                unlabeled[greedy_indices[-1], :].reshape((1, unlabeled.shape[1])), unlabeled)
            min_dist = np.vstack(
                (min_dist, dist.reshape((1, min_dist.shape[1]))))
            min_dist = np.min(min_dist, axis=0)
            min_dist = min_dist.reshape((1, min_dist.shape[0]))
            farthest = np.argmax(min_dist)
            greedy_indices.append(farthest)

        return np.array(greedy_indices)

    def query(self, X_train, Y_train, labeled_idx, amount):

        unlabeled_idx = get_unlabeled(X_train, labeled_idx)

        # use the learned representation for the k-greedy-center algorithm:
        representation_model = Model(
            inputs=self.model.input, outputs=self.model.get_layer('softmax').input)
        representation = representation_model.predict(X_train, verbose=0)
        new_indices = self.greedy_k_center(
            representation[labeled_idx, :], representation[unlabeled_idx, :], amount)
        return np.hstack((labeled_idx, unlabeled_idx[new_indices]))


class EGL(Query):

    def __init__(self, model, input_shape, num_labels, gpu):
        super().__init__(model, input_shape, num_labels, gpu)

    def get_grad(self, unlabeled, n_classes):

        self.input_placeholder = K.placeholder(
            self.model.get_input_shape_at(0))
        self.output_placeholder = K.placeholder(
            self.model.get_output_shape_at(0))
        predict = self.model.call(self.input_placeholder)
        loss = K.mean(categorical_crossentropy(
            self.output_placeholder, predict))
        weights = [tensor for tensor in self.model.trainable_weights]
        gradient = self.model.optimizer.get_gradients(loss, weights)
        gradient_flat = [K.flatten(x) for x in gradient]
        gradient_flat = K.concatenate(gradient_flat)
        gradient_length = K.sum(K.square(gradient_flat))
        self.get_gradient_length = K.function([K.learning_phase(
        ), self.input_placeholder, self.output_placeholder], [gradient_length])

        unlabeled_predictions = self.model.predict(unlabeled)
        egls = np.zeros(unlabeled.shape[0])
        for i in range(n_classes):
            calculated_so_far = 0
            while calculated_so_far < unlabeled_predictions.shape[0]:
                if calculated_so_far + 100 >= unlabeled_predictions.shape[0]:
                    next = unlabeled_predictions.shape[0] - calculated_so_far
                else:
                    next = 100

                labels = np.zeros((next, n_classes))
                labels[:, i] = 1
                grads = self.get_gradient_length(
                    [0, unlabeled[calculated_so_far:calculated_so_far+next, :], labels])[0]
                grads *= unlabeled_predictions[calculated_so_far:calculated_so_far+next, i]
                egls[calculated_so_far:calculated_so_far+next] += grads

                calculated_so_far += next

        return egls

    def query(self, X_train, Y_train, labeled_idx, amount):

        unlabeled_idx = get_unlabeled(X_train, labeled_idx)
        n_classes = Y_train.shape[1]
        egls = self.get_grad(X_train[unlabeled_idx], n_classes)
        selected_indices = np.argpartition(egls, -amount)[-amount:]
        return np.hstack((labeled_idx, unlabeled_idx[selected_indices]))


class CombinedSampling(Query):
    """
    An implementation of a query strategy which naively combines two given query strategies, sampling half of the batch
    from one strategy and the other half from the other strategy.
    """

    def __init__(self, model, input_shape, num_labels, method1, method2, gpu):
        super().__init__(model, input_shape, num_labels, gpu)
        self.method1 = method1(model, input_shape, num_labels, gpu)
        self.method2 = method2(model, input_shape, num_labels, gpu)

    def query(self, X_train, Y_train, labeled_idx, amount):
        labeled_idx = self.method1.query(
            X_train, Y_train, labeled_idx, int(amount/2))
        return self.method2.query(X_train, Y_train, labeled_idx, int(amount/2))

    def update_model(self, new_model):
        del self.model
        gc.collect()
        self.model = new_model
        self.method1.update_model(new_model)
        self.method2.update_model(new_model)
