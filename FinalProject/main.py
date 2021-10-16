from init import *
from methods import *
from models import *
from datasets import *


def arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument('--index', type=int)
    p.add_argument('--dataset', type=str, choices={
                   'mnist', 'cifar10', 'cifar100'})
    p.add_argument('--batch', type=int)
    p.add_argument('--initial', type=int)
    p.add_argument('--iterations', type=int,
                   help="number of active learning batches to sample")
    p.add_argument('--method', type=str,
                   choices={'Random', 'CoreSet',  'Uncertainty', 'Bayesian', 'UncertaintyEntropy', 'BayesianEntropy', 'EGL', 'Adversarial', 'DiscriminativeAE'})
    p.add_argument('--res_folder', type=str)
    p.add_argument('--second_method', type=str,
                   choices={None, 'Random', 'CoreSet', 'CoreSetMIP',  'Uncertainty',
                            'Bayesian', 'UncertaintyEntropy', 'BayesianEntropy', 'EGL', 'Adversarial'},
                   default=None)
    p.add_argument('--initial_label_path', type=str,
                   default=None)
    p.add_argument('--gpu', type=int, default=1)
    args = p.parse_args()
    return args


def evaluate(training_function, X_train, Y_train, X_test, Y_test, checkpoint_path):

    perm = np.random.permutation(X_train.shape[0])
    X_train = X_train[perm]
    Y_train = Y_train[perm]

    X_validation = X_train[:int(0.2*X_train.shape[0])]
    Y_validation = Y_train[:int(0.2*Y_train.shape[0])]
    X_train = X_train[int(0.2*X_train.shape[0]):]
    Y_train = Y_train[int(0.2*Y_train.shape[0]):]

    model = training_function(
        X_train, Y_train, X_validation, Y_validation, checkpoint_path, gpu=args.gpu)

    loss, acc = model.evaluate(X_test, Y_test, verbose=0)

    return acc, model


if __name__ == '__main__':

    args = arg_parser()

    if args.dataset == 'mnist':
        (X_train, Y_train), (X_test, Y_test) = get_mnist()
        num_labels = 10
        if K.image_data_format() == 'channels_last':
            input_shape = (28, 28, 1)
        else:
            input_shape = (1, 28, 28)
        trainer = train_mnist
    if args.dataset == 'cifar10':
        (X_train, Y_train), (X_test, Y_test) = get_cifar10()
        num_labels = 10
        if K.image_data_format() == 'channels_last':
            input_shape = (32, 32, 3)
        else:
            input_shape = (3, 32, 32)
        trainer = train_cifar10
    if args.dataset == 'cifar10_with_pretrain':
        (X_train, Y_train), (X_test, Y_test) = get_cifar10()
        num_labels = 10
        if K.image_data_format() == 'channels_last':
            input_shape = (32, 32, 3)
        else:
            input_shape = (3, 32, 32)
        trainer = train_mobilenet
    if args.dataset == 'cifar100':
        (X_train, Y_train), (X_test, Y_test) = get_cifar100()
        num_labels = 100
        if K.image_data_format() == 'channels_last':
            input_shape = (32, 32, 3)
        else:
            input_shape = (3, 32, 32)
        trainer = train_cifar100

    Y_train = to_categorical(Y_train)
    Y_test = to_categorical(Y_test)

    if args.initial_label_path is not None:
        idx_path = os.path.join(args.initial_label_path, '{exp}_{size}_{data}.pkl'.format(
            exp=args.index, size=args.initial, data=args.dataset))
        with open(idx_path, 'rb') as f:
            labeled_idx = pickle.load(f)
    else:
        labeled_idx = np.random.choice(
            X_train.shape[0], args.initial, replace=False)

    if args.method == 'Random':
        method = Random_Sampling
    elif args.method == 'CoreSet':
        method = CoreSetSampling
    elif args.method == 'Uncertainty':
        method = Uncertainty
    elif args.method == 'Bayesian':
        method = BayesianـUncertainty
    elif args.method == 'UncertaintyEntropy':
        method = Uncertainty_Entropy
    elif args.method == 'BayesianEntropy':
        method = BayesianـUncertaintyـEntropy
    elif args.method == 'EGL':
        method = EGL
    elif args.method == 'Adversarial':
        method = Adversarial
    elif args.method == 'DiscriminativeAE':
        method = DiscriminativeAutoencoderSampling

    if args.second_method is not None:
        if args.second_method == 'Random':
            second_method = Random_Sampling
        elif args.second_method == 'CoreSet':
            second_method = CoreSetSampling
        elif args.second_method == 'Uncertainty':
            second_method = Uncertainty
        elif args.second_method == 'Bayesian':
            second_method = BayesianـUncertainty
        elif args.second_method == 'UncertaintyEntropy':
            second_method = Uncertainty_Entropy
        elif args.second_method == 'BayesianEntropy':
            second_method = BayesianـUncertaintyـEntropy
        elif args.second_method == 'EGL':
            second_method = EGL
        elif args.second_method == 'Adversarial':
            second_method = Adversarial
    else:
        second_method = None

    if second_method is not None:
        query_method = CombinedSampling(
            None, input_shape, num_labels, method, second_method, args.gpu)
    else:
        query_method = method(None, input_shape, num_labels, args.gpu)

    if not os.path.isdir(os.path.join(args.res_folder, 'models')):
        os.mkdir(os.path.join(args.res_folder, 'models'))
    model_folder = os.path.join(args.res_folder, 'models')
    if second_method is None:
        checkpoint_path = os.path.join(model_folder, '{alg}_{datatype}_{init}_{batch_size}_{idx}.hdf5'.format(
            alg=args.method, datatype=args.dataset, batch_size=args.batch, init=args.initial, idx=args.index
        ))
    else:
        checkpoint_path = os.path.join(model_folder, '{alg}_{alg2}_{datatype}_{init}_{batch_size}_{idx}.hdf5'.format(
            alg=args.method, alg2=args.second_method, datatype=args.dataset, batch_size=args.batch, init=args.initial, idx=args.index
        ))

    if not os.path.isdir(os.path.join(args.res_folder, 'results')):
        os.mkdir(os.path.join(args.res_folder, 'results'))
    results_folder = os.path.join(args.res_folder, 'results')
    if second_method is None:
        results_path = os.path.join(results_folder, '{alg}_{datatype}_{init}_{batch_size}_{idx}.pkl'.format(
            alg=args.method, datatype=args.dataset, batch_size=args.batch, init=args.initial, idx=args.index
        ))
    else:
        results_path = os.path.join(results_folder, '{alg}_{alg2}_{datatype}_{init}_{batch_size}_{idx}.pkl'.format(
            alg=args.method, alg2=args.second_method, datatype=args.dataset, batch_size=args.batch, init=args.initial, idx=args.index
        ))

    if second_method is None:
        entropy_path = os.path.join(results_folder, '{alg}_{datatype}_{init}_{batch_size}_{idx}_entropy.pkl'.format(
            alg=args.method, datatype=args.dataset, batch_size=args.batch, init=args.initial, idx=args.index
        ))
    else:
        entropy_path = os.path.join(results_folder, '{alg}_{alg2}_{datatype}_{init}_{batch_size}_{idx}_entropy.pkl'.format(
            alg=args.method, alg2=args.second_method, datatype=args.dataset, batch_size=args.batch, init=args.initial, idx=args.index
        ))

    accuracies = []
    entropies = []
    label_distributions = []
    queries = []
    acc, model = evaluate(
        trainer, X_train[labeled_idx, :], Y_train[labeled_idx], X_test, Y_test, checkpoint_path)
    query_method.update_model(model)
    accuracies.append(acc)
    print("Test Accuracy Is " + str(acc))
    for i in range(args.iterations):

        old_labeled = np.copy(labeled_idx)
        labeled_idx = query_method.query(
            X_train, Y_train, labeled_idx, args.batch)

        new_idx = labeled_idx[np.logical_not(
            np.isin(labeled_idx, old_labeled))]
        new_labels = Y_train[new_idx]
        new_labels /= np.sum(new_labels)
        new_labels = np.sum(new_labels, axis=0)
        entropy = -np.sum(new_labels * np.log(new_labels + 1e-10))
        entropies.append(entropy)
        label_distributions.append(new_labels)
        queries.append(new_idx)

        acc, model = evaluate(
            trainer, X_train[labeled_idx], Y_train[labeled_idx], X_test, Y_test, checkpoint_path)
        query_method.update_model(model)
        accuracies.append(acc)
        print("Test Accuracy Is " + str(acc))

    with open(results_path, 'wb') as f:
        pickle.dump([accuracies, args.initial, args.batch], f)
        print("Saved results to " + results_path)
    with open(entropy_path, 'wb') as f:
        pickle.dump([entropies, label_distributions, queries], f)
        print("Saved entropy statistics to " + entropy_path)
