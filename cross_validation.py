import numpy as np
from data_processing import get_dictionary, featurize_data


def cross_validation(model, X, y, k=10, unigram=True, bigram=False, unigram_cutoff=1,
                     bigram_cutoff=1, featurized=False):
    # Split indexes
    idxs = np.array(range(len(y)))

    folds_idxs = [[] for _ in range(k)]
    for idx in idxs:
        fold = idx % k
        folds_idxs[fold].append(idx)

    # Run test
    accuracies = []
    total_y_pred = []
    total_y_test = []

    for test_fold in range(k):
        print("Running iteration {} out of {} of cross validation".format(test_fold + 1, k))
        test_idxs = folds_idxs[test_fold]
        train_idxs = list(set(np.concatenate(folds_idxs)) - set(test_idxs))
        X_train = X[train_idxs]
        y_train = y[train_idxs]

        X_test = X[test_idxs]
        y_test = y[test_idxs]

        if not featurized:
            token_to_idx = get_dictionary(X_train, unigram_cutoff=unigram_cutoff,
                                          bigram_cutoff=bigram_cutoff, unigram=unigram,
                                          bigram=bigram)
            X_train = featurize_data(X_train, token_to_idx)
            X_test = featurize_data(X_test, token_to_idx)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        total_y_pred = np.concatenate([total_y_pred, y_pred])
        total_y_test = np.concatenate([total_y_test, y_test])
        n_correct = sum(1 for i, _ in enumerate(y_pred) if y_pred[i] == y_test[i])
        accuracy = n_correct * 100 / len(X_test)
        accuracies.append(accuracy)
        # num_of_feat.append(len(token_to_idx))

        print("{0:.2f}% of sentences are correctly classified \n".format(accuracy))

    print("Finished running {}-fold cross validation".format(k))
    # print("Average number of features: {}".format(np.mean(num_of_feat)))
    print("Accuracy is {}(+- {})\n".format(np.mean(accuracies), np.std(accuracies)))

    return total_y_pred, total_y_test
