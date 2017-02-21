import numpy as np
from sklearn.metrics import roc_auc_score

def truth_table(truth, predictions):

    if truth.shape != predictions.shape:
        print 'Mismatch in truth and prediction size, aborting'
        return

    pred = (predictions + .5).astype(int)

    print 'Truth\n', truth[:10]
    print 'Predictions\n', pred[:10]

    tp = np.sum((truth == pred) * truth)
    fp = np.sum(((1 - truth) == pred) * (1 - truth))
    fn = np.sum((truth == (1 - pred)) * truth)
    tn = np.sum(((1 - truth) == (1 - pred)) * (1 - truth))

    print '\nTruth Table'
    print '\t\tActual Pos\tActual Neg'
    print 'Pred Pos\t', tp, '\t\t', fp
    print 'Pred Neg\t', fn, '\t\t', tn

    return tp, fp, fn, tn


def auc(truth, predictions):

    val = roc_auc_score(truth, predictions)

    print '\nAUC Score:', '{0:.3f}'.format(val)

    return val


def metrics(tp, fp, fn, tn):

    sensitivity = tp / float(tp + fn)
    specificity = tn / float(tn + fp)

    print '\nSensitivity: ', '{0: .3f}'.format(sensitivity)
    print 'Specificity: ', '{0: .3f}'.format(specificity)

    precision = tp / float(tp + fp)
    recall = sensitivity

    f1 = 2 * precision * recall / float(precision + recall)

    print 'F1 Score:', '{0: .3f}'.format(f1)

    return
