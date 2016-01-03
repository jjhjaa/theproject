from collections import defaultdict
import codecs
import argparse

def evaluate(true_path, prediction_path):
    true_handle = codecs.open(true_path, 'r')
    prediction_handle = codecs.open(prediction_path, 'r')

    true_labels = [line.split('\t')[2] for line in true_handle.readlines()]
    predicted_labels = [line.split('\t')[2] for line in prediction_handle.readlines()]
    # print(true_labels)
    # print(predicted_labels)

    print("F-score: {}".format(_macro_f_score(true_labels, predicted_labels)))


def _macro_f_score(true_labels, predicted_labels):
    """
    Mimics the official SemEval evaluation,
    which calculates the macro-averaged f-score of positive and negative class.
    Neutral class is ignored.
    The input should be lists or other iterable types (e.g. numpy arrays).
    """

    counts = defaultdict(lambda:defaultdict(int))

    for i in range(len(true_labels)):
        if predicted_labels[i] == true_labels[i] and predicted_labels[i] in ['positive', 'negative']:
            counts[predicted_labels[i]]['tp'] += 1
        elif predicted_labels[i] != true_labels[i] and predicted_labels[i] in ['positive', 'negative']:
            counts[predicted_labels[i]]['fp'] += 1

    counts['positive']['all_true'] = list(true_labels).count('positive')
    counts['negative']['all_true'] = list(true_labels).count('negative')

    try:
        precision_positive = float(counts['positive']['tp']) / (counts['positive']['tp'] + counts['positive']['fp'])
        recall_positive = float(counts['positive']['tp']) / counts['positive']['all_true']
        f_score_positive = 2 * precision_positive * recall_positive / (precision_positive + recall_positive)
    except:
        f_score_positive = 0.0

    try:
        precision_negative = float(counts['negative']['tp']) / (counts['negative']['tp'] + counts['negative']['fp'])
        recall_negative = float(counts['negative']['tp']) / counts['negative']['all_true']
        f_score_negative = 2 * precision_negative * recall_negative / (precision_negative + recall_negative)
    except:
        f_score_negative = 0.0


    return (f_score_positive + f_score_negative) / 2.0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Twitter sentiment evaluator script.')
    parser.add_argument('--true', help='Path to true labels', required=True)
    parser.add_argument('--predictions', help='Path to predicted labels', required=True)

    args = vars(parser.parse_args())

    evaluate(args['true'], args['predictions'])