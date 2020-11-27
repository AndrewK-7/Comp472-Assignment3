import os
import glob
import datetime

from tokenizer import generate_vocabulary
from naive_bayes_classifier import NaiveBayesClassifier
from evaluator import evaluate


def clear_old_outputs():
    """
    Simply clear any old output files left by a previous run.
    :return: void
    """
    # Use a wildcard to find all of the .txt files within the outputs directory
    files_to_remove = glob.glob("outputs/*.txt", recursive=True)

    # For every file found, remove it
    for f in files_to_remove:
        try:
            os.remove(f)
        except OSError as e:
            print("Error: %s : %s" % (f, e.strerror))
    # end: for-loop
# end: clear_old_outputs


if __name__ == '__main__':
    """
    Starting point to the application.
    """
    # Print a message indicating that the application has started
    print("\nStarting Covid-19 Fact Checking Application")
    print("===========================================\n")

    # Before we get started, let's clean up any old outputs files left by a previous run
    clear_old_outputs()

    # Let's define the two filenames that will contain our training and testing data sets
    training_set = "datasets/covid_training.tsv"
    testing_set = "datasets/covid_test_public.tsv"

    # Let's tokenize the training set to get our vocabulary
    # We want to have two versions of our vocabulary, one containing ALL WORDS in the training set
    original_vocabulary = generate_vocabulary(training_set, False)

    # And the other version will be filtered, removing all of the words that only appear once
    filtered_vocabulary = generate_vocabulary(training_set, True)

    # Print our the vocabularies in case we want to take a peek
    print("Here are the two vocabularies used:")
    print("Original Vocabulary: ", end='')
    print(original_vocabulary)
    print("Filtered Vocabulary: ", end='')
    print(filtered_vocabulary)
    print("")

    # Create two NaiveBayesClassifiers, one for each vocabulary
    nb1 = NaiveBayesClassifier(original_vocabulary, "NB-BOW-OV")
    nb2 = NaiveBayesClassifier(filtered_vocabulary, "NB-BOW-FV")

    # Start the first classification
    print("Starting the classification of the test set using model: NB-BOW-OV... ", end='')
    start_time = datetime.datetime.now()
    nb1.train(training_set)
    nb1.test(testing_set)
    execution_time = datetime.datetime.now() - start_time
    print("Done! (took %.4f ms)" % (execution_time.microseconds / 1000))

    # Start the second classification
    print("Starting the classification of the test set using model: NB-BOW-FV... ", end='')
    start_time = datetime.datetime.now()
    nb2.train(training_set)
    nb2.test(testing_set)
    execution_time = datetime.datetime.now() - start_time
    print("Done! (took %.4f ms)" % (execution_time.microseconds / 1000))

    # After the classifiers are done, we can evaluate how well they did
    evaluate("NB-BOW-OV")
    evaluate("NB-BOW-FV")
# end: __main__
