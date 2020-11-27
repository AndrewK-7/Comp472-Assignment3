import csv
import math


class NaiveBayesClassifier:
    """
    A custom Naive Bayes Classifier to be executed on the Covid-19 tweet data.
    """

    def __init__(self, vocabulary, model_name):
        """
        Constructor for the NaiveBayesClassifier class.
        :param vocabulary The vocabulary to use.
        :param model_name The model name to use when generating the output files (i.e. NB-BOW-OV or NB-BOW-FV)
        """
        # Set the vocabulary and output file name using the provided model name
        self.vocabulary = vocabulary
        self.output_file = "outputs/trace_" + model_name + ".txt"

        # Define the smoothing factor
        self.smooth = 0.01

        # Define the necessary counts that will be used in calculations
        self.tweet_count = 0
        self.factual_count = 0
        self.not_factual_count = 0

        # Define the PRIOR probabilities
        self.prob_is_factual = 0
        self.prob_is_not_factual = 0

        # Define the dictionaries for the term counts
        self.factual_term_counts = {}
        self.not_factual_term_counts = {}

        # Initialize these dictionaries to contain all of the words in our provided vocabulary
        # Start off the value of each word by the smoothing factor
        for term in vocabulary:
            self.factual_term_counts[term] = self.smooth
            self.not_factual_term_counts[term] = self.smooth
        # end: for-loop

        # Define the dictionaries for the conditional probabilities
        self.conditionals_factual = {}
        self.conditionals_not_factual = {}
    # end: __init__

    def train(self, filename):
        """
        Train the Naive Bayes Classifier on the provided training set
        :param filename: The filename of where to read the training data from.
        :return: void
        """
        # Start reading the file
        with open(filename, encoding="mbcs") as file:
            # Setup a CSV reader to read the data line by line
            reader = csv.reader(file, delimiter='\t')

            # The first line will contain the headers so we don't care about that
            is_first_line = True

            # Start reading each record in the dataset
            for row in reader:
                # Skip the first line
                if is_first_line:
                    is_first_line = False
                    continue
                # end: if

                # Add this tweet to the correct count (factual or not)
                # This property can be found on the 3rd element in the row
                is_factual = False
                if row[2].lower() == 'yes':
                    self.factual_count += 1
                    is_factual = True
                else:
                    self.not_factual_count += 1
                # end: if

                # Extract the tweet from this current record (it will be on the second index of the row)
                tweet_text = row[1].lower()

                # Count each of the terms in this tweet, using the correct "bucket" (factual or not)
                self.count_terms(tweet_text, is_factual)

                # Don't forget to increment the total tweet count
                self.tweet_count += 1
            # end: for-loop
        # end: with-file

        # Now that all of the training data has been counted, let's calculate the probabilities
        # First calculate the PRIOR probabilities
        self.prob_is_factual = self.factual_count / self.tweet_count
        self.prob_is_not_factual = self.not_factual_count / self.tweet_count

        # In order for our conditional calculations to work, we need to add the smoothing to the denominator as well
        # i.e. The total number of tokens in the particular class, plus the smoothed vocabulary size
        smoothed_factual_count = self.factual_count + (len(self.vocabulary) * self.smooth)
        smoothed_not_factual_count = self.not_factual_count + (len(self.vocabulary) * self.smooth)

        # Next let's calculate the conditional probabilities
        # Start with the factual class
        for term in self.factual_term_counts:
            self.conditionals_factual[term] = self.factual_term_counts[term] / smoothed_factual_count
        # end: for-loop

        # Then do the not-factual class
        for term in self.not_factual_term_counts:
            self.conditionals_not_factual[term] = self.not_factual_term_counts[term] / smoothed_not_factual_count
        # end: for-loop
    # end: train

    def test(self, filename):
        """
        Test the classifier on the provided data.
        Note: the test-set does not contain a first row of headers, so we may start from the 1st row.
        :param filename: The filename of the test set to use.
        :return: void
        """
        # Start reading the file
        with open(filename, encoding="mbcs") as file:
            # Setup a CSV reader to read the data line by line
            reader = csv.reader(file, delimiter='\t')

            # Start reading each record in the dataset
            for row in reader:
                # Extract the tweet-ID, which will be on the first index of the row
                tweet_id = row[0]

                # Extract the tweet text, which will be on the second index of the row
                # Don't forget to lower case the text
                tweet_text = row[1].lower()

                # Also extract the TARGET class, which will be the value on the third index of the row
                target_class = row[2].lower()

                # Calculate the probabilities that this tweet belongs to the "factual" and "not factual" class
                factual_score = self.get_probability(tweet_text, True)
                not_factual_score = self.get_probability(tweet_text, False)

                # Choose the max between the two probabilities
                chosen_score = factual_score
                chosen_class = "yes"
                if not_factual_score > factual_score:
                    chosen_score = not_factual_score
                    chosen_class = "no"
                # end: if

                # Write the output into the trace file for this tweet
                self.write_trace(tweet_id, chosen_class, chosen_score, target_class)
            # end: for-loop
        # end: with-file
    # end: test

    def write_trace(self, tweet_id, chosen_class, score, target_class):
        """
        Write the output of this tweet into the trace file for this model.
        :param tweet_id: The tweet ID.
        :param chosen_class: The chosen (predicted) class for this tweet.
        :param score: The calculated score of the tweet's chosen class.
        :param target_class: The target (actual) class of the tweet.
        :return: void
        """
        # Define the line to be written in the trace file
        # Each trace line should look like the following:
        # tweet_id  chosen_class  score(in scientific-notation)  target_class  correct/wrong_label
        line_to_write = tweet_id + "  " + chosen_class + "  " + "{:e}".format(score) + "  " + target_class + "  "

        # Compare the target and chosen classes to see if this was a correct prediction or not
        label = "correct"
        if chosen_class != target_class:
            label = "wrong"
        # end: if

        # Append the label to the line_to_write
        line_to_write += label

        # Create the file connection, append the line, and close the file connection
        f = open(self.output_file, "a")
        f.write(line_to_write + "\n")
        f.close()
    # end: write_trace

    def get_probability(self, text, target_is_factual):
        """
        Get the naive probability for this specified tweet text and target class.
        :param text: The tweet text to use in the calculation.
        :param target_is_factual: True if the target class we are checking for is the "factual" class, False otherwise.
        :return: The naive probability that this tweet belongs to the target class.
        """
        # Start by making the probability be equal to the PRIOR probability for the specified target class
        # We will be using log base 10 for the calculations
        probability = math.log(self.prob_is_factual, 10)
        if not target_is_factual:
            probability = math.log(self.prob_is_not_factual, 10)
        # end: if

        # Now, split the tweet text and loop through every token
        tokens = text.split(' ')
        for token in tokens:
            # We are only interested in the terms that are part of our vocabulary
            if token in self.vocabulary:
                # Add the log of the conditional probability of this word from the appropriate dictionary
                if target_is_factual:
                    probability += math.log(self.conditionals_factual[token], 10)
                else:
                    probability += math.log(self.conditionals_not_factual[token], 10)
                # end: if-else
            # end: if
        # end: for-loop

        # Return the probability for the target class
        return probability
    # end: get_probability

    def count_terms(self, text, is_factual):
        """
        Get the term frequencies from the text, belonging to the provided "bucket" (factual or not)
        :param text: The text to count.
        :param is_factual: True if this text belongs in the "factual" bucket, False if it belongs in the other.
        :return: void
        """
        # First, split the text by the specified tokenizing delimiter (which is a space)
        tokens = text.split(' ')

        # Choose which dictionary to use
        dictionary_to_use = self.factual_term_counts
        if not is_factual:
            dictionary_to_use = self.not_factual_term_counts
        # end: if

        # Loop through each of the tokens
        for token in tokens:
            # Check if the term is in our vocabulary (i.e. is within the dictionary since the dictionaries
            # are already initialized with the terms in the vocabulary)
            # If the term is not in the vocabulary, we simply skip it
            if token in dictionary_to_use:
                # Add one to the count for this term
                dictionary_to_use[token] += 1
            # end: if
        # end: for-loop
    # end: count_terms
# end: class NaiveBayesClassifier
