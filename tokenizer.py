import csv


def generate_vocabulary(filename, filter_tokens=False):
    """
    Tokenize all of the words in the provided file, building a dictionary with the unique tokens
    (in lower case) as the keys and their term frequencies as the value.
    :param filename The filename to use when reading in the data.
    :param filter_tokens Will be true if the vocabulary should be filtered as per the assignment specifications.
    :return: The dictionary of the terms and term frequencies.
    """
    # Define the dictionary, empty by default
    vocabulary = {}

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

            # Extract the tweet from this current record (it will be on the second index of the row)
            tweet_text = row[1].lower()

            # Tokenize this tweet and update the dictionary
            tokenize(tweet_text, vocabulary)
        # end: for-loop
    # end: with-file

    # If we must filter out all of the single-occurrence words
    terms_to_remove = []
    if filter_tokens:
        # Find all of the terms that should be removed by checking their frequencies
        for term in vocabulary:
            if vocabulary[term] == 1:
                terms_to_remove.append(term)
            # end: if
        # end: for-loop

        # Remove the terms from the vocabulary
        for term_to_remove in terms_to_remove:
            del vocabulary[term_to_remove]
        # end: for-loop
    # end: if

    # Return the compiled dictionary
    return vocabulary
# end: generate_vocabulary


def tokenize(text, vocabulary):
    """
    Tokenize the text and record the terms key:value pairs in the provided dictionary.
    :param text: The text to tokenize.
    :param vocabulary The dictionary to add the terms to.
    :return: void
    """
    # First, split the text by the specified tokenizing delimiter (which is a space)
    tokens = text.split(' ')

    # Loop through each of the tokens
    for token in tokens:
        # Get the current term frequency for this token
        frequency = 0
        if token in vocabulary:
            frequency = vocabulary[token]
        # end: if

        # Add one to the frequency
        frequency += 1

        # Set the frequency back for that token
        vocabulary[token] = frequency
    # end: for-loop
# end: tokenize
