def evaluate(model_name):
    """
    Evaluate the provided model.
    :param model_name: The model name to evaluate.
    :return: void
    """
    # Define the filename to be used as input (the trace file for the given model)
    trace_file = "outputs/trace_" + model_name + ".txt"

    # Define the filename to be used to output the results
    output_file = "outputs/eval_" + model_name + ".txt"

    # We will need these variables to calculate the desired metrics
    total_records = 0
    total_factual = 0
    total_not_factual = 0
    correct_factual_count = 0
    correct_not_factual_count = 0
    predicted_factual_count = 0
    predicted_not_factual_count = 0

    # Let's open up the trace file and being reading the results line by line
    file = open(trace_file, "r")
    while True:
        # Get the next line from the file and strip any '\n' characters found
        line = file.readline().strip("\n")

        # If the line is empty, we are done reading
        if not line:
            break
        # end: if

        # Split the line by the appropriate delimiter (two spaces)
        line_values = line.split("  ")

        # To Get the predicted class, use the 2nd index
        predicted_class = line_values[1]

        # To get the actual class, use the 4th index
        target_class = line_values[3]

        # To get whether the class was correctly predicted or not, use the 5th index
        is_correct = line_values[4] == "correct"

        # Add one to the appropriate count (only if the prediction was correct)
        if is_correct:
            if target_class == "yes":
                correct_factual_count += 1
            else:
                correct_not_factual_count += 1
            # end: if
        # end: if

        # Also add one to the totals based on the predicted class
        if predicted_class == "yes":
            predicted_factual_count += 1
        else:
            predicted_not_factual_count += 1
        # end: if

        # Also add one to the totals based on the actual class
        if target_class == "yes":
            total_factual += 1
        else:
            total_not_factual += 1
        # end: if

        # Don't forget to add to the total count
        total_records += 1
    # end: while

    # Don't forget to close our file connection
    file.close()

    # To calculate the accuracy, we simply need to divide the total number of correct predictions by the total
    accuracy = (correct_factual_count + correct_not_factual_count) / total_records

    # To calculate the precision, we must divide the true positives by the sum of the true positives and false positives
    precision_factual = correct_factual_count / predicted_factual_count
    precision_not_factual = correct_not_factual_count / predicted_not_factual_count

    # To calculate the recall, we must divide the true positives by the sum of the true positives and false negatives
    recall_factual = correct_factual_count / total_factual
    recall_not_factual = correct_not_factual_count / total_not_factual

    # To calculate the F1-measure, we must use the appropriate formula (see slides related to Lecture 2.5)
    # We need to define a "beta" value, used to represent the relative importance of recall to precision
    # When "beta" = 1, precision and recall have the same importance
    # When "beta" > 1, recall is given more weight
    # When "beta" < 1, precision is given more weight
    beta = 1
    f1_factual = ((beta**2 + 1) * precision_factual * recall_factual) / \
                 ((beta**2 * precision_factual) + recall_factual)
    f1_not_factual = ((beta**2 + 1) * precision_not_factual * recall_not_factual) / \
                     ((beta**2 * precision_not_factual) + recall_not_factual)

    # Now that we have all of our metrics calculated, we can define our output line
    # It should be in the following format:
    # "accuracy
    #  factual_precision  not_factual_precision
    #  factual_recall  not_factual_recall
    #  factual_f1  not_factual_f1"
    line_to_write = "{:.4}".format(accuracy) + "\r"
    line_to_write += "{:.4}".format(precision_factual) + "  " + "{:.4}".format(precision_not_factual) + "\r"
    line_to_write += "{:.4}".format(recall_factual) + "  " + "{:.4}".format(recall_not_factual) + "\r"
    line_to_write += "{:.4}".format(f1_factual) + "  " + "{:.4}".format(f1_not_factual) + "\r"

    # Write the line to the output file
    f = open(output_file, "a")
    f.write(line_to_write)
    f.close()
# end: evaluate
