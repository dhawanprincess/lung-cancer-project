def combine_results(image_prob, data_prob):

    # Weighted fusion of both predictions
    final_score = (0.6 * image_prob) + (0.4 * data_prob)

    if final_score > 0.5:
        return "High Risk of Lung Cancer"
    else:
        return "Low Risk of Lung Cancer"
