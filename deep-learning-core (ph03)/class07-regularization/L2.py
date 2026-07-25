def l2_regularization(weights, lambda_reg):
    penalty = 0.0
    for w in weights:
        penalty += w * w
    return lambda_reg * 0.5 * penalty


def l2_gradient(weights, lambda_reg):
    return [lambda_reg * w for w in weights]
