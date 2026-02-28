import numpy as np

def cross_entropy_loss(y_true, y_pred):
    """
    Compute average cross-entropy loss for multi-class classification.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    true_probs = y_pred[np.arange(len(y_pred)), y_true]

    print(true_probs)
    
    losses = -np.log(true_probs)

    avg_loss = np.mean(losses)

    return avg_loss
    