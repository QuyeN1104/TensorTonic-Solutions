import numpy as np

def adam_step(param, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    param, grad = np.asarray(param), np.asarray(grad)
    m, v = np.asarray(m), np.asarray(v)
    # 1. Update biased first moment estimate
    m = beta1 * m + (1 - beta1) * grad
    # 2. Update biased second raw moment estimate
    v = beta2 * v + (1 - beta2) * (grad**2) # Fixed: grad should be squared

    # 3. Compute bias-corrected first moment estimate
    m_hat = m / (1 - beta1**t)
    # 4. Compute bias-corrected second raw moment estimate
    v_hat = v / (1 - beta2**t)

    # 5. Update parameters
    param_new = param - lr * m_hat / (np.sqrt(v_hat) + eps)
    
    return param_new, m, v