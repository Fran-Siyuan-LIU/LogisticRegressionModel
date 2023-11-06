# Logistic Regression Model

## Pricipals and Details:

Reference: https://en.wikipedia.org/wiki/Logistic_regression

## Parameters:

    eta: learning rate, (0.0, 1.0)

    n_iter: number of iteration, it is recommended be larger than the converge rate to achieve desired model

    random_state: an integer to make the result reproductable

    tol: the tolerance rate, if all weights changed less than the tolerance rate, the learning process will stop

## Attributes:

    w__: the weight of each dimension of the observations, w__[0] denotes the intercept while others denote slope

    cost__: the cost of each iteration based on the pre-decided cost function. The smaller the cost, the better the prediction

## Functions:

### Helper Functions:

#### phi(X):
    Here, X is a 1-d array of net input.
    In logistic regression, phi(x) is the sigmond function.
    phi(X)[i] denotes P(y_i = 1).

    This function would return an array in the same shape of X.

#### net_input(X):
    X here is a n*m matrix, in which each line denotes an observation while each column denotes a feature.

    The w__[1: ] is a 1-d array (1*m) denotes weights of each feature.

    The dot multiplication of these two matrixs creates a new 1-d array (1*n), the result of (this array + w__[0] (the intercept)) showing the net input of each observation.

    Net input in the logistic regression model denotes the logit of P(y = 1).

    Key assumption of this model is: logit(p) = w_0 + w_1 * x_1 + w_2 * x_2 + ... + w_m * x_m.
    It also could be explained as: logit(p) has linear relationships with each feature of x.

$$ logit(p) = \log(p/(1 - p)) $$

### Fit function:

#### Parameters:
    X: 2-d array, normally n_observations*m_features
    y: the target, normally n_observations*1

#### Percedures:
    Firstly, generating a series of small weight w__(1*(m + 1)) array, where w__[0] denotes intercept while w__[1: ] denote weights for each features

    Then, for each iteration (epoch), adjust weights to achieve better predictions.

#### Cost Function and Gradient Descent Optimization

The model predicts $P(y = 1)$ as a float falls in $(0, 1)$, which is $\phi(z)$. Therefore, the probability of predicting ith observation should be:

$$
P(i) =
\begin{cases}
\phi(z)& \text{, if } y_i = 1\\
1 - \phi(z)& \text{, if } y_i = 0
\end{cases}
$$

This $P(i)$ function could also be expressed as:

$$ P(i) = \phi(z)^{y} * (1 - \phi(z)) ^ {(1 - y)} $$

As a result, the total probability of predicting each observation correctly should be:

$$ 
P(correct) = \Pi_{i = 1} ^ n P(i)
$$
This is our target to maximize.

Take the log of each side, we get
$
\log(P(correct)) = \Sigma_{i = 1} ^ n (y * \log (\phi(z)) + (1 - y) * \log(1 - \phi(z)))
$
to maximize.

Normally, the smaller the cost, the more accurate the prediction is.
Therefore, we have our cost function as:
$$
cost = -\log(P(correct))
$$
This cost function is the one we want to minimize (optimize).

We can approve that the cost function is a unimodal function with one minimun.

We assume $L = \log(P(correct))$, then we have:
$$
\frac{\partial L}{\partial w_j} = \Sigma_{i = 1} ^ n (\frac{y_i}{\phi(z)} - \frac{1 - y}{1 - \phi(z)}) * \frac{\partial \phi(z)}{\partial w_j}
$$
As we have:
$$
\begin{aligned}
\frac{\partial \phi(z)}{\partial w_j} = & \frac{e^{-z}}{(1 + e^{-z})^2}x_j\\
                    = & \frac{1}{1 + e^{-z}}*(1 - \frac{1}{1 + e^{-z}})x_j\\
                    = & \phi(z)*(1 - \phi(z))*x_j
\end{aligned}
$$
We can easily derived to have:
$$
\begin{aligned}
\frac {\partial L}{\partial w_j} =& \Sigma_{i = 1}^n (\frac{y_i}{\phi(z)} - \frac{1 - y}{1 - \phi(z)}) * \phi(z)*(1 - \phi(z))*x_j\\
                                 =& \Sigma_{i = 1}^n (y_i * (1 - \phi(z)) - (1 - y_i) * \phi(z))x_j
\end{aligned}
$$

In gradient descent, we add a small fraction of the partial derivative of $w_j$ to $w_j$ to make it closer to the $w_j*$ where $L$ achieve local maximum. This fraction is what we call learning rate. 

After a few epochs, we may stop the learning process if the optimization of each epoch is smaller than our tolerance rate, which denotes that we are near the local minimum. If we reach this stage, we call our model as "converged". That's where we use learning rate. A small learning rate may make the learning process very slow and the converge will come very late. A large learning rate exposes the model to the risk of overshooting the minimum(maximum).

### Fit function:

As we have $\phi(z) = \frac{1}{1 + e^{-z}} $, we know that if net_input $z$ is larger than 0, we achieve $\phi(z) > 0.5$.

We predict an observation as 1 if we get $\phi(z) > 0.5$.