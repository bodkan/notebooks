import numpy as np
from hmmlearn import hmm
np.random.seed(42)

model = hmm.GaussianHMM(n_components=3, covariance_type="full")

model.startprob_ = np.array([0.6, 0.3, 0.1])
model.transmat_ = np.array([[0.7, 0.2, 0.1],
                            [0.3, 0.5, 0.2],
                            [0.3, 0.3, 0.4]])


model.means_ = np.array([[0.0, 0.0], [3.0, -3.0], [5.0, 10.0]])
model.covars_ = np.tile(np.identity(2), (3, 1, 1))

X, Z = model.sample(20)


p_fair, p_loaded = 0.5, 0.8



def coin_series(p, n):
    return list(np.random.binomial(n=1, p=p, size=n))


flips = coin_series(p_fair, 10) \
    + coin_series(p_loaded, 10) \
    + coin_series(p_fair, 10) \
    + coin_series(p_loaded, 10)



# state 1 = fair coin
# state 2 = loaded coin
model = hmm.MultinomialHMM(n_components=2)

# higher chance of starting with a fair coin
model.startprob_ = np.array([0.9, 0.1])

model.transmat_ = np.array([[0.9, 0.1],
                            [0.9, 0.1]])

model.emissionprob_ = np.array([[p_fair, 1 - p_fair],
                                [p_loaded, 1 - p_loaded]])

x, states = model.sample(100)
