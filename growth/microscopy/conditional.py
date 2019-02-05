import numpy as np


class ConditionalSampler:
    """
    Class for sampling y conditioned on an existing sample x.
    """
    def __init__(self, x):
        self.x = x
        self.mu_x = x.mean()
        self.sigma_x = x.std()

    def sample(self, mu_y, sigma_y, rho=0.0):
        """ Sample y conditioned upon x. """
        mu = mu_y + (rho*sigma_y/self.sigma_x)*(self.x - self.mu_x)
        sigma = np.sqrt((1-rho**2)*(sigma_y**2))
        return mu + sigma * np.random.normal(size=self.x.size)

class ConditionalLogSampler(ConditionalSampler):

    def __init__(self, x):
        super().__init__(np.log(x))

    def sample(self, mu_y, sigma_y, rho=0.0):
        """ Sample y conditioned upon x. """
        return np.exp(super().sample(np.log(mu_y), sigma_y, rho=rho))
