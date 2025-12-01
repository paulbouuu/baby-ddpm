import torch


class EMA:
    """ exponential moving average model """
    def __init__(self, model, beta=0.995):
        self.beta = beta
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}

    def update(self, model):
        with torch.no_grad():
            for k, v in model.state_dict().items():
                self.shadow[k] = self.beta * self.shadow[k] + (1 - self.beta) * v

    def copy_to(self, model):
        model.load_state_dict(self.shadow)