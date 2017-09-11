class Eval1(object):
    def __init__(self, data):
        self.data = data
        self.icc = None
        self.kappa = None

    def calculate_icc(self):
        if self.icc:
            return
    
    def calculate_kappa(self):
        if self.kappa:
            return
