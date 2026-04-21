class SimpleModel:
    def __init__(self, weight):
        self.weight = weight
    
    def predict(self, x):
        return self.weight * x
