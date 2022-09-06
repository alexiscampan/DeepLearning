"""Synapse
    Synapse contenant le poids synaptique et le neurone entrant et le sortant
"""
class Synapse:
    def __init__(self, weight, n_in, n_out):
        self.weight = weight
        self.n_in = n_in
        self.n_out = n_out
    
    def get_weight(self):
        return self.weight
    
    def set_weight(self, val):
        self.weight = val