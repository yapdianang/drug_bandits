


class LASSOBandit(object):

    def __init__(self, q, h, lambda1, lambda20):
        self.q = q
        self.h = h
        self.lambda1 = lambda1
        self.lambda20 = lambda20
        self.initialize()
        """ Member variables from initialization:
              a
              b
        """

    def initialize(self):
        pass