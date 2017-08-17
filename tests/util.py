from aboleth.baselayers import Layer, MultiLayer


class StringLayer(Layer):

    def __init__(self, name='f', kl=0.0):
        self.name = name
        self.kl = kl

    def _build(self, X):
        result = "{}({})".format(self.name, X)
        return result, self.kl


class StringMultiLayer(MultiLayer):

    def __init__(self, name='f', args=None, kl=0.0):
        args = ['x'] if not args else args
        self.name = name
        self.args = args
        self.kl = kl

    def _build(self, **kwargs):
        l = [kwargs[i] for i in sorted(kwargs)]
        call_sig = ",".join(l)
        result = "{}({})".format(self.name, call_sig)
        return result, self.kl
