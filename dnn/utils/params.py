import copy
import importlib.util as imp_util

class ParamDict(dict):
    """ Parameter manager implemented with attributes dictionary """

    def __init__(self, **kwargs):
        super(ParamDict, self).__init__(**kwargs)
        self.__dict__ = self # allow dictionary access through attributes

    def __call__(self, **kwargs):
        params = ParamDict(**self)
        params.update(**kwargs)
        return params

    def __deepcopy__(self, memo):
        return self.__class__(**copy.deepcopy(dict(self), memo=memo))

    @classmethod
    def from_file(cls, file_path):
        spec = imp_util.spec_from_file_location("params", file_path)
        params_mod = imp_util.module_from_spec(spec)
        spec.loader.exec_module(params_mod)
        return copy.deepcopy(params_mod.PARAMS)
