import copy

SIMPLE_TYPES = {bool, int, float, str}

def _list_check(l):
    for item in l:
        if type(item) in SIMPLE_TYPES:
            pass
        elif isinstance(item, list):
            _list_check(item)
        else:
            raise ValueError('Lists in configs can contain only other lists or simple types')


# Placeholder value to indicate that a field must be specified in configuration, and it must have the given type
class Require:
    def __init__(self, dtype):
        self.dtype = dtype

    def __repr__(self):
        return f'Require({self.dtype})'


# Placeholder value to allow for optional arguments. Will be replaced by None if not specified
class Optional:
    def __init__(self, dtype):
        self.dtype = dtype

    def __repr__(self):
        return f'Optional({self.dtype})'


class TaggedUnion:
    def __init__(self, **config_classes):
        self.config_classes = config_classes

    def parse(self, d):
        assert isinstance(d, dict)
        tag = d.pop('_tag_')
        cfg = self.config_classes[tag]()
        cfg.update(d)
        return cfg


class BaseConfig:
    def vars(self):
        vars = {}
        for key in dir(self):
            val = getattr(self, key)
            if callable(val) or key.startswith('_'):
                continue
            vars[key] = val
        return vars

    def vars_recursive(self):
        vars = self.vars()
        for key in vars:
            if isinstance(vars[key], BaseConfig):
                vars[key] = vars[key].vars_recursive()
        return vars

    def __init__(self, **kwargs):
        v = self.vars()
        v.update(kwargs)
        for key, val in v.items():
            setattr(self, key, val)

    def typesafe_set(self, key, value):
        assert type(value) in SIMPLE_TYPES
        existing_val = getattr(self, key)
        if isinstance(existing_val, Optional):
            assert isinstance(value, existing_val.dtype), \
                   f'Got wrong type for key {key}: expected {existing_val.dtype} but got {type(value)}'
        else:
            if isinstance(existing_val, Require):
                expected_type = existing_val.dtype
            else:
                assert type(existing_val) in SIMPLE_TYPES
                expected_type = type(existing_val)
            assert isinstance(value, expected_type), \
                   f'Got wrong type for key {key}: expected {expected_type} but got {type(value)}'
        setattr(self, key, value)

    def update(self, d):
        for key, val in d.items():
            assert hasattr(self, key), f'Cannot set non-existent key {key} in {self}'
            if type(val) in SIMPLE_TYPES:
                self.typesafe_set(key, val)
            elif isinstance(val, dict):
                existing_val = getattr(self, key)
                if isinstance(existing_val, BaseConfig):
                    existing_val.update(val)
                elif isinstance(existing_val, TaggedUnion):
                    setattr(self, key, existing_val.parse(val))
                else:
                    raise ValueError(f'Given a dict for key {key}, which is not a BaseConfig or TaggedUnion instance, but rather {existing_val}')
            elif isinstance(val, list):
                _list_check(val)
                setattr(self, key, copy.deepcopy(val))
            else:
                raise ValueError(f'Object of unexpected type: {val} ({type(val)})')

    def _nested_set_recurse(self, path, value):
        path0 = path[0]
        if len(path) == 1:
            if hasattr(self, path0):
                self.typesafe_set(path0, value)
                return True
            else:
                return False
        else:
            subconfig = getattr(self, path0)
            assert isinstance(subconfig, BaseConfig)
            return subconfig._nested_set_recurse(path[1:], value)

    def nested_set(self, path, value):
        assert isinstance(path, list)
        if path == ['seed']:
            assert type(value) is int
            self.seed = value
        if path == ['debug']:
            assert type(value) is bool
            self.debug = value

        if not self._nested_set_recurse(path, value):
            key = '.'.join(path)
            raise ValueError(f'Cannot override non-existent key {key}')

    def verify(self):
        for key, val in self.vars().items():
            if isinstance(val, list):
                _list_check(val)
            elif isinstance(val, BaseConfig):
                val.verify()
            elif isinstance(val, Require):
                raise ValueError(f'Required key {key} has not been set')
            elif isinstance(val, Optional):
                # Optional was never set, so it will be None
                setattr(self, key, None)
            elif isinstance(val, TaggedUnion):
                raise ValueError(f'TaggedUnion for key {key} has not been set')
            else:
                assert type(val) in SIMPLE_TYPES, f'Invalid value for key {key}: {val}'

    def __str__(self):
        args = ', '.join(f'{key}={val}' for key, val in vars(self).items())
        return f'Config({args})'


class Configurable:
    """All subclasses must define a nested class called Config, which specifies the configurable fields."""
    def __init__(self, config):
        assert type(config) is self.__class__.Config
        self.config = copy.deepcopy(config)
        for key, val in vars(self.config).items():
            setattr(self, key, val)