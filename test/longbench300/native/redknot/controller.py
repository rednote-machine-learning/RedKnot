"""Clone a controller hierarchy while preserving native methods and closures."""

from types import FunctionType

def _cell(value):
    return (lambda: value).__closure__[0]


def _clone_class(original, base, class_map, replacements):
    cloned = type('S2Private' + original.__name__, (base,), {'__module__': __name__, '__doc__': original.__doc__})
    class_map[original] = cloned
    for name, member in original.__dict__.items():
        descriptor = type(member) if isinstance(member, (staticmethod, classmethod)) else None
        fn = member.__func__ if descriptor else member
        if not isinstance(fn, FunctionType):
            if name not in ('__dict__', '__weakref__', '__module__', '__doc__'):
                setattr(cloned, name, member)
            continue
        namespace = dict(fn.__globals__)
        namespace.update({old.__name__: new for old, new in class_map.items()})
        namespace.update(replacements)
        closure = tuple(_cell(class_map.get(c.cell_contents, c.cell_contents))
                        if isinstance(c.cell_contents, type) else c for c in (fn.__closure__ or ())) or None
        copied = FunctionType(fn.__code__, namespace, fn.__name__, fn.__defaults__, closure)
        copied.__kwdefaults__ = fn.__kwdefaults__
        copied.__annotations__ = dict(fn.__annotations__)
        setattr(cloned, name, descriptor(copied) if descriptor else copied)
    return cloned
