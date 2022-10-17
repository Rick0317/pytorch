import contextlib
import torch
from torch._C._functorch import (
    TransformType,
    CInterpreter,
    CGradInterpreterPtr,
    CVmapInterpreterPtr,
    WithoutTop,
)


class FunctorchInterpreter:
    @contextlib.contextmanager
    def lower(self):
        # TODO: this is sketch
        try:
            guard = WithoutTop()
            yield
        finally:
            del guard


class VmapInterpreter(FunctorchInterpreter):
    def __init__(self, cdata: CVmapInterpreterPtr):
        self._cdata = cdata

    def py_process(self, op, args, kwargs):
        kernel = op.functorch_table[TransformType.Vmap]
        return kernel(self, *args, **kwargs)


class GradInterpreter(FunctorchInterpreter):
    def __init__(self, cdata: CGradInterpreterPtr):
        self.cdata = cdata

    def py_process(self, op, args, kwargs):
        kernel = op.functorch_table[TransformType.Grad]
        return kernel(self, *args, **kwargs)


def coerce_cinterpreter(cinterpreter: CInterpreter) -> FunctorchInterpreter:
    key = cinterpreter.key()
    if key == TransformType.Grad:
        return GradInterpreter(CGradInterpreterPtr(cinterpreter))
    if key == TransformType.Vmap:
        return VmapInterpreter(CVmapInterpreterPtr(cinterpreter))
    raise RuntimeError(f"Don't know how to handle {key}")


def retrieve_current_functorch_interpreter():
    interpreter = torch._C._functorch.peek_interpreter_stack()
    assert interpreter is not None
    return coerce_cinterpreter(interpreter)


def dispatch_functorch(op, args, kwargs):
    interpreter = retrieve_current_functorch_interpreter()
    return interpreter.py_process(op, args, kwargs)
