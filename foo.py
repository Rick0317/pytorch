import torch
from torch._ops import PyOperator
from torch._C._functorch import TransformType
from functorch import vmap
import functools

mysum = PyOperator("mysum")

@mysum.py_functorch_impl(TransformType.Vmap)
def mysum_batch_rule(interpreter, x, dim):
    print("invoked")

    if not torch._C._functorch.is_batchedtensor(x): 
        with interpreter.lower():
            return mysum(x, dim)

    bdim = torch._C._functorch.maybe_get_bdim(x)
    value = torch._C._functorch.get_unwrapped(x)

    with interpreter.lower():
        value = value.movedim(bdim, 0)
        return mysum(value, dim + 1)

@mysum.py_impl(torch._C.DispatchKey.AutogradCPU)
def mysum_autograd(x, dim):
    return torch.sum(x, dim)


torch.manual_seed(0)
x = torch.randn(2, 3)
y = mysum(x, 1)
assert torch.allclose(y, x.sum(1))

def test(f, f_p, in_dims, args):
    expected = vmap(f, in_dims)(*args)
    result = vmap(f_p, in_dims)(*args)
    assert torch.allclose(result, expected)

# single vmap
test(torch.sum, mysum, (0, None), (x, 0))

# nested vmap
x = torch.randn(2, 3, 4)
test(vmap(functools.partial(torch.sum, dim=0)),
     vmap(functools.partial(mysum, dim=0)),
     (0,),
     (x,))
