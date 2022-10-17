import contextlib
import dataclasses
import functools
from pathlib import Path
from typing import Dict, List

import sympy

import torch
from torch._prims_common import is_float_dtype

from .. import codecache, config, ir, metrics
from ..utils import sympy_product, sympy_subs
from ..virtualized import ops, V
from .common import (
    BracesBuffer,
    DeferredIndentedBuffer,
    ExprPrinter,
    IndentedBuffer,
    Kernel,
    KernelArgs,
    OpOverrides,
)

DTYPE_TO_CPP = {
    torch.float32: "float",
    torch.float64: "double",
    torch.float16: "half",
    torch.int64: "long",
    torch.int32: "int",
    torch.int16: "short",
    torch.int8: "signed char",
    torch.uint8: "unsigned char",
    torch.bool: "bool",
    torch.bfloat16: "bfloat16",
}
INDEX_TYPE = "long"

RTYPE_TO_CPP = {
    "sum": "+",
    "min": "min",
    "max": "max",
    "argmin": "argmin",
    "argmax": "argmax",
    "any": "||",
}


def reduction_init(reduction_type, dtype):
    if reduction_type in ("sum", "any"):
        return 0
    if reduction_type in {"max", "argmax"}:
        return (
            f"-std::numeric_limits<{DTYPE_TO_CPP[dtype]}>::infinity()"
            if is_float_dtype(dtype)
            else f"std::numeric_limits<{DTYPE_TO_CPP[dtype]}>::min()"
        )
    if reduction_type in {"min", "argmin"}:
        return (
            f"std::numeric_limits<{DTYPE_TO_CPP[dtype]}>::infinity()"
            if is_float_dtype(dtype)
            else f"std::numeric_limits<{DTYPE_TO_CPP[dtype]}>::max()"
        )
    raise AssertionError(reduction_type)


def reduction_combine(reduction_type, var, next_value):
    if reduction_type == "sum":
        return f"{var} += {next_value}"
    if reduction_type == "any":
        return f"{var} = {var} || {next_value}"
    return f"{var} = std::{reduction_type}({var}, {next_value})"


index_value_name_counter = 1


def argmax_argmin_prefix(reduction_type, src_dtype, tmpvar):
    global index_value_name_counter
    struct_name = f"IndexValue_{index_value_name_counter}"
    index_value_name_counter += 1

    # A small annoyance, due to it being a little cumbersome to just throw {} into strings
    prefix = [
        f"struct {struct_name} {{size_t index; {DTYPE_TO_CPP[src_dtype]} value;}};",
        f"{struct_name} {tmpvar}{{0, {reduction_init(reduction_type, src_dtype)}}};",
    ]
    if reduction_type == "argmax":
        prefix.extend(
            [
                f"#pragma omp declare reduction(argmax : struct {struct_name} :\\",
                "    omp_out.value = omp_in.value < omp_out.value ? omp_out.value : omp_in.value,\\",
                "    omp_out.index = omp_in.value < omp_out.value ? omp_out.index : omp_in.index)\\",
                f"\tinitializer(omp_priv = {{0, {reduction_init(reduction_type, src_dtype)}}})",
            ]
        )
    elif reduction_type == "argmin":
        prefix.extend(
            [
                f"#pragma omp declare reduction(argmin : struct {struct_name} :\\",
                "    omp_out.value = omp_in.value > omp_out.value ? omp_out.value : omp_in.value,\\",
                "    omp_out.index = omp_in.value > omp_out.value ? omp_out.index : omp_in.index)\\",
                f"\tinitializer(omp_priv = {{0, {reduction_init(reduction_type, src_dtype)}}})",
            ]
        )
    return prefix


def float16_reduction_prefix(rtype):
    # TODO: This user-defined reduction uses float16 accumulation for sum. To reduce numerical
    # errors, float32 accumulation should be used instead.
    assert rtype in (
        "sum",
        "any",
    ), f"float16 user-defined reduction only supports 'sum' and 'any' but got {rtype}"
    prefix = [
        f"#pragma omp declare reduction({RTYPE_TO_CPP[rtype]}:{DTYPE_TO_CPP[torch.float16]}:"
        + f"omp_out = omp_out {RTYPE_TO_CPP[rtype]} omp_in)"
    ]
    return prefix


@functools.lru_cache()
def cpp_prefix():
    path = Path(__file__).parent / "cpp_prefix.h"
    with path.open() as f:
        _, filename = codecache.write(
            f.read(),
            "h",
        )
    return f'#include "{filename}"'


class CppPrinter(ExprPrinter):
    def _print_ModularIndexing(self, expr):
        x, div, mod = expr.args
        x = self.paren(self.doprint(x))
        div = self.paren(self.doprint(div))
        mod = self.paren(self.doprint(mod))
        if div != "1":
            x = f"({x} / {div})"
        return f"{x} % {mod}"

    def _print_IndexingDiv(self, expr):
        x, div = expr.args
        x = self.paren(self.doprint(x))
        div = self.paren(self.doprint(div))
        return f"({x} / {div})"


cexpr = CppPrinter().doprint


class CppVecOverrides(OpOverrides):
    """Map element-wise ops to aten vectorization C++"""

    @staticmethod
    def add(a, b):
        return f"{a} + {b}"

    @staticmethod
    def sub(a, b):
        return f"{a} - {b}"

    @staticmethod
    def mul(a, b):
        return f"{a} * {b}"

    @staticmethod
    def div(a, b):
        return f"{a} / {b}"

    @staticmethod
    def abs(x):
        return f"{x}.abs()"

    @staticmethod
    def sin(x):
        return f"{x}.sin()"

    @staticmethod
    def cos(x):
        return f"{x}.cos()"

    @staticmethod
    def exp(x):
        return f"{x}.exp()"

    @staticmethod
    def sqrt(x):
        return f"{x}.sqrt()"

    @staticmethod
    def rsqrt(x):
        return f"{x}.rsqrt()"

    @staticmethod
    def pow(a, b):
        return f"{a}.pow({b})"

    @staticmethod
    def log(x):
        return f"{x}.log()"

    @staticmethod
    def round(x):
        return f"{x}.round()"

    @staticmethod
    def floor(x):
        return f"{x}.floor()"

    @staticmethod
    def ceil(x):
        return f"{x}.ceil()"

    @staticmethod
    def trunc(x):
        return f"{x}.trunc()"

    @staticmethod
    def fmod(a, b):
        return f"{a}.fmod({b})"

    @staticmethod
    def lgamma(x):
        return f"{x}.lgamma()"

    @staticmethod
    def logical_and(a, b):
        return f"{a} && {b}"

    @staticmethod
    def logical_or(a, b):
        return f"{a} || {b}"


class CppOverrides(OpOverrides):
    """Map element-wise ops to C++"""

    @staticmethod
    def to_dtype(x, dtype):
        assert dtype in DTYPE_TO_CPP, f"{dtype} missing from {__name__}.DTYPE_TO_CPP"
        return f"static_cast<{DTYPE_TO_CPP[dtype]}>({x})"

    @staticmethod
    def abs(x):
        return f"std::abs({x})"

    @staticmethod
    def sin(x):
        return f"std::sin({x})"

    @staticmethod
    def cos(x):
        return f"std::cos({x})"

    @staticmethod
    def exp(x):
        # return f"Sleef_expf_u10({x})"
        return f"std::exp({x})"

    @staticmethod
    def sqrt(x):
        return f"std::sqrt({x})"

    @staticmethod
    def rsqrt(x):
        return f"1 / std::sqrt({x})"

    @staticmethod
    def signbit(x):
        return f"std::signbit({x})"

    @staticmethod
    def pow(a, b):
        return f"std::pow({a}, {b})"

    @staticmethod
    def log(x):
        return f"std::log({x})"

    @staticmethod
    def round(x):
        return f"std::nearbyint({x})"

    @staticmethod
    def floor(x):
        return f"std::floor({x})"

    @staticmethod
    def floordiv(a, b):
        # a and b are integer type
        quot = f"{a} / {b}"
        rem = f"{a} % {b}"
        return f"(({a} < 0) != ({b} < 0) ? ({rem} != 0 ? {quot} - 1 : {quot}) : {quot})"

    @staticmethod
    def ceil(x):
        return f"std::ceil({x})"

    @staticmethod
    def trunc(x):
        return f"std::trunc({x})"

    @staticmethod
    def truncdiv(a, b):
        # a and b are integer type
        return f"{a} / {b}"

    @staticmethod
    def fmod(a, b):
        return f"std::fmod({a}, {b})"

    @staticmethod
    def isinf(x):
        return f"std::isinf({x})"

    @staticmethod
    def isnan(x):
        return f"std::isnan({x})"

    @staticmethod
    def lgamma(x):
        return f"std::lgamma({x})"

    @staticmethod
    def relu(x):
        return f"{x} * ({x}>0)"

    @staticmethod
    def minimum(a, b):
        return f"std::min({a}, {b})"

    @staticmethod
    def maximum(a, b):
        return f"std::max({a}, {b})"

    @staticmethod
    def where(a, b, c):
        return f"{a} ? {b} : {c}"

    @staticmethod
    def mod(a, b):
        return f"mod({a}, {b})"

    @staticmethod
    def constant(val, dtype):
        if val == float("inf"):
            return f"std::numeric_limits<{DTYPE_TO_CPP[dtype]}>::infinity()"
        elif val == float("-inf"):
            return f"-std::numeric_limits<{DTYPE_TO_CPP[dtype]}>::infinity()"
        elif val is True or val is False:
            return ops.to_dtype(str(val).lower(), dtype)
        return ops.to_dtype(repr(val), dtype)

    @staticmethod
    def index_expr(expr, dtype):
        return ops.to_dtype(cexpr(V.kernel.rename_indexing(expr)), dtype)

    @staticmethod
    def masked(mask, body, other):
        code = BracesBuffer()
        var = V.kernel.cse.newvar()
        if other == float("-inf"):
            code.writeline(f"float {var} = -std::numeric_limits<float>::infinity();")
        elif other == float("inf"):
            code.writeline(f"float {var} = std::numeric_limits<float>::infinity();")
        else:
            code.writeline(f"auto {var} = {other!r};")
        code.writeline(f"if({mask})")
        with V.kernel.swap_buffers(code), code.indent():
            result = body()
            code.writeline(f"{var} = {result};")
        V.kernel.compute.splice(code)
        return var

    @staticmethod
    def logical_and(a, b):
        return f"{a} && {b}"

    @staticmethod
    def logical_or(a, b):
        return f"{a} || {b}"

    @staticmethod
    def rand(seed: sympy.Expr, offset: sympy.Expr, dtype):
        return f"static_cast<{DTYPE_TO_CPP[dtype]}>(normalized_rand_cpu({seed}, {offset}));"

    @staticmethod
    def randn(seed: sympy.Expr, offset: sympy.Expr, dtype):
        return f"static_cast<{DTYPE_TO_CPP[dtype]}>(randn_cpu({seed}, {offset}));"


class CppKernel(Kernel):
    overrides = CppOverrides
    sexpr = cexpr
    newvar_prefix = "auto "
    suffix = ";"

    def __init__(self, args, num_threads):
        super(CppKernel, self).__init__(args)
        self.call_ranges = None
        self.ranges = None
        self.itervars = None
        self.reduction_depth = None
        self.reduction_prefix = IndentedBuffer()
        self.reduction_suffix = DeferredIndentedBuffer()
        self.reduction_vars = {}
        self.num_threads = num_threads  # num_threads the kernel specialized for

    def load(self, name: str, index: sympy.Expr):
        var = self.args.input(name)
        index = self.rename_indexing(index)
        line = f"{var}[{cexpr(index)}]"
        if V.graph.get_dtype(name) in (torch.float16, torch.bfloat16):
            line = f"static_cast<float>({line})"
        return self.cse.generate(self.loads, line)

    def store(self, name, index, value, mode=None):
        assert "buf" in name
        var = self.args.output(name)
        index = self.rename_indexing(index)
        if mode is None:
            line = f"{var}[{cexpr(index)}] = {value};"
        elif mode == "atomic_add":
            if not config.cpp.dynamic_threads and self.num_threads == 1:
                line = f"{var}[{cexpr(index)}] += {value};"
            else:
                line = f"atomic_add(&{var}[{cexpr(index)}], {value});"
        else:
            raise NotImplementedError(f"store mode={mode}")
        self.stores.writeline(name, line)

    def reduction(self, name, dtype, src_dtype, reduction_type, index, value):
        argmax_or_argmin = reduction_type in {"argmax", "argmin"}
        tmpvar = self.cse.generate(
            self.loads, f"reduction {name} {cexpr(index)}", write=False
        )
        index = self.rename_indexing(index)
        self.reduction_vars[tmpvar] = reduction_type
        if argmax_or_argmin:
            self.reduction_prefix.writelines(
                argmax_argmin_prefix(reduction_type, src_dtype, tmpvar)
            )
            compare_op = "<" if reduction_type == "argmax" else ">"
            self.stores.writelines(
                None,
                [
                    f"if ({tmpvar}.value {compare_op} {value}) {{",
                    f"    {tmpvar}.index = {self.itervars[-1]}; {tmpvar}.value = {value};",
                    "}",
                ],
            )
        else:
            if dtype == torch.float16:
                self.reduction_prefix.writelines(
                    float16_reduction_prefix(reduction_type)
                )
            self.reduction_prefix.writeline(
                f"{DTYPE_TO_CPP[dtype]} {tmpvar} = {reduction_init(reduction_type, dtype)};"
            )
            self.stores.writeline(
                None, f"{reduction_combine(reduction_type, tmpvar, value)};"
            )

        if name not in V.graph.removed_buffers:
            var = self.args.output(name)
            member_name = ".index" if argmax_or_argmin else ""
            self.reduction_suffix.writeline(
                name, f"{var}[{cexpr(index)}] = {tmpvar}{member_name};"
            )
        self.cse.store_cache[name] = tmpvar

    def set_ranges(self, lengths, reduction_lengths):
        if self.call_ranges:
            assert self.call_ranges == tuple(lengths) + tuple(
                reduction_lengths
            ), f"{self.call_ranges} == {tuple(lengths)} + {tuple(reduction_lengths)}"
            assert self.reduction_depth == len(lengths)
        else:
            self.call_ranges = tuple(lengths) + tuple(reduction_lengths)
            self.ranges = [self.rename_indexing(x) for x in self.call_ranges]
            self.itervars = [sympy.Symbol(f"i{n}") for n in range(len(self.ranges))]
            self.reduction_depth = len(lengths)
        return (
            self.itervars[: self.reduction_depth],
            self.itervars[self.reduction_depth :],
        )

    def size_hint(self):
        return V.graph.sizevars.size_hint(sympy_product(self.call_ranges))

    def parallel_num_threads(self):
        threads = config.cpp.threads
        if threads < 1:
            threads = torch.get_num_threads()
        return threads

    def codegen_loops(self, code, worksharing):
        threads = self.parallel_num_threads()

        loops = [LoopLevel(var, size) for var, size in zip(self.itervars, self.ranges)]
        loops, reductions = LoopNest(loops[: self.reduction_depth]), LoopNest(
            loops[self.reduction_depth :]
        )
        reductions.mark_reduction(self.reduction_vars)

        if config.cpp.simdlen:
            # TODO(jansel): detect stride-1 dimension and vectorize that
            if reductions and len(reductions.loops) > 0:
                reductions.loops[-1].simd = True
            else:
                if len(loops.loops) > 0:
                    loops.loops[-1].simd = True

        par_depth = 0
        reduction_par_depth = 0
        if loops:
            par_depth = self.decide_parallel_depth(
                self.call_ranges[: self.reduction_depth], threads
            )
        else:
            reduction_par_depth = self.decide_parallel_depth(
                self.call_ranges[self.reduction_depth :], threads
            )

        with contextlib.ExitStack() as stack:
            if par_depth:
                worksharing.parallel(threads)
                loops.mark_parallel(par_depth)
            elif reduction_par_depth:
                # need to close the worksharing scope to define reduction vars outside it
                worksharing.close()
                reductions.mark_parallel(reduction_par_depth)
            elif threads > 1:
                if worksharing.single():
                    stack.enter_context(code.indent())

            loops.codegen(code, stack)

            with contextlib.ExitStack() as stack_outer:
                if self.reduction_prefix:
                    stack_outer.enter_context(code.indent())
                code.splice(self.reduction_prefix)

                if reduction_par_depth:
                    worksharing.parallel(threads)

                with contextlib.ExitStack() as stack:
                    reductions.codegen(code, stack)
                    code.splice(self.loads)
                    code.splice(self.compute)
                    code.splice(self.stores)

                if reduction_par_depth:
                    worksharing.close()

                code.splice(self.reduction_suffix)

    def decide_parallel_depth(self, ranges, threads):
        seq = self.size_hint()
        par = 1
        depth = 0
        for expr in ranges:
            hint = V.graph.sizevars.size_hint(expr)
            if par >= 2 * threads or par == threads:
                break
            if seq // threads < config.cpp.min_chunk_size:
                # not enough work
                break
            depth += 1
            par *= hint
            seq /= hint
        # if we assume thread number is dynamic, make sure we
        # have at least one parallel scope and let OMP runtime
        # to manage the serial vs. parallel.
        if config.cpp.dynamic_threads and depth == 0 and len(ranges) > 0:
            depth = 1
        return depth

    @contextlib.contextmanager
    def write_to_suffix(self):
        prior = (self.loads, self.compute, self.stores, self.cse)
        self.loads = IndentedBuffer()
        self.compute = IndentedBuffer()
        self.stores = DeferredIndentedBuffer()
        self.cse = self.cse.clone()
        yield
        self.reduction_suffix.splice(self.loads)
        self.reduction_suffix.splice(self.compute)
        self.reduction_suffix.splice(self.stores)
        (self.loads, self.compute, self.stores, self.cse) = prior


class CppVecKernel(CppKernel):
    overrides = CppVecOverrides

    def __init__(self, args, num_threads):
        super(CppVecKernel, self).__init__(args, num_threads)
        self.simd_len = config.cpp.simdlen
        metrics.generated_cpp_vec_kernel_count += 1

    def is_single_step_var(self, var: sympy.Symbol, index: sympy.Expr):
        replacement = {var: var + 1}
        new_index = sympy_subs(index, replacement)
        delta = sympy.simplify(new_index - index)
        return delta == 1

    def is_var_irrevelant(self, var: sympy.Symbol, index: sympy.Expr):
        expanded_index = sympy.expand(index)
        return not expanded_index.has(var)

    def transform_index(self, index: sympy.Expr):
        expanded_index = sympy.expand(index)
        assert self.simd_len
        assert self.simd_len > 0
        most_inner_var = self.itervars[-1]
        replacement = {most_inner_var: most_inner_var * self.simd_len}
        new_index = sympy_subs(expanded_index, replacement)
        return new_index

    def load(self, name: str, index: sympy.Expr):
        var = self.args.input(name)
        index = self.rename_indexing(index)

        expanded_index = sympy.expand(index)
        new_index = self.transform_index(index)

        if expanded_index == new_index:
            line = f"at::vec::Vectorized<float>({var}[{cexpr(index)}])"
        else:
            line = f"at::vec::Vectorized<float>::loadu({var} + {cexpr(new_index)})"

        return self.cse.generate(self.loads, line)

    def store(self, name, index, value, mode=None):
        assert "buf" in name
        var = self.args.output(name)
        index = self.rename_indexing(index)
        assert mode is None

        expanded_index = sympy.expand(index)
        new_index = self.transform_index(index)
        assert new_index != expanded_index
        line = f"{value}.store({var} + {cexpr(new_index)});"
        self.stores.writeline(name, line)


class CppVecKernelChecker(CppVecKernel):
    def __init__(self, args, num_threads):
        super(CppVecKernelChecker, self).__init__(args, num_threads)

        # Since this kernel is only for checker but does not genreate any
        # code, so we need to decrease the kernel count.
        metrics.generated_kernel_count -= 1
        metrics.generated_cpp_vec_kernel_count -= 1

        self.simd_vec = True
        self.fast_vec_list = []
        for dict_obj in CppVecOverrides.__dict__:
            if isinstance(CppVecOverrides.__dict__[dict_obj], staticmethod):
                self.fast_vec_list.append(dict_obj)
        self.exit_stack = contextlib.ExitStack()

    def is_legal_data_access(self, var: sympy.Symbol, index: sympy.Expr):
        return self.is_var_irrevelant(var, index) or self.is_single_step_var(var, index)

    def could_be_simd_vec(self, name: str, index: sympy.Expr):
        if V.graph.get_dtype(name) is not torch.float:
            return False

        assert self.itervars is not None
        # Not a loop
        if len(self.itervars) == 0:
            return False

        most_inner_var = self.itervars[-1]
        return self.is_legal_data_access(most_inner_var, index)

    def load(self, name: str, index: sympy.Expr):
        index = self.rename_indexing(index)

        self.simd_vec = self.simd_vec and self.could_be_simd_vec(name, index)
        return self.simd_vec

    def store(self, name, index, value, mode=None):
        assert "buf" in name
        index = self.rename_indexing(index)

        if mode:
            self.simd_vec = False
            return False

        self.simd_vec = self.simd_vec and self.could_be_simd_vec(name, index)
        return self.simd_vec

    def reduction(self, name, dtype, src_dtype, reduction_type, index, value):
        self.simd_vec = False
        return self.simd_vec

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.exit_stack.__exit__(exc_type, exc_val, exc_tb)

    def __enter__(self):
        class VecCheckerProxy:
            @staticmethod
            def __getattr__(name):
                def inner(*args, **kwargs):
                    if not (name in self.fast_vec_list):
                        self.simd_vec = False
                    return self.simd_vec

                return inner

            @staticmethod
            def load(name: str, index: sympy.Expr):
                return self.load(name, index)

            @staticmethod
            def store(name, index, value, mode=None):
                return self.store(name, index, value, mode=mode)

            @staticmethod
            def reduction(name, dtype, src_dtype, reduction_type, index, value):
                return self.reduction(
                    name, dtype, src_dtype, reduction_type, index, value
                )

            @staticmethod
            def index_expr(expr, dtype):
                self.simd_vec = False
                return self.cse.newvar()

            @staticmethod
            def indirect_indexing(index_var):
                return sympy.Symbol(str(index_var))

            @staticmethod
            def masked(mask, body, other):
                return V.kernel.cse.newvar()

        self.exit_stack.enter_context(V.set_ops_handler(VecCheckerProxy()))
        self.exit_stack.enter_context(V.set_kernel_handler(self))
        return self


class CppKernelProxy(CppKernel):
    def __init__(self, args=None, num_threads=None):
        super(CppKernelProxy, self).__init__(args, num_threads)
        self.simd_vec_kernel = None
        self.simd_omp_kernel = None

    def vectorize_most_inner_loop(self, loop_nest):
        loop_nest.split_most_inner_loop(config.cpp.simdlen)
        loop_with_tail = loop_nest.loops[-1]
        assert isinstance(loop_with_tail, LoopLevelWithTail)

        self.simd_vec_kernel.simd = False
        self.simd_vec_kernel.fast_vec = True

        loop_with_tail.tail_loop.simd_omp = True
        loop_with_tail.tail_loop.simd_len = int(config.cpp.simdlen / 2)
        loop_with_tail.tail_loop.simd_vec = False

        loop_with_tail.main_loop_body = self.simd_vec_kernel
        loop_with_tail.tail_loop_body = self.simd_omp_kernel
        return loop_nest

    def codegen_loops(self, code, worksharing):
        threads = self.parallel_num_threads()

        if self.simd_vec_kernel is None:
            assert self.simd_omp_kernel
            return self.simd_omp_kernel.codegen_loops(code, worksharing)

        assert self.simd_vec_kernel.itervars == self.simd_omp_kernel.itervars
        assert self.simd_vec_kernel.ranges == self.simd_omp_kernel.ranges

        itervars = self.simd_vec_kernel.itervars
        rangs = self.simd_vec_kernel.ranges
        loops = [LoopLevel(var, size) for var, size in zip(itervars, rangs)]

        # TODO: Support reductions
        loops_nest_non_reduc, _ = LoopNest(loops[: self.reduction_depth]), LoopNest(
            loops[self.reduction_depth :]
        )

        assert config.cpp.simdlen
        loops_nest_non_reduc.loops[-1].simd_omp = True

        par_depth = 0
        if loops_nest_non_reduc:
            par_depth = self.simd_vec_kernel.decide_parallel_depth(
                self.simd_vec_kernel.call_ranges[: self.reduction_depth], threads
            )

        with contextlib.ExitStack() as stack:
            if par_depth:
                worksharing.parallel(threads)
                loops_nest_non_reduc.mark_parallel(par_depth)
            elif threads > 1:
                if worksharing.single():
                    stack.enter_context(code.indent())

            self.vectorize_most_inner_loop(loops_nest_non_reduc)

            for loop in loops_nest_non_reduc.loops[0:-1]:
                code.writelines(loop.lines())
                stack.enter_context(code.indent())

            loop_with_tail: LoopLevelWithTail = loops_nest_non_reduc.loops[-1]
            for loop, kernel in (
                (loop_with_tail.main_loop, loop_with_tail.main_loop_body),
                (loop_with_tail.tail_loop, loop_with_tail.tail_loop_body),
            ):

                code.writelines(loop.lines())
                with contextlib.ExitStack() as stack:
                    stack.enter_context(code.indent())
                    code.splice(kernel.loads)
                    code.splice(kernel.compute)
                    code.splice(kernel.stores)


class CppScheduling:
    def __init__(self, scheduler):
        self.scheduler = scheduler
        self.kernel_group = KernelGroup()

    def group_fn(self, sizes):
        return tuple(tuple(map(V.graph.sizevars.simplify, s)) for s in sizes)

    @staticmethod
    def can_fuse_horizontal(node1, node2):
        _, (vars1, reduce1) = node1.group
        _, (vars2, reduce2) = node2.group
        if vars1 == vars2 and reduce1 == reduce2:
            return True
        if reduce1 == () and vars1 == vars2 + reduce2:
            return True
        # TODO(jansel): allow fusion pointwise (vars1, ()) suffix?
        return False

    @classmethod
    def can_fuse_vertical(cls, node1, node2):
        return cls.can_fuse_horizontal(node1, node2) and not node1.is_reduction()

    def can_vec(self, nodes):
        # TODO: Query cpu arch and vec length from aten
        if config.cpp.simdlen is None or config.cpp.simdlen != 8:
            return False

        _, (group, reduction_group) = max(
            nodes, key=lambda x: int(x.is_reduction())
        ).group

        with CppVecKernelChecker(
            self.kernel_group.args, self.kernel_group.ws.num_threads
        ) as kernel_checker:
            vars, reduction_vars = kernel_checker.set_ranges(group, reduction_group)
            for node in nodes:
                if node.group[1] in [
                    (group, reduction_group),
                    (group + reduction_group, ()),
                ]:
                    node.run(vars, reduction_vars)
                else:
                    assert node.group[1] == (
                        group,
                        (),
                    ), f"unexpected group: {node.group[1]} != {group}, {reduction_group}"
                    node.run(vars, ())

            return kernel_checker.simd_vec

    def _codegen_nodes_impl(self, nodes, is_simd_vec=False):
        """
        Turn an set of pre-fused nodes into a C++ kernel.
        """
        kernel_group = self.kernel_group
        _, (group, reduction_group) = max(
            nodes, key=lambda x: int(x.is_reduction())
        ).group
        in_suffix = False

        with kernel_group.new_kernel(is_simd_vec) as kernel:
            vars, reduction_vars = kernel.set_ranges(group, reduction_group)

            for node in nodes:
                if node.group[1] in [
                    (group, reduction_group),
                    (group + reduction_group, ()),
                ]:
                    assert not in_suffix
                    node.run(vars, reduction_vars)
                else:
                    in_suffix = True
                    assert node.group[1] == (
                        group,
                        (),
                    ), f"unexpected group: {node.group[1]} != {group}, {reduction_group}"
                    # we can fuse in some extra pointwise into the suffix
                    with kernel.write_to_suffix():
                        node.run(vars, ())

        return kernel

    def codegen_nodes(self, nodes):
        """
        Turn an set of pre-fused nodes into a C++ kernel.
        """
        kernel_group = self.kernel_group

        can_be_simd_vec = self.can_vec(nodes)
        simd_vec_kernel = (
            self._codegen_nodes_impl(nodes, can_be_simd_vec)
            if can_be_simd_vec
            else None
        )
        simd_omp_kernel = self._codegen_nodes_impl(nodes)

        assert simd_omp_kernel
        metrics.generated_kernel_count -= 1
        # Maitain the metrics kernel count
        if simd_vec_kernel:
            metrics.generated_kernel_count -= 1

        cpp_kernel_proxy = CppKernelProxy(
            kernel_group.args, kernel_group.ws.num_threads
        )
        cpp_kernel_proxy.simd_vec_kernel = simd_vec_kernel
        cpp_kernel_proxy.simd_omp_kernel = simd_omp_kernel

        kernel_group.finalize_kernel(cpp_kernel_proxy, None)

    def flush(self):
        self.kernel_group.codegen_define_and_call(V.graph.wrapper_code)
        self.kernel_group = KernelGroup()


class KernelGroup:
    def __init__(self):
        super().__init__()
        self.args = KernelArgs()
        self.loops_code = BracesBuffer()
        self.ws = WorkSharing(self.loops_code)
        self.stack = contextlib.ExitStack()
        self.stack.enter_context(self.ws)
        self.count = 0

    def new_kernel(self, simd_vec=False):
        if simd_vec:
            return CppVecKernel(self.args, self.ws.num_threads)
        else:
            return CppKernel(self.args, self.ws.num_threads)

    def finalize_kernel(self, new_kernel, scheduler):
        self.count += 1
        code = self.loops_code
        ws = self.ws
        new_kernel.codegen_loops(code, ws)

    def codegen_define_and_call(self, wrapper):
        self.stack.close()
        if self.count == 0:
            return

        arg_defs, call_args = self.args.cpp_argdefs()
        arg_defs = ",\n".ljust(25).join(arg_defs)
        code = BracesBuffer()
        code.writelines([cpp_prefix(), "" f'extern "C" void kernel({arg_defs})'])
        with code.indent():
            for old, new in self.args.aliases():
                code.writeline(f"auto {old} = {new};")
            code.splice(self.loops_code)

        codecache_def = IndentedBuffer()
        codecache_def.writeline("async_compile.cpp('''")
        codecache_def.splice(code)
        codecache_def.writeline("''')")

        kernel_name = wrapper.next_kernel_name()
        codecache_str = codecache_def.getvalue()
        # TODO(voz): Ostensibly, we should not need this. But there are cases where C++ codegen does
        # not use BracesBuffer, so we have no good indicator of a C++ buffer atm.
        codecache_str = codecache_str.replace("#pragma CMT", "//")
        wrapper.define_kernel(kernel_name, codecache_str)

        # generate the code to call this
        wrapper.writeline(
            "{}({})".format(kernel_name, ", ".join(call_args)),
        )


class WorkSharing:
    def __init__(self, code):
        self.code = code
        self.in_parallel = False
        self.num_threads = None
        self.stack = contextlib.ExitStack()

    def parallel(self, threads):
        if self.in_parallel and threads != self.num_threads:
            # wrong number of threads
            self.close()
        if not self.in_parallel:
            self.num_threads = threads
            self.in_parallel = True
            if config.cpp.dynamic_threads:
                self.code.writeline("#pragma omp parallel")
            else:
                self.code.writeline(f"#pragma omp parallel num_threads({threads})")
            self.stack.enter_context(self.code.indent())

    def single(self):
        if self.in_parallel:
            self.code.writeline("#pragma omp single")
        return self.in_parallel

    def close(self):
        self.stack.close()
        self.in_parallel = False

    def __enter__(self):
        self.stack.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stack.__exit__(exc_type, exc_val, exc_tb)


@dataclasses.dataclass
class LoopLevel:
    var: sympy.Expr = None
    size: sympy.Expr = None
    offset: sympy.Expr = sympy.Integer(0)
    steps: sympy.Expr = sympy.Integer(1)
    parallel: int = 0
    simd_omp: bool = False
    simd_len: int = config.cpp.simdlen
    simd_vec: bool = False
    collapsed: bool = False
    reduction_vars: Dict[str, str] = None

    def lines(self):
        if self.reduction_vars:
            reduction = " " + " ".join(
                f"reduction({RTYPE_TO_CPP[rtype]}:{var})"
                for var, rtype in self.reduction_vars.items()
            )
        else:
            reduction = ""
        simd = f"simd simdlen({self.simd_len})" if self.simd_omp else ""
        if self.parallel:
            # TODO(jansel): look into chunk size and other schedules
            line1 = f"#pragma omp for{reduction} "
            if self.parallel > 1:
                line1 += f" collapse({self.parallel})"
            if self.simd_omp:
                line1 = line1.replace(" for ", f" for {simd}")
        elif self.simd_vec:
            line1 = ""
        elif self.simd_omp:
            line1 = f"#pragma omp {simd}{reduction}"
        elif not self.reduction_vars and codecache.is_gcc():
            line1 = "#pragma GCC ivdep"
        else:
            line1 = ""
        line2 = f"for({INDEX_TYPE} {self.var}={cexpr(self.offset)}; {self.var}<{cexpr(self.size)}; {self.var}+={cexpr(self.steps)})"
        if self.collapsed or not line1:
            return [line2]
        return [line1, line2]


class LoopLevelWithTail(LoopLevel):
    def __init__(self, main_loop: LoopLevel, tail_loop: LoopLevel):
        super().__init__()
        self.main_loop = main_loop
        self.tail_loop = tail_loop
        self.main_loop_body = None
        self.tail_loop_body = None

    def lines(self):
        raise AssertionError("Not Implemented")


@dataclasses.dataclass
class LoopNest:
    loops: List[LoopLevel]

    def __bool__(self):
        return bool(self.loops)

    def mark_reduction(self, reduction_vars):
        for loop in self.loops:
            loop.reduction_vars = reduction_vars

    def mark_parallel(self, par_depth):
        loops = self.loops
        loops[0].parallel = par_depth
        for i in range(1, par_depth):
            loops[i].collapsed = True

    def split_most_inner_loop(self, factor):
        sympy_factor = sympy.Integer(factor)

        most_inner_loop = self.loops[-1]
        main_loop_range = ir.IndexingDiv(most_inner_loop.size, sympy_factor)

        main_loop = LoopLevel(most_inner_loop.var, main_loop_range)
        main_loop.parallel = most_inner_loop.parallel

        offset = main_loop_range * sympy_factor
        tail_loop = LoopLevel(most_inner_loop.var, most_inner_loop.size)
        tail_loop.offset = offset
        tail_loop.parallel = most_inner_loop.parallel

        loop_with_tail = LoopLevelWithTail(main_loop, tail_loop)
        self.loops[-1] = loop_with_tail

    def codegen(self, code, stack):
        for loop in self.loops:
            code.writelines(loop.lines())
            stack.enter_context(code.indent())
        else:
            stack.enter_context(code.indent())
