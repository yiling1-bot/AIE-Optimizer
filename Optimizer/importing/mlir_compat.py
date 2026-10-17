"""Small audited compatibility definitions for xDSL 0.71's MLIR gaps.

Upstream MLIR allows tensor integer casts and signed widening in linalg.matmul.
xDSL 0.71 casts are scalar-only and its named matmul body omits widening. Keep
these adaptations local to our parsing context; never patch the dependency.
"""
from xdsl.dialects import arith
from xdsl.dialects.linalg import Linalg
from xdsl.dialects.linalg.ops import MatmulOp, YieldOp
from xdsl.dialects.builtin import IntegerType
from xdsl.ir import Dialect
from xdsl.irdl import IRDLOperation, irdl_op_definition, operand_def, result_def, traits_def
from xdsl.traits import Pure
from xdsl.builder import Builder


class IntegerCast(IRDLOperation):
    input = operand_def()
    result = result_def()
    traits = traits_def(Pure())
    assembly_format = '$input attr-dict `:` type($input) `to` type($result)'

    def verify_(self):
        from .mlir_frontend import mlir_type
        a,b = mlir_type(self.input.type),mlir_type(self.result.type)
        if a['shape'] != b['shape'] or not (a['bits'] < b['bits'] if self.name == 'arith.extsi' else a['bits'] > b['bits']):
            raise ValueError('integer cast must preserve shape and have valid bit widths')


@irdl_op_definition
class ExtSI(IntegerCast):
    name = 'arith.extsi'


@irdl_op_definition
class TruncI(IntegerCast):
    name = 'arith.trunci'


class SignedMatmul(MatmulOp):
    @classmethod
    def get_hidden_region(cls, inputs, outputs):
        types = cls.body_arg_types((*inputs,*outputs))
        if not all(isinstance(t,IntegerType) for t in types):
            return super().get_hidden_region(inputs,outputs)
        @Builder.implicit_region(types)
        def body(args):
            a,b,acc = args
            if a.type != acc.type: a = ExtSI(operands=[a],result_types=[acc.type]).result
            if b.type != acc.type: b = ExtSI(operands=[b],result_types=[acc.type]).result
            product = arith.MuliOp(a,b)
            summed = arith.AddiOp(product,acc)
            YieldOp(summed)
        return body


def dialects():
    casts = {'arith.extsi':ExtSI,'arith.trunci':TruncI}
    arithmetic = Dialect('arith',[casts.get(op.name,op) for op in arith.Arith.operations],list(arith.Arith.attributes))
    linalg = Dialect('linalg',[SignedMatmul if op.name == 'linalg.matmul' else op for op in Linalg.operations],list(Linalg.attributes))
    return arithmetic,linalg
