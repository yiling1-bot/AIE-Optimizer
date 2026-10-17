"""Ordered, versioned affine access proofs for descriptor refinements.

This validates a finite sequence, not merely its histogram. It does not emit DMA
programs or establish multi-channel deadlock freedom. Hardware limits must be
provided by the caller; no AIE1 limits are silently applied to AIE-ML.
"""
from dataclasses import dataclass
from itertools import product
from math import prod
from .registry import Decision


@dataclass(frozen=True)
class Access:
    buffer: str
    epoch: str
    sizes: tuple
    strides: tuple  # in elements, outermost to innermost
    offset: int
    allocation_elements: int
    element_bytes: int
    write: bool = False

    def addresses(self):
        for index in product(*(range(n) for n in self.sizes)):
            yield self.offset+sum(i*s for i,s in zip(index,self.strides))


def check_access(reference,candidate,limits=None,max_elements=1000000):
    if type(max_elements) is not int or max_elements<1:raise ValueError('positive proof budget required')
    if not isinstance(reference,Access) or not isinstance(candidate,Access):
        raise ValueError('typed access descriptors required')
    for a in (reference,candidate):
        if (not a.buffer or not a.epoch or type(a.write) is not bool or
            not isinstance(a.sizes,tuple) or not isinstance(a.strides,tuple) or
            not a.sizes or len(a.sizes)!=len(a.strides) or
            any(type(x) is not int for x in a.sizes+a.strides+(a.offset,a.allocation_elements,a.element_bytes,)) or
            min(a.sizes)<=0 or a.offset<0 or a.allocation_elements<=0 or a.element_bytes<=0):
            return Decision('DENY','invalid affine access descriptor')
        low=a.offset+sum(min(0,(n-1)*s) for n,s in zip(a.sizes,a.strides))
        high=a.offset+sum(max(0,(n-1)*s) for n,s in zip(a.sizes,a.strides))
        if low<0 or high>=a.allocation_elements:return Decision('DENY','access exceeds allocation')
    if any(getattr(reference,k)!=getattr(candidate,k) for k in ('buffer','epoch','element_bytes','write')):
        return Decision('DENY','buffer identity, input version, dtype width or effects differ')
    if prod(reference.sizes)!=prod(candidate.sizes):return Decision('DENY','access counts differ')
    if prod(reference.sizes)>max_elements:return Decision('UNKNOWN','ordered access proof budget exceeded')
    seen=set()
    for a,b in zip(reference.addresses(),candidate.addresses()):
        if a!=b:return Decision('DENY','ordered element sequence differs')
        if reference.write and b in seen:return Decision('DENY','overlapping destination writes')
        seen.add(b)
    required={'max_dimensions','max_dimension_size','max_abs_stride','address_alignment_bytes'}
    if limits is None:return Decision('UNKNOWN','sequence equivalent; target descriptor limits missing')
    if not isinstance(limits,dict) or set(limits)!=required or any(type(v) is not int or v<1 for v in limits.values()):
        raise ValueError('explicit positive target descriptor limits required')
    if (len(candidate.sizes)>limits['max_dimensions'] or max(candidate.sizes)>limits['max_dimension_size'] or
        max(abs(s) for s in candidate.strides)>limits['max_abs_stride'] or
        candidate.offset*candidate.element_bytes%limits['address_alignment_bytes']):
        return Decision('DENY','candidate exceeds supplied hardware limits/alignment')
    return Decision('ALLOW','identical ordered versioned access sequence within supplied target limits')
