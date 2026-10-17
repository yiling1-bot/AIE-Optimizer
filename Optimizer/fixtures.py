"""Deterministic synthetic fixtures and an independent row-major reference."""
import random


def reference(x,w0,w1,relu):
    def layer(a,b,activate):
        out=[]
        for row in a:
            vals=[]
            for col in zip(*b):
                v=sum(u*v for u,v in zip(row,col))
                v=max(-128,min(127,v))
                vals.append(max(0,v) if activate else v)
            out.append(vals)
        return out
    return layer(layer(x,w0,relu),w1,False)


def pack(a,rm,cn):
    m,n=len(a),len(a[0])
    if m%rm or n%cn: raise ValueError('incomplete packing tile')
    return [a[i+r][j+c] for i in range(0,m,rm) for j in range(0,n,cn)
            for r in range(rm) for c in range(cn)]


def make_frames(spec, count=4, seed=20261014):
    if type(count) is not int or count<1: raise ValueError('positive frame count required')
    rng=random.Random(seed); out=[]
    for f in range(count):
        values=((-3,-1,0,1,2,4),(-128,-127,-1,0,1,126,127),(-8,-2,1,3,7))[f%3]
        matrices={name:[[rng.choice(values) for _ in range(n)] for _ in range(m)]
                  for name,m,n in [('X',spec.m,spec.k),('W0',spec.k,spec.hidden),('W1',spec.hidden,spec.n)]}
        gold=reference(matrices['X'],matrices['W0'],matrices['W1'],spec.hidden_relu)
        out.append({**{k:pack(v,4 if k=='X' else 8,8) for k,v in matrices.items()},
                    'Y':pack(gold,4,8)})
    return out
