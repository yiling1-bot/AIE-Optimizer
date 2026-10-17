"""Count reported physical resources without double-counting aliased ports."""
def compiler_resources(report):
    mapping=report['mapping']; allocations=set(); endpoints=set()
    def memory(a):
        fields=(a['tile'],a['column'],a['row'],a['offset'],a['size'])
        if any(type(v) is not int or v<0 for v in fields[1:]):
            raise ValueError('invalid memory allocation')
        allocations.add(fields)
    for port in mapping['portInstanceMapping'].values():
        for allocation in port.get('bufferInfo',[]): memory(allocation)
        dma=port.get('dmaInfo')
        if dma:
            endpoints.add((dma['tile'],dma['column'],dma['row'],dma['direction'],dma['channel']))
    for allocation in mapping.get('systemMemory',[]): memory(allocation)
    modules={}
    for kind,col,row,offset,size in allocations:
        modules.setdefault((kind,col,row),[]).append((offset,offset+size))
    total=0; details=[]
    for key,spans in sorted(modules.items()):
        union=[]
        for lo,hi in sorted(spans):
            if union and lo<=union[-1][1]: union[-1][1]=max(hi,union[-1][1])
            else: union.append([lo,hi])
        occupied=sum(hi-lo for lo,hi in union);total+=occupied
        details.append({'module':list(key),'address_union_bytes':occupied,'max_address_end':max(hi for _,hi in spans)})
    return {'memory_modules':len(modules),'reported_address_union_bytes':total,
            'unique_reported_allocations':len(allocations),'dma_endpoints':len(endpoints),
            'memory':details,'scope':'compiler-reported port buffers and systemMemory; not dynamic live-byte peak or all device resources'}
