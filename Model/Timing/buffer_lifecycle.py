"""Typed whole-window ownership for synthetic lifecycle diagnostics.

Generations and leases are analysis identities, not hardware generation tags.
No measured time, native cost or implicit reset is accepted here.
"""
from dataclasses import dataclass, field
from analysis_errors import InvalidAnalysisInput, UnsupportedAnalysis


def integer(value, label, minimum=0, maximum=None):
    if type(value) is not int or value < minimum or maximum is not None and value > maximum:
        raise InvalidAnalysisInput('invalid_lifecycle_integer', label+' requires an exact bounded integer')


def name(value, label):
    if not isinstance(value,str) or not value or value.strip()!=value:
        raise InvalidAnalysisInput('invalid_lifecycle_identity', label+' requires an explicit identity')


def evidence(value):
    if not isinstance(value,tuple) or not value or any(not isinstance(x,str) or not x.strip() for x in value):
        raise InvalidAnalysisInput('invalid_lifecycle_evidence','Explicit immutable scope evidence required')


@dataclass(frozen=True)
class ClockDomain:
    period_ps: int
    phase_ps: int = 0

    def validate(self):
        integer(self.period_ps,'clock period',1);integer(self.phase_ps,'clock phase')
        if self.phase_ps>=self.period_ps:raise InvalidAnalysisInput('invalid_lifecycle_clock','Phase must be less than period')

    def edge(self,time):
        integer(time,'clock edge time')
        return self.phase_ps+max(0,(time-self.phase_ps+self.period_ps-1)//self.period_ps)*self.period_ps


@dataclass(frozen=True)
class MemoryTileId:
    physical_col: int
    physical_row: int
    architecture: str = 'AIE1'


def validate_clock(clock):
    if not isinstance(clock,ClockDomain):raise UnsupportedAnalysis('unknown_lifecycle_clock','A declared clock domain is required')
    clock.validate()


@dataclass(frozen=True)
class SlotId:
    tile: MemoryTileId
    byte_start: int
    byte_length: int


@dataclass(frozen=True)
class BinaryLockId:
    tile: MemoryTileId
    local_id: int


def validate_tile(tile):
    if not isinstance(tile,MemoryTileId):raise InvalidAnalysisInput('missing_memory_tile_identity','Typed physical memory tile required')
    integer(tile.physical_col,'tile column');integer(tile.physical_row,'tile row')
    if tile.architecture!='AIE1':raise UnsupportedAnalysis('unsupported_lifecycle_architecture','AIE1 candidate scope only')
    if tile.physical_row==0:raise UnsupportedAnalysis('shim_is_not_aie_memory_tile','This scope uses AIE1 memory modules, not row-zero shim resources')


def validate_slot_id(slot):
    if not isinstance(slot,SlotId):raise InvalidAnalysisInput('missing_slot_identity','Typed complete extent required')
    validate_tile(slot.tile);integer(slot.byte_start,'slot address',0,32767);integer(slot.byte_length,'slot length',1,32768)
    if slot.byte_start+slot.byte_length>32768:
        raise InvalidAnalysisInput('aie1_memory_extent_overflow','Complete local extent must fit the AIE1 32 KiB memory module')


def validate_lock_id(lock):
    if not isinstance(lock,BinaryLockId):raise InvalidAnalysisInput('missing_binary_lock_identity','Typed complete binary lock identity required')
    validate_tile(lock.tile);integer(lock.local_id,'binary lock ID',0,15)


@dataclass(frozen=True)
class SlotSpec:
    id: SlotId
    lock_id: BinaryLockId
    aliases: tuple[str,...]
    evidence: tuple[str,...]


@dataclass(frozen=True)
class BinaryLockSpec:
    id: BinaryLockId
    initial_value: int | None
    acquire_ps: int | None
    release_visible_ps: int | None
    clock: ClockDomain
    evidence: tuple[str,...]
    initial_owner: None = None


@dataclass(frozen=True)
class GenerationId:
    slot_id: SlotId
    ordinal: int


@dataclass(frozen=True)
class LeaseId:
    actor: str
    occurrence: int


@dataclass
class BufferGeneration:
    id: GenerationId
    frame_id: int
    payload: tuple[int,...] | None = None
    written_bytes: int = 0
    published: bool = False
    retired: bool = False
    leases: dict = field(default_factory=dict)
    reads_requiring_memory: set = field(default_factory=set)


class BufferLifecycle:
    """Ownership transitions; scheduling and visibility are the coordinator's job."""
    def __init__(self,slots,locks):
        self.slots={};self.locks={};self.current={};self.history={};self.next_generation={};self.used_leases=set()
        for lock in locks:
            if not isinstance(lock,BinaryLockSpec):raise InvalidAnalysisInput('missing_binary_lock_spec','Typed binary lock required')
            validate_lock_id(lock.id);evidence(lock.evidence);validate_clock(lock.clock)
            if lock.initial_value is None or lock.acquire_ps is None or lock.release_visible_ps is None:
                raise UnsupportedAnalysis('unknown_lifecycle_lock','Lock reset and service costs cannot be guessed')
            if type(lock.initial_value)is not int or lock.initial_value not in (0,1) or lock.initial_owner is not None:
                raise InvalidAnalysisInput('invalid_binary_lock_reset','Only known unowned binary reset is supported')
            integer(lock.acquire_ps,'lock acquire');integer(lock.release_visible_ps,'lock release visibility')
            if lock.id in self.locks:raise InvalidAnalysisInput('duplicate_lifecycle_lock','Complete lock identity must be unique')
            self.locks[lock.id]={'spec':lock,'value':lock.initial_value,'owner':None,'generation':None,'release_pending':False}
        for slot in slots:
            if not isinstance(slot,SlotSpec):raise InvalidAnalysisInput('missing_lifecycle_slot','Typed slot required')
            evidence(slot.evidence);validate_slot_id(slot.id);validate_lock_id(slot.lock_id)
            if slot.lock_id.tile!=slot.id.tile or slot.lock_id not in self.locks:
                raise InvalidAnalysisInput('unbound_lifecycle_lock','Slot needs its complete local memory lock')
            if not isinstance(slot.aliases,tuple) or not slot.aliases or len(set(slot.aliases))!=len(slot.aliases):
                raise InvalidAnalysisInput('invalid_slot_aliases','Complete distinct aliases required')
            for alias in slot.aliases:name(alias,'slot alias')
            if slot.id in self.slots:raise InvalidAnalysisInput('duplicate_slot_extent','Same extent is one slot, not two buffers')
            for old in self.slots:
                if old.tile==slot.id.tile and max(old.byte_start,slot.id.byte_start)<min(old.byte_start+old.byte_length,slot.id.byte_start+slot.id.byte_length):
                    raise UnsupportedAnalysis('overlapping_slot_extents','Partially overlapping extents are not independent slots')
            self.slots[slot.id]=slot;self.current[slot.id]=None;self.next_generation[slot.id]=0
        if not self.slots:raise InvalidAnalysisInput('empty_lifecycle_slots','At least one window slot required')
        if len({s.lock_id for s in self.slots.values()})!=len(self.slots):
            raise UnsupportedAnalysis('shared_slot_lock','First candidate scope requires one lock per slot')

    def can_acquire(self,slot_id,mode,frame_id):
        validate_slot_id(slot_id);integer(frame_id,'frame ID')
        if slot_id not in self.slots or mode not in ('READ','WRITE'):
            raise InvalidAnalysisInput('invalid_lifecycle_acquire','Bound slot and READ/WRITE mode required')
        lock=self.locks[self.slots[slot_id].lock_id];generation=self.current[slot_id]
        if lock['owner'] is not None or lock['value']!=int(mode=='READ'):return False
        if mode=='WRITE':return generation is None or generation.retired and not generation.leases and not generation.reads_requiring_memory
        return generation is not None and generation.frame_id==frame_id and generation.published and not generation.retired

    def acquire(self,slot_id,mode,frame_id,lease):
        if not isinstance(lease,LeaseId):raise InvalidAnalysisInput('missing_generation_lease','Typed lease identity required')
        name(lease.actor,'lease actor');integer(lease.occurrence,'lease occurrence');integer(frame_id,'frame ID')
        if not self.can_acquire(slot_id,mode,frame_id):
            raise InvalidAnalysisInput('ineligible_lifecycle_acquire','Acquisition cannot bypass lock, generation or publication')
        if lease in self.used_leases:
            raise InvalidAnalysisInput('duplicate_generation_lease','Lease identity belongs to one occurrence only')
        if mode=='WRITE':
            gid=GenerationId(slot_id,self.next_generation[slot_id]);self.next_generation[slot_id]+=1
            generation=BufferGeneration(gid,frame_id);self.current[slot_id]=generation;self.history[gid]=generation
        else:generation=self.current[slot_id]
        self.used_leases.add(lease)
        generation.leases[lease]=mode
        if mode=='READ':generation.reads_requiring_memory.add(lease)
        lock=self.locks[self.slots[slot_id].lock_id];lock.update(owner=lease,generation=generation.id)
        return generation.id

    def _owned(self,gid,lease,mode=None):
        if not isinstance(gid,GenerationId) or not isinstance(lease,LeaseId):
            raise InvalidAnalysisInput('missing_owned_generation_identity','Typed generation and lease identities required')
        validate_slot_id(gid.slot_id);integer(gid.ordinal,'generation ordinal')
        name(lease.actor,'lease actor');integer(lease.occurrence,'lease occurrence')
        generation=self.history.get(gid)
        if generation is None or self.current.get(gid.slot_id) is not generation or generation.retired or lease not in generation.leases:
            raise InvalidAnalysisInput('stale_generation_lease','Event must address its still-owned generation and lease')
        if mode is not None and generation.leases[lease]!=mode:
            raise InvalidAnalysisInput('wrong_generation_lease_mode','Lease mode does not permit this action')
        return generation

    def write_complete(self,gid,lease,payload):
        g=self._owned(gid,lease,'WRITE')
        if not isinstance(payload,tuple) or len(payload)!=gid.slot_id.byte_length or any(type(x)is not int or not 0<=x<=255 for x in payload):
            raise InvalidAnalysisInput('invalid_generation_payload','Exactly one full byte extent required')
        if g.payload is not None:raise InvalidAnalysisInput('duplicate_generation_write','Extent cannot be written twice')
        g.payload=payload;g.written_bytes=len(payload)

    def publish(self,gid,lease):
        g=self._owned(gid,lease,'WRITE')
        if g.published or g.written_bytes!=gid.slot_id.byte_length:
            raise InvalidAnalysisInput('invalid_generation_publish','Publication needs one complete written extent')
        g.published=True

    def safe_copy(self,gid,lease):
        g=self._owned(gid,lease,'READ')
        if lease not in g.reads_requiring_memory:
            raise InvalidAnalysisInput('duplicate_generation_copy','Local memory read may complete only once')
        g.reads_requiring_memory.remove(lease)
        return tuple(g.payload)

    def release_request(self,gid,lease,value):
        g=self._owned(gid,lease)
        if type(value)is not int or value not in (0,1):raise InvalidAnalysisInput('invalid_binary_lock_value','Binary release value required')
        mode=g.leases[lease];lock=self.locks[self.slots[gid.slot_id].lock_id]
        if lock['owner']!=lease or lock['generation']!=gid or lock['release_pending']:
            raise InvalidAnalysisInput('duplicate_or_stale_lock_release','Only its owning lease can request release once')
        if (mode=='WRITE' and (value!=1 or not g.published)) or (mode=='READ' and (value!=0 or lease in g.reads_requiring_memory)):
            raise InvalidAnalysisInput('unsafe_generation_release','Release cannot expose partial data or destroy unread source bytes')
        lock['release_pending']=True;lock['pending_value']=value

    def release_visible(self,gid,lease,value):
        if type(value)is not int or value not in (0,1):
            raise InvalidAnalysisInput('invalid_binary_lock_value','Visible release requires an exact binary value')
        g=self._owned(gid,lease);lock=self.locks[self.slots[gid.slot_id].lock_id]
        if lock['owner']!=lease or lock['generation']!=gid or not lock['release_pending'] or lock.get('pending_value')!=value:
            raise InvalidAnalysisInput('stale_lock_visibility','Visible release must match its pending owner and generation')
        mode=g.leases.pop(lease)
        lock.update(value=value,owner=None,generation=None,release_pending=False)
        if mode=='READ':g.retired=True
