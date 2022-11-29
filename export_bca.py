from .util import *
from math import degrees

class AnimDesc:
    def __init__(self, vals, bytes_per_elem):
        self.vals = vals
        self.one = False
        self.interp = False
        self.bytes_per_elem = bytes_per_elem

        if all(v == vals[0] for v in vals):
            self.vals = [self.vals[0]]
            self.one = True

        # interpolation range
        ir = (len(vals) - 1) // 2 * 2 + 1
        if not self.one and all(v1 == (v0 + v2) // 2 for v0, v1, v2 in
                zip(vals[0:ir-2:2], vals[1:ir-1:2], vals[2:ir:2])):
            del self.vals[1:ir-1:2]
            self.interp = True

        self.bytestr = from_vec(self.vals, bytes_per_elem, 0)

    def add_to_bytearr(self, bytearr):
        self.index = -2
        while not (self.index == -1 or 
                (self.index >= 0 and self.index % self.bytes_per_elem == 0)):
            self.index = bytearr.find(self.bytestr)

        if self.index == -1: # Not found
            self.index = len(bytearr)
            bytearr += self.bytestr

        self.index //= self.bytes_per_elem # index, not offset

    def export(self):
        bytestr = bytearray()
        bytestr += from_uint(self.interp, 1)
        bytestr += from_uint(not self.one, 1)
        bytestr += from_uint(self.index, 2)
        return bytestr


class AnimBone:
    def __init__(self, transforms):
        self.scales = [m.to_scale() for m in transforms]
        self.rots = [[degrees(e) for e in m.to_euler("XYZ")] for m in transforms]
        self.transs = [m.to_translation() for m in transforms]

        scales_xyz = [[fix_to_int(v[i], 12) for v in self.scales] for i in range(3)]
        rots_xyz = [[(deg_to_int(v[i]) / 16) for v in self.rots] for i in range(3)]
        transs_xyz = [[fix_to_int(v[i], 12) for v in self.transs] for i in range(3)]

        self.descs = [AnimDesc(vs, bpe) for vs, bpe in 
                zip(scales_xyz + rots_xyz + transs_xyz, [4, 4, 4, 2, 2, 2, 4, 4, 4])]

    def add_to_bytestrs(self, scale_bytestr, rot_bytestr, trans_bytestr):
        for desc, b in zip(self.descs,
                [scale_bytestr] * 3 + [rot_bytestr] * 3 + [trans_bytestr] * 3):
            desc.add_to_bytearr(b)

    def export(self):
        return b"".join(desc.export() for desc in self.descs)
    

def export_anim(context, obj):
    mtxs = []
    fraam = context.scene.frame_current
    for frame in range(context.scene.frame_start, context.scene.frame_end + 1):
        context.scene.frame_set(frame)
        mtxs.append([b.matrix.copy() for b in obj.pose.bones])
    context.scene.frame_set(fraam)

    num_frames = len(mtxs)
    bones = [AnimBone([m[i] for m in mtxs]) for i in range(len(obj.data.bones))]

    scale_bytestr = bytearray()
    rot_bytestr = bytearray()
    trans_bytestr = bytearray()
    for bone in bones:
        bone.add_to_bytestrs(scale_bytestr, rot_bytestr, trans_bytestr)

    bytestr_list = BytesWithPtrs()

    header = bytearray()
    header_aligned = AlignedBytes(header, 4)

    # Number of bones and frames, loop flag
    header += from_uint(len(bones), 2)
    header += from_uint(num_frames, 2)
    header += from_uint(1, 4)

    # Scale, rotation, translation values
    header += from_uint(0, 4) # pointer
    scale_marker = AlignedBytes(b'', 4)
    scale_ptr = BytesPtr(header_aligned, 0x8, scale_marker, 0, 4)

    header += from_uint(0, 4) # pointer
    rot_marker = AlignedBytes(b'', 4)
    rot_ptr = BytesPtr(header_aligned, 0xc, rot_marker, 0, 4)

    header += from_uint(0, 4) # pointer
    trans_marker = AlignedBytes(b'', 4)
    trans_ptr = BytesPtr(header_aligned, 0x10, trans_marker, 0, 4)

    header += from_uint(0, 4) # pointer
    anim_marker = AlignedBytes(b'', 4)
    anim_ptr = BytesPtr(header_aligned, 0x14, anim_marker, 0, 4)
    
    bytestr_list.bytestrs.append(header_aligned)
    bytestr_list.bytestrs.append(scale_marker)
    bytestr_list.bytestrs.append(AlignedBytes(scale_bytestr, 4))
    bytestr_list.bytestrs.append(rot_marker)
    bytestr_list.bytestrs.append(AlignedBytes(rot_bytestr, 4))
    bytestr_list.bytestrs.append(trans_marker)
    bytestr_list.bytestrs.append(AlignedBytes(trans_bytestr, 4))
    bytestr_list.bytestrs.append(anim_marker)
    bytestr_list.bytestrs.append(AlignedBytes(b"".join(b.export() for b in bones), 4))

    bytestr_list.ptrs += [scale_ptr, rot_ptr, trans_ptr, anim_ptr]

    return bytestr_list.assemble()


def save(context, filepath):
    bytestr = export_anim(context, context.active_object)
    with open(filepath, "wb") as f:
        f.write(bytestr)
        
    return {"FINISHED"}
