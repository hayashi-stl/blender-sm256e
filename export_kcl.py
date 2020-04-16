from .util import *
from .kcl_util import *
from math import log2

def export_mesh(context, obj, scale=1.0, one_clps_index=False):
    mesh = obj.data
    tri_mod = obj.modifiers.new("Triangulate", "TRIANGULATE")
    mesh = obj.to_mesh(context.scene, True, "RENDER")
    obj.modifiers.remove(tri_mod)

    tris = [KclTriangle([mesh.vertices[v].co * scale for v in f.vertices],
        f.normal, 0 if one_clps_index else f.material_index) 
            for f in mesh.polygons]
    kcl_mesh = KclMesh(tris)
    
    bytestr_list = BytesWithPtrs()

    header = bytearray()
    header_aligned = AlignedBytes(header, 4)
    
    header += from_uint(0, 4) # pointer
    vertex_list = AlignedBytes(b"".join(kcl_mesh.vertex_list), 4)
    vertex_ptr = BytesPtr(header_aligned, 0x0, vertex_list, 0, 4)

    header += from_uint(0, 4) # pointer
    normal_list = AlignedBytes(b"".join(kcl_mesh.normal_list), 4)
    normal_ptr = BytesPtr(header_aligned, 0x4, normal_list, 0, 4)

    header += from_uint(0, 4) # pointer
    tri_list = AlignedBytes(kcl_mesh.export(), 4)
    tri_ptr = BytesPtr(header_aligned, 0x8, tri_list, -0x10, 4)

    octree = Octree.create(kcl_mesh, 15, 1)
    header += from_uint(0, 4) # pointer
    octree_bytes = AlignedBytes(octree.export(), 4)
    octree_ptr = BytesPtr(header_aligned, 0xc, octree_bytes, 0, 4)

    header += from_uint(327680, 4) # unknown
    header += from_vec(octree.base, 4, 6)
    header += from_uint_list([~(int(c) - 1) % 2 ** 32 for c in octree.widths], 4)
    header += from_uint(int(log2(octree.base_width)), 4)
    header += from_uint(int(log2(octree.num_c[0])), 4)
    header += from_uint(int(log2(octree.num_c[0])) + int(log2(octree.num_c[1])), 4)

    bytestr_list.bytestrs.append(header_aligned)
    bytestr_list.bytestrs.append(vertex_list)
    bytestr_list.bytestrs.append(normal_list)
    bytestr_list.bytestrs.append(tri_list)
    bytestr_list.bytestrs.append(octree_bytes)
    bytestr_list.ptrs += [vertex_ptr, normal_ptr, tri_ptr, octree_ptr]

    return bytestr_list.assemble()


def save(context, filepath, *, scale=1.0, one_clps_index=False):
    obj = context.active_object
    bytestr = export_mesh(context, obj, scale=scale, one_clps_index=one_clps_index)
    with open(filepath, "wb") as f:
        f.write(bytestr)
        
    return {"FINISHED"}
