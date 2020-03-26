import bpy
import math
from mathutils import Color, Vector, Matrix
from itertools import chain
from .util import *

def export_bone(bone, bones, materials, mesh, meshes, group_names):
    """ Returns (bone_data, other_data, ptrs) where:
    bone_data: AlignedBytes is the bytestring of the bone
    name_data: [AlignedBytes] is a list of relevant detached bytestrings
    ptrs: [BytesPtr] is a list of relevant pointers
    
    If bone is None, the model is exported as rooms, so mesh and meshes must be provided.
    """
    bytestr = bytearray()
    aligned = AlignedBytes(bytestr, 4)

    num_bones = len(bones) if bone else len(meshes)
    # Bone ID
    bone_id = bones.find(bone.name) if bone else meshes.index(mesh)
    bytestr += from_uint(bone_id, 4)
    
    # Name
    name_bytes = AlignedBytes(str_to_cstr(bone.name if bone else "r" + str(bone_id)), 1)
    bytestr += from_uint(0, 4)
    name_ptr = BytesPtr(aligned, 4, name_bytes, 0, 4)

    # Offset to parent
    bytestr += from_int(bones.find(bone.parent.name) - bone_id \
            if bone and bone.parent else 0, 2)
    # Has children
    bytestr += from_uint(len(bone.children) > 0 if bone else 0, 2)
    # Next sibling
    sibling_id = 1 if bone_id < num_bones - 1 else 0
    if bone:
        later_siblings = [b for b in bones[bone_id + 1:] if b.parent == bone.parent]
        sibling_id = bones.find(later_siblings[0].name) - bone_id if later_siblings else 0
    bytestr += from_int(sibling_id, 2)
    bytestr += from_uint(0, 2) # padding
    
    # Transform
    transform = bone.matrix_local if bone else Matrix.Identity(4)
    if bone and bone.parent:
        transform = bone.parent.matrix_local.inverted() * transform

    bytestr += from_vec(transform.to_scale(), 4, 12)
    euler = transform.to_euler('XYZ')
    for e in euler:
        bytestr += from_deg(math.degrees(e) / 16) # division by 16 because of format
    bytestr += from_uint(0, 2) # padding
    bytestr += from_vec(transform.to_translation(), 4, 12)

    # Materials and display lists
    indexes = None
    if bone:
        # verts = {i for i,v in enumerate(mesh.vertices) if 
        #         group_names[v.groups[0].group if v.groups else 0] == bone.name}
        # indexes = {p.material_index for p in mesh.polygons if set(p.vertices) & verts}
        indexes = list(range(len(materials))) if bone_id == 0 else []
    else:
        start = sum(len(m.materials) for m in meshes[:bone_id])
        indexes = list(range(start, start + len(materials)))
    if len(indexes) > 32:
        print("You can have at most 32 bones. (This is probably not the limit, " +
                "but exporting more than 32 bones is tricky and not supported right now.)")
    bytestr += from_uint(len(indexes), 4)
    bytestr += from_uint(0, 4) * 2 # placeholder pointers

    ids = lambda: AlignedBytes(from_uint_list(indexes, 1), 1)
    mat_ids = ids()
    dl_ids = ids()
    mat_ptr = BytesPtr(aligned, 0x34, mat_ids, 0, 4)
    dl_ptr =  BytesPtr(aligned, 0x38, dl_ids,  0, 4)

    # Billboard
    bytestr += from_uint(0, 4) # No billboard

    return (aligned, [name_bytes, mat_ids, dl_ids], [name_ptr, mat_ptr, dl_ptr])
    
def add_command(dl_bytestr, command, arg, cmd_offset):
    """ Adds the command to the display list bytestring and returns the new
    command offset. """
    # Parameterless command should not be the last command in a pack for some reason
    if len(arg) == 0 and cmd_offset % 4 == 3:
        cmd_offset = len(dl_bytestr)

    if cmd_offset == len(dl_bytestr):
        dl_bytestr += from_uint(0, 4)

    dl_bytestr[cmd_offset] = command
    dl_bytestr += arg
    cmd_offset += 1
    if cmd_offset % 4 == 0:
        cmd_offset = len(dl_bytestr)
    return cmd_offset

def add_primitive(dl_bytestr, primitive, p_type, transform_ids, \
        tex_size, cmd_offset, prev_vertex):
    """ Adds the primitive to the display list bytestring and returns the new
    command offset and previous vertex. """
    cmd_offset = add_command(dl_bytestr, 0x40, from_uint(p_type, 4), cmd_offset)
    for v in primitive:
        if v.group != prev_vertex.group:
            cmd_offset = add_command(dl_bytestr, 0x14, \
                    from_uint(transform_ids.index(v.group), 4), cmd_offset)

        if v.uv and v.uv != prev_vertex.uv:
            cmd_offset = add_command(dl_bytestr, 0x22, \
                    from_fix(v.uv[0] * tex_size[0], 2, 4) + \
                    from_fix(v.uv[1] * tex_size[1], 2, 4), cmd_offset)

        if v.color and v.color != prev_vertex.color:
            cmd_offset = add_command(dl_bytestr, 0x20, \
                    from_uint(color_to_uint16(v.color, 1), 4), cmd_offset)

        if v.normal and v.normal != prev_vertex.normal:
            cmd_offset = add_command(dl_bytestr, 0x21, \
                    from_vecb(v.normal * 511 / 512, 10, 9, 4), cmd_offset)

        optimized = False
        if prev_vertex.position:
            # Unchanged coordinate
            if any(v.position[i] == prev_vertex.position[i] for i in range(3)):
                optimized = True
                index = [i for i in range(3) if v.position[i] == prev_vertex.position[i]][0]
                swiz = v.position.yz if index == 0 else \
                        v.position.xz if index == 1 else \
                        v.position.xy
                cmd_offset = add_command(dl_bytestr, 0x27 - index,
                        from_vec(swiz, 2, 12), cmd_offset)

            # Coordinates representable without needing a lot of precision
            elif all(fix_to_int(c, 6) * 64 == fix_to_int(c, 12) for c in v.position):
                optimized = True
                cmd_offset = add_command(dl_bytestr, 0x24,
                        from_vecb(v.position, 10, 6, 4), cmd_offset)

            # Not too far from previous vertex
            elif all(-512 <= fix_to_int(c, 12) - fix_to_int(pc, 12) < 512 for c, pc in \
                    zip(v.position, prev_vertex.position)):
                optimized = True
                cmd_offset = add_command(dl_bytestr, 0x28,
                        from_vecb([fix_to_int(c, 12) - fix_to_int(pc, 12) for c, pc in \
                                zip(v.position, prev_vertex.position)], 10, 0, 4), cmd_offset)

        if not optimized:
            cmd_offset = add_command(dl_bytestr, 0x23, \
                    from_vec(v.position, 2, 12) + from_uint(0, 2), cmd_offset)

        prev_vertex = v
    
    cmd_offset = add_command(dl_bytestr, 0x41, b'', cmd_offset)
    return cmd_offset, prev_vertex

def export_display_list(mesh, material, bones, group_names, scale_factor):
    """ Returns (bone_data, other_data, ptrs) where:
    bone_data: AlignedBytes is the bytestring of the bone
    name_data: [AlignedBytes] is a list of relevant detached bytestrings
    ptrs: [BytesPtr] is a list of relevant pointers
    """
    bytestr = bytearray()
    aligned = AlignedBytes(bytestr, 4)

    header_bytestr = bytearray()
    header_aligned = AlignedBytes(header_bytestr, 4)
    bytestr += from_uint(1, 4)
    bytestr += from_uint(0, 4) # pointer
    header_ptr = BytesPtr(aligned, 4, header_aligned, 0, 4)

    # Compile the geometry info
    if any(len(face.vertices) > 4 for face in mesh.polygons):
        raise Exception("A face has too many (more than 4) vertices. " + \
                "All faces should be triangles or quadrilaterals.")

    vertices = []
    faces = []
    counter = 0
    for face in mesh.polygons:
        if material == mesh.materials[face.material_index]:
            for vertex_index in face.vertices:
                vertex = mesh.vertices[vertex_index]
                
                normal = None if material.use_vertex_color_paint else \
                        vertex.normal if face.use_smooth else face.normal
                if normal and bones and len(vertex.groups) > 0:
                    normal = bones[group_names[vertex.groups[0].group \
                            ]].matrix_local.inverted().to_quaternion() * normal
                    
                vertices.append(Vertex(
                            (bones[group_names[vertex.groups[0].group]].matrix_local.inverted() \
                                    * vertex.co if bones and len(vertex.groups) > 0 else \
                            vertex.co) / 2 ** scale_factor, \
                        normal,
                        mesh.uv_layers[0].data[counter].uv \
                                if material.texture_slots[0] and mesh.uv_layers else None,
                        mesh.vertex_colors[0].data[counter].color \
                                if material.use_vertex_color_paint and mesh.vertex_colors \
                                else None,
                        bones.find(group_names[vertex.groups[0].group]) \
                                if bones and len(vertex.groups) > 0 else 0))
                counter += 1
            faces.append(Face(vertices[-len(face.vertices):]))

        else:
            counter += len(face.vertices)
            
    # Transform ID list
    transform_ids = list({v.group for v in vertices})
    header_bytestr += from_uint(len(transform_ids), 4)
    header_bytestr += from_uint(0, 4) # pointer
    transform_bytestr = AlignedBytes(from_uint_list(transform_ids, 1), 1)
    transform_ptr = BytesPtr(header_aligned, 4, transform_bytestr, 0, 4)

    # Display list
    dl_bytestr = bytearray()
    dl_aligned = AlignedBytes(dl_bytestr, 4)

    geo = Geometry(vertices, faces)
    tri_strips, quad_strips, tris, quads = geo.strip()

    tex_size = material.texture_slots[0].texture.image.size if material.texture_slots[0] \
            else (32, 32)
    cmd_offset = 0
    prev_vertex = Vertex(None, None, None, None, None)
    for p_type, strips in ((2, tri_strips), (3, quad_strips)):
        for strip in strips:
            cmd_offset, prev_vertex = add_primitive(dl_bytestr, strip, p_type,
                    transform_ids, tex_size, cmd_offset, prev_vertex)

    for p_type, sep in ((0, tris), (1, quads)):
        if sep:
            cmd_offset, prev_vertex = add_primitive(dl_bytestr, sep, p_type,
                    transform_ids, tex_size, cmd_offset, prev_vertex)

    # If the last command is parameterless (which it is), then add some 0s.
    dl_bytestr += from_uint(0, 4)

    header_bytestr += from_uint(len(dl_bytestr), 4)
    header_bytestr += from_uint(0, 4) # pointer
    dl_ptr = BytesPtr(header_aligned, 0xc, dl_aligned, 0, 4)

    return (aligned, [header_aligned, transform_bytestr, dl_aligned], \
            [header_ptr, transform_ptr, dl_ptr])

def export_texture(texture):
    """ Returns (tex_header, tex_name_data, tex_data, tex_ptrs,
            pal_header, pal_name_data, pal_data, pal_ptrs)
    """
    tex = Texture.from_bpy_texture(texture)

    tex_bytestr = bytearray()
    tex_aligned = AlignedBytes(tex_bytestr, 4)
    
    # Name
    tex_bytestr += from_uint(0, 4)
    tex_name_bytes = AlignedBytes(str_to_cstr(texture.image.name), 1)
    tex_name_ptr = BytesPtr(tex_aligned, 0, tex_name_bytes, 0, 4)

    # Data
    tex_bytestr += from_uint(0, 4)
    tex_data = AlignedBytes(tex.tex_bytestr, 4)
    tex_ptr = BytesPtr(tex_aligned, 4, tex_data, 0, 4)
    tex_bytestr += from_uint(len(tex.tex_bytestr) * 2 // 3 if \
            tex.type == Texture.COMPRESSED else len(tex.tex_bytestr), 4)

    # Size
    for i in range(2):
        tex_bytestr += from_uint(texture.image.size[i], 2)

    # Texture parameters
    size = [s.bit_length() - 4 for s in texture.image.size]
    tex_param = size[0] << 20 | size[1] << 23 | \
            tex.type << 26 | tex.transparent_color << 29
    tex_bytestr += from_uint(tex_param, 4)
    
    # Palette
    pal_bytestr = bytearray()
    pal_aligned = AlignedBytes(pal_bytestr, 4)
    
    # Palette name
    pal_bytestr += from_uint(0, 4)
    pal_name_bytes = AlignedBytes(str_to_cstr(texture.image.name + "_pl"), 1)
    pal_name_ptr = BytesPtr(pal_aligned, 0, pal_name_bytes, 0, 4)

    # Palette data
    pal_bytestr += from_uint(0, 4)
    pal_data = AlignedBytes(tex.pal_bytestr, 4)
    pal_ptr = BytesPtr(pal_aligned, 4, pal_data, 0, 4)
    pal_bytestr += from_uint((len(tex.pal_bytestr) + 3) // 4 * 4, 4)
    pal_bytestr += from_int(-1, 4)

    return (tex_aligned, tex_name_bytes, tex_data, [tex_name_ptr, tex_ptr],
            pal_aligned, pal_name_bytes, pal_data, [pal_name_ptr, pal_ptr])

def export_material(material, ref_textures):
    """ Returns (material_data, name_data, name_ptr) where:
    material_data: AlignedBytes is the bytestring of the material
    name_data: AlignedBytes is the bytestring of the material name
    ptrs: BytesPtr is the pointer to the material name
    Also writes this material's textures to ref_textures if they'ren't there already.
    """
    bytestr = bytearray()
    aligned = AlignedBytes(bytestr, 4)

    # Name
    name_bytes = AlignedBytes(str_to_cstr(material.name), 1)
    bytestr += from_uint(0, 4)
    name_ptr = BytesPtr(aligned, 0, name_bytes, 0, 4)

    # Texture and palette index
    ref_textures += [slot.texture for slot in material.texture_slots \
            if slot and slot.texture.image not in [tex.image for tex in ref_textures]]
    tex_id = [tex.image for tex in ref_textures].index(material.texture_slots[0].texture.image) \
            if material.texture_slots[0] else -1
    bytestr += from_int(tex_id, 4) * 2

    # Texture transformation (identity)
    bytestr += from_fix(1, 4, 12) * 2
    bytestr += from_deg(0)
    bytestr += from_int(0, 2) # padding
    bytestr += from_fix(0, 4, 12) * 2

    # Texture parameters (repeat, flip, transform source)
    tex_param = 0
    if material.texture_slots[0]:
        tex = material.texture_slots[0].texture
        repeat = tex.extension != "EXTEND"
        flip_x = repeat and tex.use_mirror_x
        flip_y = repeat and tex.use_mirror_y
        tex_param = repeat * 0b11 << 16 | flip_x << 18 | flip_y << 19

    tex_transform = 2 if material.get("Environment Map") else 0
    tex_param |= tex_transform << 30
    bytestr += from_uint(tex_param, 4)

    # Polygon parameters
    lights = 0 if material.use_vertex_color_paint else 1
    poly_mode = 1 if material.texture_slots[0] and material.texture_slots[0].blend_type == "MIX"\
            else 0
    render_back = not material.game_settings.use_backface_culling
    render_front = True
    translucent_depth_write = False
    render_far_intersecting = True
    render_1_dot = False
    depth_equal = bool(material.get("Depth Equal"))
    fog = True
    alpha = int_round_mid_up(material.alpha * 31)
    poly_id = material.get("Polygon ID", 0)
    poly_param = lights << 0 | \
            poly_mode << 4 | \
            render_back << 6 | \
            render_front << 7 | \
            translucent_depth_write << 11 | \
            render_far_intersecting << 12 | \
            render_1_dot << 13 | \
            depth_equal << 14 | \
            fog << 15 | \
            alpha << 16 | \
            poly_id << 24
    bytestr += from_uint(poly_param, 4)

    # Diffuse & ambient
    diffuse = color_to_uint16(material.diffuse_color, 2.2, material.diffuse_intensity)
    set_color = not material.use_vertex_color_paint
    ambient = color_to_uint16(material.diffuse_color, 2.2, material.ambient)
    bytestr += from_uint(diffuse | set_color << 15 | ambient << 16, 4)

    # Specular & emission
    specular = color_to_uint16(material.specular_color, 2.2, material.specular_intensity)
    shininess = False
    emission = color_to_uint16(Color((1, 1, 1)), 2.2, material.emit)
    bytestr += from_uint(specular | shininess << 15 | emission << 16, 4)

    return aligned, name_bytes, name_ptr

def get_group_names(mesh):
    return [g.name for g in [obj.vertex_groups for obj in bpy.data.objects \
            if obj.data == mesh][0]]

def save(context, filepath):
    meshes = sorted((obj.data for obj in context.selected_objects if obj.type == "MESH"),
            key=lambda mesh: mesh.name)
    rigs = [obj.data for obj in context.selected_objects if obj.type == "ARMATURE"]
    bones = rigs[0].bones if rigs else []

    if (bones and (len(meshes) != 1 or len(rigs) != 1)) or \
            (not bones and len(meshes) > 8):
        raise Exception("Select either exactly 1 mesh and 1 armature or up to 8 meshes.")

    bytestr_list = BytesWithPtrs()

    max_coord = max(max(max(abs(c) for c in \
                (bones[get_group_names(m)[v.groups[0].group]].matrix_local.inverted() * v.co \
                    if bones and len(v.groups) > 0 else v.co)) \
            for v in m.vertices) \
            for m in meshes)
    scale_factor = max(int(math.log2(max_coord)) - 2, 0)
    # Values really close to 8 can still round to 8, so account for that
    max_coord = int_round_mid_up(max_coord / 2 ** scale_factor)
    if max_coord >= 8:
        scale_factor += 1

    if scale_factor > 13: # What's the actual limit?
        raise Exception("Your model is way too big.")

    bone_data = [export_bone(b, bones, m.materials, m, meshes, get_group_names(m)) \
            for m in meshes for b in (bones if bones else [None])]
    textures = []
    displist_data = [export_display_list(m, mat, bones, get_group_names(m), scale_factor) 
            for m in meshes for mat in m.materials]
    material_data = [export_material(mat, textures) \
            for m in meshes for mat in m.materials]
    texture_data = [export_texture(tex) for tex in textures]

    header = bytearray()
    header_aligned = AlignedBytes(header, 4)

    # Scale
    header += from_uint(scale_factor, 4)

    # Quantities and offsets
    header += from_uint(len(bone_data), 4)
    header += from_uint(0, 4) # pointer
    bone_marker = AlignedBytes(b'', 4)
    bone_ptr = BytesPtr(header_aligned, 0x8, bone_marker, 0, 4)

    header += from_uint(len(displist_data), 4)
    header += from_uint(0, 4) # pointer
    displist_marker = AlignedBytes(b'', 4)
    displist_ptr = BytesPtr(header_aligned, 0x10, displist_marker, 0, 4)

    header += from_uint(len(texture_data), 4)
    header += from_uint(0, 4) # pointer
    texture_marker = AlignedBytes(b'', 4)
    texture_ptr = BytesPtr(header_aligned, 0x18, texture_marker, 0, 4)

    header += from_uint(len(texture_data), 4)
    header += from_uint(0, 4) # pointer
    palette_marker = AlignedBytes(b'', 4)
    palette_ptr = BytesPtr(header_aligned, 0x20, palette_marker, 0, 4)

    header += from_uint(len(material_data), 4)
    header += from_uint(0, 4) # pointer
    material_marker = AlignedBytes(b'', 4)
    material_ptr = BytesPtr(header_aligned, 0x28, material_marker, 0, 4)

    # Transform-bone map
    header += from_uint(0, 4) # pointer
    tb_bytestr = AlignedBytes(from_uint_list(range(len(bone_data)), 2), 2)
    tb_ptr = BytesPtr(header_aligned, 0x2c, tb_bytestr, 0, 4)

    # Texture and palette data block
    header += from_uint(0, 4) * 2 # unknown stuff
    header += from_uint(0, 4) # pointer
    tex_data_marker = AlignedBytes(b'', 4)
    tex_data_ptr = BytesPtr(header_aligned, 0x38, tex_data_marker, 0, 4)

    bytestr_list.bytestrs.append(header_aligned)
    bytestr_list.bytestrs.append(bone_marker)
    bytestr_list.bytestrs += [v[0] for v in bone_data]
    bytestr_list.bytestrs += list(chain(*(v[1] for v in bone_data)))
    bytestr_list.bytestrs.append(tb_bytestr)
    bytestr_list.bytestrs.append(displist_marker)
    bytestr_list.bytestrs += [v[0] for v in displist_data]
    bytestr_list.bytestrs += list(chain(*(v[1] for v in displist_data)))
    bytestr_list.bytestrs.append(texture_marker)
    bytestr_list.bytestrs += [v[0] for v in texture_data]
    bytestr_list.bytestrs += [v[1] for v in texture_data]
    bytestr_list.bytestrs.append(palette_marker)
    bytestr_list.bytestrs += [v[4] for v in texture_data]
    bytestr_list.bytestrs += [v[5] for v in texture_data]
    bytestr_list.bytestrs.append(material_marker)
    bytestr_list.bytestrs += [v[0] for v in material_data]
    bytestr_list.bytestrs += [v[1] for v in material_data]
    # Texture data must come last
    bytestr_list.bytestrs.append(tex_data_marker)
    bytestr_list.bytestrs += [v[2] for v in texture_data]
    bytestr_list.bytestrs += [v[6] for v in texture_data]

    bytestr_list.ptrs += [bone_ptr, tb_ptr, displist_ptr, texture_ptr, palette_ptr,
            material_ptr, tex_data_ptr]
    bytestr_list.ptrs += list(chain(*(v[2] for v in bone_data)))
    bytestr_list.ptrs += list(chain(*(v[2] for v in displist_data)))
    bytestr_list.ptrs += list(chain(*(v[3] for v in texture_data)))
    bytestr_list.ptrs += list(chain(*(v[7] for v in texture_data)))
    bytestr_list.ptrs += [v[2] for v in material_data]

    full_bytestr = bytestr_list.assemble()
    with open(filepath, "wb") as f:
        f.write(full_bytestr)
        
    return {"FINISHED"}
