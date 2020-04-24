import bpy
import bmesh
from mathutils import Color, Vector
from functools import reduce
from itertools import permutations, takewhile, chain

def int_round_mid_up(num):
    return int((num + 0.5) // 1)

def get_n_bytes(bytestr, offset, len_):
    return bytestr[offset : offset + len_]

def to_int(bytestr, offset, num_bytes):
    return int.from_bytes(bytestr[offset : offset+num_bytes], byteorder="little", signed=True)

def to_uint(bytestr, offset, num_bytes):
    return int.from_bytes(bytestr[offset : offset+num_bytes], byteorder="little", signed=False)

def int_to_fix(integer, bit_precision):
    return integer / (1 << bit_precision)

def to_fix(bytestr, offset, num_bytes, bit_precision):
    return int_to_fix(to_int(bytestr, offset, num_bytes), bit_precision)

def to_deg(bytestr, offset):
    return to_int(bytestr, offset, 2) * 360 / 65536

def to_uint_list(bytestr, offset, bytes_per_elem, num_elems):
    return [to_uint(bytestr, offset + bytes_per_elem * i, bytes_per_elem) 
            for i in range(num_elems)]

def to_vec(bytestr, offset, bytes_per_elem, num_elems, bit_precision):
    return [to_fix(bytestr, offset + bytes_per_elem * i, bytes_per_elem, bit_precision)
            for i in range(num_elems)]

def sign_int(integer, num_bits):
    return (integer + 2 ** (num_bits - 1)) % 2 ** num_bits - 2 ** (num_bits - 1)

def to_vecb(bytestr, offset, bits_per_elem, num_elems, bit_precision):
    long_int = to_uint(bytestr, offset, (bits_per_elem * num_elems + 7) // 8)
    return [int_to_fix(sign_int((long_int >> bits_per_elem * i) % (1 << bits_per_elem),
        bits_per_elem), bit_precision) for i in range(num_elems)]

def uint16_to_rgb555(integer):
    return [integer & 0x1f, integer >> 5 & 0x1f, integer >> 10 & 0x1f]

def uint16_to_color(integer, gamma, scale=1):
    return Color([(c / 31 / scale) ** gamma for c in uint16_to_rgb555(integer)]) \
            if scale != 0 else Color((0, 0, 0))

def cstr_to_str(bytestr, offset):
    end = offset
    while bytestr[end] != 0x00:
        end += 1
    return bytestr[offset : end].decode("ascii")

def from_int(integer, num_bytes):
    return integer.to_bytes(num_bytes, byteorder="little", signed=True)

def from_uint(integer, num_bytes):
    return integer.to_bytes(num_bytes, byteorder="little", signed=False)

def fix_to_int(number, bit_precision):
    return int_round_mid_up(number * (1 << bit_precision))

def from_fix(number, num_bytes, bit_precision):
    return from_int(fix_to_int(number, bit_precision), num_bytes)

def from_deg(degrees):
    return from_int(int_round_mid_up(degrees * 65536 / 360), 2)

def from_uint_list(list_, bytes_per_elem):
    return b''.join(from_uint(e, bytes_per_elem) for e in list_)

def from_vec(vec, bytes_per_elem, bit_precision):
    return b''.join(from_fix(v, bytes_per_elem, bit_precision) for v in vec)

def from_vecb(vec, bits_per_elem, bit_precision, num_bytes):
    num = sum(int_round_mid_up(v * (1 << bit_precision)) % (1 << bits_per_elem) \
            << bits_per_elem * i for i, v in enumerate(vec))
    return from_uint(num, num_bytes)

def rgb555_to_uint16(vec):
    return vec[0] | vec[1] << 5 | vec[2] << 10

def color_to_uint16(color, gamma, scale=1):
    vec = [int_round_mid_up(scale * c ** (1 / gamma) * 31) for c in color]
    return rgb555_to_uint16(vec)

def str_to_cstr(string):
    return string.encode("ascii") + b'\0'

class Vertex:
    def __init__(self, position, normal, uv, color, group):
        """
        position: Vector
        normal: Vector
        uv: Vector
        color: Color
        group: int
        """
        self.position = Vector(position).freeze() if position else None
        self.normal = Vector(normal).freeze() if normal else None
        self.uv = Vector(uv).freeze() if uv else None
        self.color = Color(color).freeze() if color else None
        self.group = group

    def rep(self):
        return (self.position, self.normal, self.uv, self.color, self.group)

    def from_rep(rep):
        return Vertex(*rep)

class Face:
    def __init__(self, vertices, material_id = None):
        """
        vertices: [Vertex]
        material_id: int | None
        """
        self.vertices = vertices[:]
        self.material_id = material_id

    def can_connect_to(self, other):
        """ Whether this face can be connected to the other face when making
        a tri/quad strip """
        if self is other or len(self.vertices) != len(other.vertices):
            return False

        shared = list(set(self.vertices) & set(other.vertices))
        if len(shared) < 2:
            return False

        diffs = [(vs.index(shared[0]) - vs.index(shared[1])) % len(vs) \
                for vs in (self.vertices, other.vertices)]
        return (len(self.vertices) == 3 or 2 not in diffs) and diffs[0] != diffs[1]

class Geometry:
    def __init__(self, vertices, faces, compute_face_graph = True):
        """
        vertices: [Vertex]
        faces: [Face]
        face_graph: [{int}] says which faces are connected to which faces by index
        """
        self.vertices = vertices[:]
        self.faces = faces[:]

        if compute_face_graph:
            # Vertices with equal reps are equivalent, so make them identical
            by_rep = {v.rep(): v for v in self.vertices}
            self.vertices = list(by_rep.values())

            for face in self.faces:
                face.vertices = [by_rep[v.rep()] for v in face.vertices]

            self.face_graph = [{j for j, other in enumerate(self.faces) \
                if face.can_connect_to(other)} for face in self.faces]

    def strip(self):
        """ Returns (tri_strips, quad_strips, tris, quads) where:
        tri_strips: [[Vertex]] contains triangle strips as sequences of vertices
        quad_strips: [[Vertex]] contains quad strips as sequences of vertices
        tris: [Vertex] contains separate triangles as a sequence of vertices
        quads: [Vertex] contains separate quads as a sequence of vertices
        """
        # Construct modified face graph based on where connections are
        # so that turning can be detected
        # graph = [[None for _ in face.vertices] for face in self.faces]
        # for i, face in enumerate(self.faces):
        #     for j in self.face_graph[i]:
        #         indexes = [face.vertices.index(v) for v in \
        #                 set(face.vertices) & set(self.faces[j].vertices)][:2]
        #         graph[i][min(indexes[0], indexes[1]) \
        #                 if abs(indexes[0] - indexes[1]) == 1 \
        #                 else max(indexes[0], indexes[1])] = j
        #         
        # for i,l in enumerate(graph):
        #     print(i,l)
        
        def adjacent_by_edge(face_index, edge):
            face = self.faces[face_index]
            for j in self.face_graph[face_index]:
                other = self.faces[j]
                if edge[0] in other.vertices and edge[1] in other.vertices:
                    indexes = [other.vertices.index(e) for e in edge]
                    return (j, indexes[0] \
                            if (indexes[0] + 1) % len(other.vertices) == indexes[1] \
                            else indexes[1])
            return (None, 0)
            
        def extend_strip(face_index, order, faces_left):
            face = self.faces[face_index]
            vertices = [face.vertices[e] for e in order]
            result_indexes = {face_index}
            
            # Extend forwards
            next_index, v_index = adjacent_by_edge(face_index, vertices[-2:])
            while next_index is not None and next_index in faces_left and \
                    next_index not in result_indexes:
                result_indexes.add(next_index)
                vertices += list(reversed([self.faces[next_index].vertices[\
                        (v_index + i) % len(face.vertices)] \
                        for i in range(2, len(face.vertices))]))
                next_index, v_index = adjacent_by_edge(next_index, vertices[-2:])

            # Extend backwards
            next_index, v_index = adjacent_by_edge(face_index, vertices[:2])
            num_exts = 0
            last_index = None
            while next_index is not None and next_index in faces_left and \
                    next_index not in result_indexes:
                result_indexes.add(next_index)
                vertices = [self.faces[next_index].vertices[\
                        (v_index + i) % len(face.vertices)] \
                        for i in range(2, len(face.vertices))] + vertices
                num_exts += 1
                last_index = next_index
                next_index, v_index = adjacent_by_edge(next_index, vertices[:2])

            # Backwards extension must be by an even amount of triangles!
            if num_exts % 2 != 0 and len(face.vertices) == 3:
                result_indexes.remove(last_index)
                vertices = vertices[1:]

            return vertices, result_indexes

        unstripped = set(range(len(self.faces)))
        tri_strips = []
        quad_strips = []
        tris = []
        quads = []
        while unstripped:
            i = next(iter(unstripped))

            face = self.faces[i]
            orders = [[0,1,3,2], [1,2,0,3]] if len(face.vertices) == 4 else \
                    [[0,1,2], [1,2,0], [2,0,1]]

            strip, face_indexes = sorted((extend_strip(i, order, unstripped) \
                    for order in orders),
                    key=lambda s: len(s[0]))[-1]

            unstripped -= face_indexes
            if len(face_indexes) > 1:
                (quad_strips if len(face.vertices) == 4 else tri_strips).append(strip)
            else:
                if len(face.vertices) == 4:
                    quads += [strip[0], strip[1], strip[3], strip[2]]
                else:
                    tris += strip

        return (tri_strips, quad_strips, tris, quads)

    def create_mesh(self, context, name, skeleton, scale, *, create_obj=True):
        get_equiv = lambda v: (v.position, v.normal, v.group)

        equiv = {}
        for v in self.vertices:
            equiv.setdefault(get_equiv(v), []).append(v)

        mesh = bpy.data.meshes.new(name)
        obj = None
        if create_obj:
            obj = bpy.data.objects.new(name, mesh)
            
            context.scene.collection.objects.link(obj)
            context.view_layer.objects.active = obj
            obj.select_set(True)

        equiv_vertices = [v for v in self.vertices if equiv[get_equiv(v)][0] == v]
        vertices = [skeleton.bones[v.group].abs_transform @ (scale * v.position) 
                for v in equiv_vertices]
        faces = [[equiv_vertices.index(equiv[get_equiv(v)][0]) for v in f.vertices]
                for f in self.faces]

        # Billboard hack
        for v in self.vertices:
            v.billboard = skeleton.bones[v.group].billboard
        
        mesh.from_pydata(vertices, [], faces)

        for face in mesh.polygons:
            face.use_smooth = True

        # UV
        if any(v.uv for v in self.vertices):
            mesh.uv_layers.new(name="UVMap")
            uvs = [v.uv if v.uv else Vector((0, 0)) 
                    for f in self.faces for v in f.vertices]

            for i, data in enumerate(mesh.uv_layers[0].data):
                data.uv = uvs[i]

        # Vertex colors
        if any(v.color for v in self.vertices):
            mesh.vertex_colors.new(name="Col")
            colors = [v.color if v.color else Color((1, 1, 1))
                    for f in self.faces for v in f.vertices]

            for i, data in enumerate(mesh.vertex_colors[0].data):
                data.color = tuple(colors[i]) + (1,)

        return obj, mesh


class Bone:
    def __init__(self, name, parent_id, sibling_id, rel_transform, material_ids, displist_ids,
            billboard):
        """
        name: string
        parent_id: int (-1 means no parent),
        sibling_id: int (-1 means last sibling),
        rel_transform: Matrix (a 4x4 matrix),
        material_ids: {int}
        displist_ids: {int}
        billboard: bool
        """
        self.name = name
        self.parent_id = parent_id
        self.sibling_id = sibling_id
        self.rel_transform = rel_transform
        self.material_ids = material_ids
        self.displist_ids = displist_ids
        self.billboard = billboard

    def attach_to(self, skeleton):
        self.parent = skeleton.bones[self.parent_id] if self.parent_id >= 0 else None
        self.abs_transform = self.parent.abs_transform @ self.rel_transform \
                if self.parent else self.rel_transform

    def update_parent_lists(self, skeleton):
        self.sibling = skeleton.bones[self.sibling_id] if self.sibling_id >= 0 else None
        if self.parent:
            self.parent.material_ids |= self.material_ids
            self.parent.displist_ids |= self.displist_ids

class Skeleton:
    def __init__(self, bones):
        """
        bones: [Bone]
        """
        self.bones = bones
        for bone in bones:
            bone.attach_to(self)
        for bone in reversed(bones):
            bone.update_parent_lists(self)

class Texel4x4:
    def __init__(self, colors):
        """
        colors: {color} RGBA5551
        """
        self.colors, self.palette = Texture.reduce_colors(\
                [c if c[3] != 0 else (0, 0, 0, 0) for c in colors], 4)
        self.transparency = any(c[3] == 0 for c in self.colors)
        self.palette_set = {c for c in self.palette if c[3] != 0}
        self.interp = False

        for perm in permutations(self.palette_set):
            if len(perm) == 3 and \
                    all(c2 == (c0 + c1) // 2 for c0, c1, c2 in zip(*perm)):
                self.palette_set = {perm[0], perm[1]}
                self.interp = True
                self.transparency = True # consequence of midpoint interpolation

            if len(perm) == 4 and \
                    all(c2 == (c0 * 5 + c1 * 3) // 8 and \
                        c3 == (c0 * 3 + c1 * 5) // 8 for c0, c1, c2, c3 in zip(*perm)):
                self.palette_set = {perm[0], perm[1]}
                self.interp = True
                assert(not self.transparency)

        self.cmap = (self.palette_set, 
                set(range(2 if len(self.palette_set) < 2 or self.interp else \
                    3 if self.transparency else 4)))

        # Order: most constrained to least constrained
        self.cmap_order = 0 if len(self.palette_set) == 4 else \
                1 if len(self.palette_set) == 3 and self.transparency else \
                2 if len(self.palette_set) == 3 else \
                3 if len(self.palette_set) == 2 and self.interp else \
                4 if len(self.palette_set) == 2 and self.transparency else \
                5 if len(self.palette_set) == 2 else \
                6 if len(self.palette_set) == 1 else \
                7

    def add_to_color_map(self, cmap_arr):
        self.index = Texel4x4.add_color_map(cmap_arr, self.cmap)
    
    def get_bytestrs(self, palette):
        """ Returns (tex_bytestr, pal_index_bytestr) """
        self.palette = palette[self.index : self.index + 4]
        self.palette += [None for _ in range(len(self.palette), 4)]
        
        if self.transparency:
            self.palette[3] = (0, 0, 0, 0)

        if self.interp:
            if self.transparency:
                self.palette[2] = tuple((c0 + c1) // 2 for c0, c1 in zip(*self.palette[0:2]))
            else:
                self.palette[2] = tuple((c0 * 5 + c1 * 3) // 8 for c0, c1 in \
                        zip(*self.palette[0:2]))
                self.palette[3] = tuple((c0 * 3 + c1 * 5) // 8 for c0, c1 in \
                        zip(*self.palette[0:2]))

        indexes = Texture.get_indexes(self.colors, self.palette)
        tex_bytestr = from_uint(sum(idx << (2 * i) for i, idx in enumerate(indexes)), 4)

        pal_index = self.index // 2 | \
                self.interp << 14 | \
                (not self.transparency) << 15
        pal_index_bytestr = from_uint(pal_index, 2)
        return tex_bytestr, pal_index_bytestr

    def get_color_map_partition(cmap_arr, begin, end):
        """ Returns [({color}, {int}, {int})] """
        partition = []
        while begin < end:
            cmap = cmap_arr[begin]
            partition.append((*cmap, {n for n in cmap[1] if begin <= n < end}))
            begin = 1 + max(partition[-1][2])
        return partition

    def try_add_color_map_at(cmap_arr, new_cmap, i):
        partition = Texel4x4.get_color_map_partition(cmap_arr,
                i + min(new_cmap[1]), i + max(new_cmap[1]) + 1)

        for perm in permutations(list(new_cmap[0]) + \
                [None] * (len(new_cmap[1]) - len(new_cmap[0]))):
            # Partition the permutation
            part_perm = []
            part_start = 0
            for _, _, p in partition:
                part_perm.append(set(perm[part_start : part_start + len(p)]) - \
                        {None})
                part_start += len(p)

            # Attempt to add the permutation
            if all(len(new_cols - cols) <= len(idxs) - len(cols) \
                    for (cols, idxs, _), new_cols in zip(partition, part_perm)):
                # Add the permutation, taking advantage of aliasing
                for (cols, idxs, s_idxs), new_cols in zip(partition, part_perm):
                    idxs -= s_idxs
                    cols -= new_cols
                    # Oops, may have too many colors left over
                    dragged = set(list(cols)[:max(0, len(cols) - len(idxs))])
                    cols -= dragged
                    new_cols_ = new_cols | dragged

                    for idx in s_idxs:
                        cmap_arr[idx] = (new_cols_, s_idxs)

                return True

    def add_color_map(cmap_arr, new_cmap):
        """
        cmap_arr: [({color}, {int})] is an array of mappings from colors to indexes.
            The indexes are the same as the indexes of the array.
        new_cmap: ({color}, {int}) is the new mapping from colors to indexes to add.
            These indexes are relative to some multiple-of-2 index in the color map array.
        Color maps are allowed to have more indexes than colors to signify that an
            open slot exists. Also, indexes are assumed to be consecutive.
        """
        for i in range(0, len(cmap_arr), 2):
            if Texel4x4.try_add_color_map_at(cmap_arr, new_cmap, i):
                return i

    def convert_to_palette(cmap_arr):
        palette = []

        for i in range(len(cmap_arr)): # List will be modified and iterated over at the same time
            cols, idxs = cmap_arr[i]
            if cols:
                palette.append(next(iter(cols)))
                Texel4x4.try_add_color_map_at(cmap_arr, ({palette[-1]}, {0}), i)
            else:
                palette.append(None)

        num_nones = len(list(takewhile(lambda c: c is None, reversed(palette))))
        palette = [(0, 0, 0, 31) if c is None else c for c in palette[:-num_nones]]
        return palette

class Texture:
    A3I5 = 1
    COLOR_4 = 2
    COLOR_16 = 3
    COLOR_256 = 4
    COMPRESSED = 5
    A5I3 = 6
    COLOR_DIRECT = 7

    def calc_rgba5555(self):
        self.rgba5555 = [tuple(int_round_mid_up(c * 31) for c in \
                self.texture.image.pixels[4 * i : 4 * i + 4]) for i in \
                range(self.texture.image.size[0] * self.texture.image.size[1])]
        
    def calc_type(self):
        self.calc_rgba5555()
        self.transparent_color = False

        num_colors = len({c[0:3] for c in self.rgba5555 if c[3] != 0})
        # Translucency
        if any(c[3] not in (0, 31) for c in self.rgba5555):
            self.type = Texture.A3I5 if num_colors > 8 else Texture.A5I3

        else:
            if any(c[3] == 0 for c in self.rgba5555):
                num_colors += 1
                self.transparent_color = True

            if num_colors <= 4:
                self.type = Texture.COLOR_4 # Less space than a compressed texture

            else:
                self.type = Texture.COMPRESSED if not self.texture.get("Uncompressed") else \
                        Texture.COLOR_16 if num_colors <= 16 else \
                        Texture.COLOR_256 if num_colors <= 256 else \
                        Texture.COLOR_DIRECT

    def get_indexes(colors, palette):
        return [palette.index(c) for c in colors]

    def reduce_colors(colors, new_num):
        reduced = set(colors)
        new_colors = colors[:]

        while len(reduced) > new_num:
            pair = sorted(((c0, c1) for c0 in reduced for c1 in reduced if c0 != c1),
                    # Don't merge a transparent pixel with an opaque one regardless of distance
                    key=lambda cp: (len(cp[0]) == 4 and (cp[0][3] == 0) != (cp[1][3] == 0),
                        sum((a - b) ** 2 for a, b in zip(*cp))))[0]
            # Keep more common color
            new_color = max(pair, key=lambda c: colors.count(c))

            for i in range(2):
                reduced.remove(pair[i])
            reduced.add(new_color)

            for i in range(len(new_colors)):
                if new_colors[i] in pair:
                    new_colors[i] = new_color
                
        return new_colors, list(reduced)

    def calc_bytestr_alpha(self, alpha_bits):
        palette = list({c[0:3] for c in self.rgba5555 if c[3] != 0})
        if not palette:
            palette.append((0, 0, 0))
        colors, palette = Texture.reduce_colors(\
                [c[0:3] if c[3] != 0 else palette[0] for c in self.rgba5555], 
                2 ** (8 - alpha_bits))
        indexes = Texture.get_indexes(colors, palette)

        self.tex_bytestr = from_uint_list([idx | \
                int_round_mid_up(c[3] * (2 ** alpha_bits - 1) / 31) << (8 - alpha_bits) \
                for idx, c in zip(indexes, self.rgba5555)], 1)

        self.pal_bytestr = from_uint_list(map(rgb555_to_uint16, palette), 2)

    def calc_bytestr_ncol(self, index_bits):
        colors, palette = Texture.reduce_colors(\
                [c if c[3] != 0 else (0, 0, 0, 0) for c in self.rgba5555], 2 ** index_bits)
        palette.sort(key=lambda c: c[3]) # Transparent color is first color if exists
        indexes = Texture.get_indexes(colors, palette)

        stride = 8 // index_bits
        self.tex_bytestr = from_uint_list([sum(idx * 2 ** (j * index_bits) \
                for j, idx in enumerate(indexes[stride * i : stride * (i + 1)])) \
                for i in range(len(indexes) // stride)], 1)

        self.pal_bytestr = from_uint_list([rgb555_to_uint16(c[0:3]) for c in palette], 2)

    def calc_bytestr_direct(self):
        self.tex_bytestr = from_uint_list([rgb555_to_uint16(c[0:3]) | (c[3] != 0) << 15 \
                for c in self.rgba5555], 2)

        self.pal_bytestr = b''

    def calc_bytestr_compressed(self):
        qwidth = self.width // 4
        qheight = self.height // 4
        texels = [Texel4x4(chain.from_iterable(
            self.rgba5555[self.width * (4 * (i // qwidth) + j) + 4 * (i % qwidth) :
                          self.width * (4 * (i // qwidth) + j) + 4 * (i % qwidth + 1)] 
            for j in range(4))) for i in range(qwidth * qheight)]

        init_colors = set()
        init_indexes = set(range(4 * len(texels)))
        cmap_arr = [(init_colors, init_indexes) for _ in range(4 * len(texels))]
        for texel in sorted(texels, key=lambda t: t.cmap_order):
            texel.add_to_color_map(cmap_arr)

        palette = Texel4x4.convert_to_palette(cmap_arr)
        tex_bytestr = bytearray()
        pal_index_bytestr = bytearray()

        for texel in texels:
            t, p = texel.get_bytestrs(palette)
            tex_bytestr += t
            pal_index_bytestr += p

        self.tex_bytestr = tex_bytestr + pal_index_bytestr
        self.pal_bytestr = from_uint_list([rgb555_to_uint16(c[0:3]) for c in palette], 2)

    def calc_bytestr(self):
        self.calc_type()
        if self.type in (Texture.A3I5, Texture.A5I3):
            self.calc_bytestr_alpha(3 if self.type == Texture.A3I5 else 5)

        elif self.type in (Texture.COLOR_4, Texture.COLOR_16, Texture.COLOR_256):
            self.calc_bytestr_ncol(2 if self.type == Texture.COLOR_4 else \
                    4 if self.type == Texture.COLOR_16 else 8)

        elif self.type == Texture.COLOR_DIRECT:
            self.calc_bytestr_direct()

        else:
            self.calc_bytestr_compressed()

    def from_bpy_texture(texture):
        """
        texture: bpy.types.Texture
        """
        print(texture.image.name)
        tex = Texture()
        tex.texture = texture
        if any(s != 2 ** (s.bit_length() - 1) or s < 8 or s > 1024 for s in texture.image.size):
            raise Exception("Texture dimensions must be powers of 2 between 8 and 1024.")

        tex.width = texture.image.size[0]
        tex.height = texture.image.size[1]
        tex.calc_bytestr()
        return tex

    def get_colors(indexes, palette):
        return [palette[index] for index in indexes]

    def calc_rgba5555_alpha(self, num_alpha_bits):
        self.rgba5555 = [self.palette[v % 2 ** (8 - num_alpha_bits)][0:3] +
                [int_round_mid_up((v >> (8 - num_alpha_bits)) * 31 / (2 ** num_alpha_bits - 1))]
                for v in to_uint_list(self.tex_bytestr, 0, 1, self.width * self.height)]

    def calc_rgba5555_ncol(self, index_bits):
        if self.transparency:
            self.palette[0] = [0, 0, 0, 0]

        indexes = [(v >> index_bits * i) % 2 ** index_bits
                for v in to_uint_list(self.tex_bytestr, 0, 1, 
                    self.width * self.height // (8 // index_bits))
                for i in range(8 // index_bits)]
        self.rgba5555 = Texture.get_colors(indexes, self.palette)

    def calc_rgba5555_direct(self):
        self.rgba5555 = [uint16_to_rgb555(c) + [31 * (c >> 15)] for c in
                to_uint_list(self.tex_bytestr, 0, 2, self.width * self.height)]

    def calc_rgba5555_compressed(self):
        self.rgba5555 = [None for _ in range(self.width * self.height)]
        pal_index_offset = self.width * self.height // 4
        self.palette += [[0, 0, 0, 0] for i in range(2)] # in case of optimization

        for y in range(self.height // 4):
            for x in range(self.width // 4):
                texel_int = to_uint(self.tex_bytestr, (y * (self.width // 4) + x) * 4, 4)
                texel = [texel_int >> (2 * i) & 3 for i in range(16)]

                pal_int = to_uint(self.tex_bytestr,
                        pal_index_offset + (y * (self.width // 4) + x) * 2, 2)
                pal_index = (pal_int % 2 ** 14) * 2
                interp = pal_int >> 14 & 1
                transparency = not (pal_int >> 15 & 1)

                colors = self.palette[pal_index:][:2] + [None] * 2
                if transparency:
                    colors[3] = [0, 0, 0, 0]
                    colors[2] = [int_round_mid_up((c0 + c1) // 2) for c0, c1 in
                            zip(colors[0], colors[1])] \
                                    if interp else self.palette[pal_index + 2]

                elif interp:
                    colors[2] = [int_round_mid_up((c0 * 5 + c1 * 3) // 8) for c0, c1 in
                            zip(colors[0], colors[1])] 
                    colors[3] = [int_round_mid_up((c0 * 3 + c1 * 5) // 8) for c0, c1 in
                            zip(colors[0], colors[1])] 

                else:
                    for i in range(2,4):
                        colors[i] = self.palette[pal_index + i]

                texel_colors = Texture.get_colors(texel, colors)
                for i in range(4):
                    offset = (4 * y + i) * self.width + 4 * x
                    self.rgba5555[offset : offset + 4] = texel_colors[4 * i:][:4]
                    

    def calc_bpy_texture(self, name, material):
        if self.pal_bytestr:
            self.palette = [uint16_to_rgb555(c) + [31] for c in 
                    to_uint_list(self.pal_bytestr, 0, 2, len(self.pal_bytestr) // 2)]

        if self.type in (Texture.A3I5, Texture.A5I3): 
            self.calc_rgba5555_alpha(3 if self.type == Texture.A3I5 else 5) 

        elif self.type in (Texture.COLOR_4, Texture.COLOR_16, Texture.COLOR_256):
            self.calc_rgba5555_ncol(2 if self.type == Texture.COLOR_4 else
                    4 if self.type == Texture.COLOR_16 else 8)

        elif self.type == Texture.COLOR_DIRECT:
            self.calc_rgba5555_direct()

        elif self.type == Texture.COMPRESSED:
            self.calc_rgba5555_compressed()

        else:
            raise Exception("Unknown type: " + hex(self.type) + 
                    " in texture: " + name)

        image = bpy.data.images.new(name, self.width, self.height, alpha=True)
        image.colorspace_settings.name = "sRGB"
        image.pixels = [v / 31 for c in self.rgba5555 for i, v in enumerate(c)]
        self.image = image

        if self.type in (Texture.COLOR_16, Texture.COLOR_256, Texture.COLOR_DIRECT):
            material["Uncompressed"] = 1

    def from_bytestr(tex_bytestr, pal_bytestr, name, width, height, type_, transparency,
            material):
        tex = Texture()
        tex.tex_bytestr = tex_bytestr
        tex.pal_bytestr = pal_bytestr
        tex.width = width
        tex.height = height
        tex.type = type_
        tex.transparency = transparency
        tex.calc_bpy_texture(name, material)
        return tex


class AlignedBytes:
    def __init__(self, bytestr, byte_align):
        """
        bytestr: bytearray
        byte_align: int
        """
        self.bytestr = bytestr
        self.byte_align = byte_align

# A way to represent pointers in bytestrings.
class BytesPtr:
    def __init__(self, src_bytestr, src_offset, dest_bytestr, dest_offset, num_bytes):
        """
        src_bytestr: AlignedBytes
        src_offset: int
        dest_bytestr: AlignedBytes
        dest_offset: int
        num_bytes: int
        """
        self.src_bytestr = src_bytestr
        self.src_offset = src_offset
        self.dest_bytestr = dest_bytestr
        self.dest_offset = dest_offset
        self.num_bytes = num_bytes

class BytesWithPtrs:
    def __init__(self):
        """
        byetstrs: [AlignedBytes]
        ptrs: [BytesPtr]
        """
        self.bytestrs = []
        self.ptrs = []
        
    def assemble(self):
        """ Creates a long bytestring out of all the individual bytestrings,
        resolving pointers. """
        indexed_ptrs = [(self.bytestrs.index(ptr.src_bytestr), \
                ptr.src_offset, \
                self.bytestrs.index(ptr.dest_bytestr), \
                ptr.dest_offset, ptr.num_bytes) for ptr in self.ptrs]
        
        placed_bytestrs = []
        position = 0
        for i,bytestr in enumerate(self.bytestrs):
            byte_align = self.bytestrs[i+1].byte_align if i+1 < len(self.bytestrs) else 4
            placed_bytestrs.append((bytestr.bytestr + (byte_align - position - \
                    len(bytestr.bytestr)) \
                    % byte_align * b"\0", position))
            position += len(placed_bytestrs[-1][0])

        for src_index, src_offset, dest_index, dest_offset, num_bytes in indexed_ptrs:
            placed_bytestrs[src_index][0][src_offset : src_offset + num_bytes] = \
                    from_uint(placed_bytestrs[dest_index][1] + dest_offset, num_bytes)

        return reduce((lambda b, pb: b + pb[0]), placed_bytestrs, b"")
