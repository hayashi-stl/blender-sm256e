from mathutils import Color, Vector
from functools import reduce

def int_round_mid_up(num):
    return int((num + 0.5) // 1)

def get_n_bytes(bytestr, offset, len):
    return bytestr[offset : offset + len]

def to_int(bytestr, offset, num_bytes):
    return int.from_bytes(bytestr[offset : offset+num_bytes], byteorder="little", signed=True)
def to_uint(bytestr, offset, num_bytes):
    return int.from_bytes(bytestr[offset : offset+num_bytes], byteorder="little", signed=False)

def sign_int10(integer):
    return integer - 0x400 if integer >= 0x200 else integer
def to_color(bytestr, offset, gamma):
    vec = to_uint(bytestr, offset, 2)
    return Color(Vector((vec & 0x1f, vec >> 5 & 0x1f, vec >> 10 & 0x1f)) * 1.0 / 0x1f)
def to_uvec30(bytestr, offset):
    vec = to_uint(bytestr, offset, 4)
    return Vector((vec & 0x3ff, vec >> 10 & 0x3ff, vec >> 20 & 0x3ff))
def to_vec30(bytestr, offset):
    vec = to_uint(bytestr, offset, 4)
    return Vector((sign_int10(vec & 0x3ff), sign_int10(vec >> 10 & 0x3ff), sign_int10(vec >> 20 & 0x3ff)))
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

    def __eq__(self, other):
        if not isinstance(other, Vertex):
            return False
        return self.position == other.position and \
                self.normal == other.normal and \
                self.uv == other.uv and \
                self.color == other.color and \
                self.group == other.group

    def __hash__(self):
        return hash((self.position, self.normal, self.uv, self.color, self.group))

class Face:
    def __init__(self, vertices):
        """
        vertices: [Vertex]
        """
        self.vertices = vertices[:]

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
    def __init__(self, vertices, faces):
        """
        vertices: [Vertex]
        faces: [Face]
        face_graph: [{int}] says which faces are connected to which faces by index
        """
        self.vertices = vertices[:]
        self.faces = faces[:]

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
        reduction = {}

        while len(reduced) > new_num:
            pair = sorted(((c0, c1) for c0 in reduced for c1 in reduced if c0 != c1),
                    key=lambda cp: sum((a - b) ** 2 for a, b in zip(*cp)))[0]
            new_color = tuple((a + b) // 2 for a, b in zip(*pair))

            for i in range(2):
                reduced.remove(pair[i])
            reduced.add(new_color)

            for i in range(2):
                if pair[i] != new_color:
                    reduction[pair[i]] = new_color

        new_colors = colors[:]
        for i in range(len(new_colors)):
            while new_colors[i] in reduction:
                new_colors[i] = reduction[new_colors[i]]
                
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
        # TODO
        pass

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
        tex = Texture()
        tex.texture = texture
        if any(s != 2 ** (s.bit_length() - 1) or s < 8 or s > 1024 for s in texture.image.size):
            raise Exception("Texture dimensions must be powers of 2 between 8 and 1024.")

        tex.calc_bytestr()
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
