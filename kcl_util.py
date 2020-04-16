from .util import *
from math import log2, ceil, floor

class KclTriangle:
    def list_map_add(list_, map_, elem):
        if elem not in map_:
            map_[elem] = len(list_)
            list_.append(elem)
        return map_[elem]

    def __init__(self, vertices, normal, clps_index):
        self.vertices = vertices
        self.normal = normal
        self.clps_index = clps_index

        cosines = [(vertices[(i + 1) % 3] - vertices[i]).normalized().dot(
                   (vertices[(i - 1) % 3] - vertices[i]).normalized())
                for i in range(3)]
        v_index = max(range(3), key=lambda i: cosines[i])

        self.v_first = vertices[v_index]
        self.edge_normals = [(vertices[i % 3] - vertices[(i - 1) % 3]).cross(normal).normalized()
                for i in range(v_index, v_index + 3)]
        # Distance from first vertex to opposite edge
        self.length = (vertices[(v_index + 1) % 3] - vertices[v_index % 3]).dot(
                self.edge_normals[2])

    def add_vertex_and_normals_to_lists(self, v_list, v_map, n_list, n_map):
        self.vertex_id = KclTriangle.list_map_add(v_list, v_map, from_vec(self.v_first, 4, 6))
        self.normal_id = KclTriangle.list_map_add(n_list, n_map, from_vec(self.normal, 2, 10))
        self.edge_ids = [KclTriangle.list_map_add(n_list, n_map, from_vec(e, 2, 10))
            for e in self.edge_normals]

    def export(self, bytestr):
        bytestr += from_fix(self.length, 4, 16)
        bytestr += from_uint(self.vertex_id, 2)
        bytestr += from_uint(self.normal_id, 2)
        for e in self.edge_ids:
            bytestr += from_uint(e, 2)
        bytestr += from_uint(self.clps_index, 2)

    def import_(bytestr, offset):
        v_addr = to_uint(bytestr, 0x0, 4)
        n_addr = to_uint(bytestr, 0x4, 4)

        length = to_fix(bytestr, offset, 4, 16)
        vertex = Vector(to_vec(bytestr, v_addr + 12 * to_uint(bytestr, offset + 0x4, 2), 4, 3, 6))
        normal = Vector(to_vec(bytestr, n_addr + 6 * to_uint(bytestr, offset + 0x6, 2), 2, 3, 10))
        edge_normals = [Vector(to_vec(bytestr, 
                n_addr + 6 * to_uint(bytestr, offset + i, 2), 2, 3, 10))
                for i in [0x8, 0xa, 0xc]]
        clps_index = to_uint(bytestr, offset + 0xe, 2)

        # From SM64DSe
        crossB = normal.cross(edge_normals[1])
        crossA = normal.cross(edge_normals[0])
        dotB = crossB.dot(edge_normals[2])
        dotA = crossA.dot(edge_normals[2])
        vertices = [vertex,
                vertex + crossB * (length / dotB if dotB != 0 else 0),
                vertex + crossA * (length / dotA if dotA != 0 else 0)]

        return KclTriangle(vertices, normal, clps_index)


    def intersects_box(self, center, half_width):
        """
        Intersection test for triangle and axis-aligned cube.

        Test if the triangle  intersects the axis-aligned cube given by the center
        and half width. This algorithm is an adapted version of the algorithm
        presented here:
        http://fileadmin.cs.lth.se/cs/Personal/Tomas_Akenine-Moller/code/tribox3.txt
        """
        vs = [v - center for v in self.vertices]
        n = self.normal

        # Test for separation along the axes normal to the faces of the cube
        for i in range(3):
            if max(v[i] for v in vs) < -half_width or min(v[i] for v in vs) > half_width:
                return False

        # Test for separation along the axis normal to the face of the triangle
        d = n.dot(vs[0])
        r = half_width * sum(abs(c) for c in n)
        if d < -r or d > r:
            return False

        # Test for separation along the axes parallel to the cross products of the
        # edges of the triangle and the edges of the cube
        for i in range(3):
            if KclTriangle.edge_test(vs[i % 3], vs[(i + 1) % 3], vs[(i + 2) % 3],
                    half_width):
                return False

        # Triangle and box intersects
        return True

    def edge_test(v0, v1, v2, half_width):
        e = v1 - v0
        if KclTriangle.edge_axis_test(e.z, -e.y, v0.y, v0.z, v2.y, v2.z, half_width):
            return True
        if KclTriangle.edge_axis_test(-e.z, e.x, v0.x, v0.z, v2.x, v2.z, half_width):
            return True
        if KclTriangle.edge_axis_test(e.y, -e.x, v0.x, v0.y, v2.x, v2.y, half_width):
            return True
        return False

    def edge_axis_test(a1, a2, b1, b2, c1, c2, half_width):
        p = a1 * b1 + a2 * b2
        q = a1 * c1 + a2 * c2
        r = half_width * (abs(a1) + abs(a2))
        return max(p, q) < -r or min(p, q) > r
        

class KclMesh:
    def __init__(self, triangles):
        self.triangles = triangles
        self.vertex_list = []
        self.normal_list = []
        vertex_map = {}
        normal_map = {}
        for t in triangles:
            t.add_vertex_and_normals_to_lists(self.vertex_list, vertex_map,
                self.normal_list, normal_map)

    def export(self):
        bytestr = bytearray()
        for t in self.triangles:
            t.export(bytestr)
        return bytestr


# Taken from SM64DSe
class Octree:
    def create(mesh, max_triangles, min_width):
        """
        Returns an octree where the cube of each leaf node intersects less than
        max_triangles of the triangles, unless that would make the width of the cube
        less than min_width.
        """
        self = Octree()
        self.triangles = mesh.triangles
        self.max_triangles = max_triangles
        self.min_width = min_width

        min_c = [min(v[i] for t in mesh.triangles for v in t.vertices) for i in range(3)]
        max_c = [max(v[i] for t in mesh.triangles for v in t.vertices) for i in range(3)]
        
        # If model only uses two axes, eg. flat square, 
        # the base width will get set to min_width (1)
        # which can create an octree with 100's of thousands of tiny empty or almost empty nodes
        # which is very computationally expensive
        for i in range(3):
            if min_c[i] == max_c[i]:
                max_c[i] += max(max_c[j % 3] - min_c[j % 3] for j in (i + 1, i + 2)) / 4
        
        self.widths = [2 ** ceil(log2(max(max_c[i] - min_c[i], min_width))) for i in range(3)]
        self.base_width = min(self.widths)
        self.base = Vector(min_c)

        # Cap number of boxes at 128
        while self.widths[0] * self.widths[1] * self.widths[2] / (self.base_width ** 3) > 128:
            for i in range(3):
                if self.widths[i] == self.base_width:
                    self.widths[i] *= 2
            self.base_width *= 2

        self.num_c = [int(floor(self.widths[i] / self.base_width)) for i in range(3)]

        indexes = list(range(len(mesh.triangles)))
        self.children = []

        for k in range(self.num_c[2]):
            for j in range(self.num_c[1]):
                for i in range(self.num_c[0]):
                    self.children.append(Octree.child(
                        self.base + Vector((i, j, k)) * self.base_width,
                        self.base_width, indexes, mesh.triangles,
                        max_triangles, min_width, 0))

        return self

    def child(base, width, indexes, triangles, max_triangles, min_width, depth):
        self = Octree()
        center = base + Vector((width,) * 3) / 2
        self.width = width
        self.triangles = triangles
        self.max_triangles = max_triangles
        self.min_width = min_width
        self.depth = depth # Just for debugging

        # SM64DSe divides self.widths by 2 and initializes self.base_width using self.widths.
        # However, those would just be 0 since they didn't get initialized.

        self.base = Vector((width / 2,) * 3)
        self.real_base = base # for debugging
        self.indexes = []
        for i in indexes:
            if triangles[i].intersects_box(center, width / 2):
                self.indexes.append(i)

        # self.add_debug_mesh()
        self.is_leaf = True
        self.children = []
        if len(self.indexes) > max_triangles and width >= 2 * min_width:
            for k in range(2):
                for j in range(2):
                    for i in range(2):
                        self.children.append(Octree.child(
                            base + Vector((i, j, k)) / 2 * width,
                            width / 2, self.indexes, triangles,
                            max_triangles, min_width, depth + 1))

            self.indexes.clear()
            self.is_leaf = False

        return self

    debug_mesh_verts = [(x, y, z) for x in [0,1] for y in [0,1] for z in [0,1]]
    debug_mesh_edges = [[0,1], [1,5], [5,4], [4,0],
                        [0,2], [1,3], [5,7], [4,6],
                        [3,2], [2,6], [6,7], [7,3]]
    debug_mesh_faces = [[0,1,5,4], [0,2,3,1], [1,3,7,5], [5,7,6,4], [4,6,2,0], [3,2,6,7]]

    def add_debug_mesh(self):
        debug_mesh = bpy.data.meshes.new("Debug Mesh")
        debug_mesh.from_pydata(Octree.debug_mesh_verts, [], Octree.debug_mesh_faces)

        obj = bpy.data.objects.new("Debug Mesh", debug_mesh)
        bpy.context.scene.objects.link(obj)
        obj.location = self.real_base
        obj.scale = Vector((self.width,) * 3)
        obj.modifiers.new("Wireframe", "WIREFRAME")

    def export(self):
        branches = [self]
        free_list_offset = 0
        list_offsets_idx = []
        list_offsets_addr = []

        # Not a for loop because branches may get extra elements it has to deal with
        i = 0
        while i < len(branches):
            for node in branches[i].children:
                if node.is_leaf:
                    if len(node.indexes) == 0:
                        continue

                    if node.indexes in list_offsets_idx:
                        continue
                    
                    #print(node.indexes)
                    list_offsets_idx.append(node.indexes)
                    list_offsets_addr.append(free_list_offset)
                    free_list_offset += 2 * (len(node.indexes) + 1)

                else:
                    branches.append(node)

            i += 1

        list_base = 0
        for b in branches:
            list_base += 4 * len(b.children)

        list_offsets_idx.append([])
        list_offsets_addr.append(free_list_offset - 2)
        branch_base = 0
        free_branch_offset = 4 * len(self.children)

        bytestr = bytearray()

        for branch in branches:
            for node in branch.children:
                if node.is_leaf:
                    key = list_offsets_idx.index(node.indexes)
                    bytestr += from_uint(1 << 31 | 
                            (list_base + list_offsets_addr[key] - 2 - branch_base), 4)
                else:
                    bytestr += from_uint(free_branch_offset - branch_base, 4)
                    free_branch_offset += 4 * len(node.children)

            branch_base += 4 * len(branch.children)

        del list_offsets_idx[-1]
        del list_offsets_addr[-1]

        for indexes in list_offsets_idx:
            for index in indexes:
                bytestr += from_uint(index + 1, 2)
            bytestr += from_uint(0, 2)

        return bytestr

    def import_(bytestr):
        self = Octree()

        self.base = Vector(to_vec(bytestr, 0x14, 4, 3, 6))
        self.widths = [-c for c in to_vec(bytestr, 0x20, 4, 3, 0)]
        self.base_width = 2 ** to_uint(bytestr, 0x2c, 4)
        self.num_c = [int(w // self.base_width) for w in self.widths]

        self.children = []
        addr = to_uint(bytestr, 0x0c, 4)
        for k in range(self.num_c[2]):
            for j in range(self.num_c[1]):
                for i in range(self.num_c[0]):
                    self.children.append(Octree.import_child(bytestr, addr,
                            addr + 4 * ((k * self.num_c[1] + j) * self.num_c[0] + i),
                            self.base + Vector((i, j, k)) * self.base_width,
                            self.base_width, 0))
        
        return self

    def import_child(bytestr, grid_offset, offset, base, width, depth):
        self = Octree()
        self.real_base = base
        self.width = width
        self.depth = depth

        word = to_uint(bytestr, offset, 4)

        self.add_debug_mesh()

        self.children = []
        if word >> 31 & 1:
            self.is_leaf = True
            self.triangles = []

            tri_addr = grid_offset + word % 2 ** 31 + 2
            tri_index = to_uint(bytestr, tri_addr, 2)
            while tri_index != 0:
                tri_addr += 2
                tri_index = to_uint(bytestr, tri_addr, 2)

        else:
            self.is_leaf = False
            for k in range(2):
                for j in range(2):
                    for i in range(2):
                        self.children.append(Octree.import_child(bytestr,
                            grid_offset + word,
                            grid_offset + word + 4 * ((k * 2 + j) * 2 + i),
                            base + Vector((i, j, k)) * width / 2,
                            width / 2, depth + 1))

        return self

