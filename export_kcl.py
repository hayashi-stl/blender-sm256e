from .util import *

def export_mesh(context, obj):
    mesh = obj.data
    tri_mod = obj.modifiers.new("Triangulate", "TRIANGULATE")
    mesh = obj.to_mesh(context.scene, True, "RENDER")
    obj.modifiers.remove(tri_mod)

    tris = [([mesh.vertices[v].co for v in f.vertices], f.material_index) for f in mesh.polygons]
    for f in tris:
        print(f)

def save(context, filepath):
    obj = context.active_object
    bytestr = export_mesh(context, obj)
    # with open(filepath, "wb") as f:
    #     f.write(bytestr)
        
    return {"FINISHED"}
