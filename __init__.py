import bpy
from bpy.props import StringProperty, BoolProperty, FloatProperty, EnumProperty

from bpy_extras.io_utils import ExportHelper
bl_info = {
    "name": "SM256e",
    "author": "Josh65536",
    "blender": (2, 7, 9),
    "description": ("Adds tools for Super Mario 256 development"),
    "category": "Import-Export"}

if "bpy" in locals():
    import importlib
    if "export_bmd" in locals():
        importlib.reload(export_bmd)  # noqa


class ExportBMD(bpy.types.Operator, ExportHelper):
    """Selection to BMD"""
    bl_idname = "export_scene.bmd"
    bl_label = "Export BMD"
    bl_options = {"PRESET"}

    filename_ext = ".bmd"
    filter_glob = StringProperty(default="*.bmd", options={"HIDDEN"})

    @property
    def check_extension(self):
        return True

    def execute(self, context):
        if not self.filepath:
            raise Exception("filepath not set")

        from . import export_bmd
        return export_bmd.save(context, self.filepath)


def menu_func(self, context):
    self.layout.operator(ExportBMD.bl_idname, text="NDS Binary Model Format (.bmd)")


def register():
    bpy.utils.register_module(__name__)

    bpy.types.INFO_MT_file_export.append(menu_func)


def unregister():
    bpy.utils.unregister_module(__name__)

    bpy.types.INFO_MT_file_export.remove(menu_func)

if __name__ == "__main__":
    register()
