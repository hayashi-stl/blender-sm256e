import bpy
import os
from bpy.props import StringProperty, BoolProperty, FloatProperty, EnumProperty

from bpy_extras.io_utils import ExportHelper, ImportHelper
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

class ImportBMD(bpy.types.Operator, ImportHelper):
    bl_idname = "import_scene.bmd"
    bl_label = "Import BMD"
    bl_options = {"PRESET"}

    filename_ext = ".bmd"
    filter_glob = StringProperty(default="*.bmd", options={"HIDDEN"})

    @property
    def check_extension(self):
        return True

    def execute(self, context):
        if not self.filepath:
            raise Exception("filepath not set")

        from . import import_bmd
        return import_bmd.load(context, self.filepath)
    

class ExportBMD(bpy.types.Operator, ExportHelper):
    """Selection to BMD"""
    bl_idname = "export_scene.bmd"
    bl_label = "Export BMD"
    bl_options = {"PRESET"}

    filename_ext = ".bmd"
    filter_glob = StringProperty(default="*.bmd", options={"HIDDEN"})

    sm256 = BoolProperty(name="SM256-specific BMD",
            description="Export a SM256-specific BMD, which includes range and range offset Y",
            default="SM256_ROOT" in os.environ)

    @property
    def check_extension(self):
        return True

    def execute(self, context):
        if not self.filepath:
            raise Exception("filepath not set")

        keywords = self.as_keywords(ignore=("axis_forward",
                                            "axis_up",
                                            "global_scale",
                                            "check_existing",
                                            "filter_glob",
                                            "xna_validate",
                                            ))

        from . import export_bmd
        return export_bmd.save(context, **keywords)


class ExportBCA(bpy.types.Operator, ExportHelper):
    """Selection to BCA"""
    bl_idname = "export_scene.bca"
    bl_label = "Export BCA"
    bl_options = {"PRESET"}

    filename_ext = ".bca"
    filter_glob = StringProperty(default="*.bca", options={"HIDDEN"})

    @property
    def check_extension(self):
        return True

    def execute(self, context):
        if not self.filepath:
            raise Exception("filepath not set")

        from . import export_bca
        return export_bca.save(context, self.filepath)


class ImportKCL(bpy.types.Operator, ImportHelper):
    bl_idname = "import_scene.kcl"
    bl_label = "Import KCL (Debug only!)"
    bl_options = {"PRESET"}

    filename_ext = ".kcl"
    filter_glob = StringProperty(default="*.kcl", options={"HIDDEN"})

    @property
    def check_extension(self):
        return True

    def execute(self, context):
        if not self.filepath:
            raise Exception("filepath not set")

        from . import import_kcl
        return import_kcl.load(context, self.filepath)
    

class ExportKCL(bpy.types.Operator, ExportHelper):
    """Selection to KCL"""
    bl_idname = "export_scene.kcl"
    bl_label = "Export KCL"
    bl_options = {"PRESET"}

    filename_ext = ".kcl"
    filter_glob = StringProperty(default="*.kcl", options={"HIDDEN"})

    scale = FloatProperty(name="Scale",
            description="Factor to scale the model by",
            default=1.0)
    one_clps_index = BoolProperty(name="Use CLPS index 0",
            description="Have all faces use CLPS index 0",
            default=False)

    @property
    def check_extension(self):
        return True

    def execute(self, context):
        if not self.filepath:
            raise Exception("filepath not set")

        keywords = self.as_keywords(ignore=("axis_forward",
                                            "axis_up",
                                            "global_scale",
                                            "check_existing",
                                            "filter_glob",
                                            "xna_validate",
                                            ))

        from . import export_kcl
        return export_kcl.save(context, **keywords)


def export_menu_func(self, context):
    self.layout.operator(ExportBMD.bl_idname, text="NDS Binary Model Format (.bmd)")


def import_menu_func(self, context):
    self.layout.operator(ImportBMD.bl_idname, text="NDS Binary Model Format (.bmd)")


def export_bca_menu_func(self, context):
    self.layout.operator(ExportBCA.bl_idname, text="NDS Binary Character Animation (.bca)")


def export_kcl_menu_func(self, context):
    self.layout.operator(ExportKCL.bl_idname, text="NDS K. Collision Format (.kcl)")


def import_kcl_menu_func(self, context):
    self.layout.operator(ImportKCL.bl_idname, text="NDS K. Collision Format (.kcl)")


def register():
    bpy.utils.register_module(__name__)

    bpy.types.INFO_MT_file_import.append(import_menu_func)
    bpy.types.INFO_MT_file_export.append(export_menu_func)
    bpy.types.INFO_MT_file_export.append(export_bca_menu_func)
    bpy.types.INFO_MT_file_import.append(import_kcl_menu_func)
    bpy.types.INFO_MT_file_export.append(export_kcl_menu_func)


def unregister():
    bpy.utils.unregister_module(__name__)

    bpy.types.INFO_MT_file_import.remove(import_menu_func)
    bpy.types.INFO_MT_file_export.remove(export_menu_func)
    bpy.types.INFO_MT_file_export.remove(export_bca_menu_func)
    bpy.types.INFO_MT_file_import.remove(import_kcl_menu_func)
    bpy.types.INFO_MT_file_export.remove(export_kcl_menu_func)

if __name__ == "__main__":
    register()
