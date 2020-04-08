import bpy
import os
import re
from bpy.props import StringProperty, BoolProperty, FloatProperty, EnumProperty

from bpy_extras.io_utils import ExportHelper, ImportHelper
bl_info = {
    "name": "SM256e",
    "author": "Josh65536",
    "blender": (2, 82, 0),
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
    filter_glob: StringProperty(default="*.bmd", options={"HIDDEN"})

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
    filter_glob: StringProperty(default="*.bmd", options={"HIDDEN"})

    @property
    def check_extension(self):
        return True

    def execute(self, context):
        if not self.filepath:
            raise Exception("filepath not set")

        from . import export_bmd
        return export_bmd.save(context, self.filepath)


class ImportLevelJson(bpy.types.Operator, ExportHelper):
    bl_idname = "import_scene.json"
    bl_label = "Import JSON"
    bl_options = {"PRESET"}

    filename_ext = ".json"
    filter_glob: StringProperty(default="*.json", options={"HIDDEN"})

    @property
    def check_extension(self):
        return True

    def execute(self, context):
        if not self.filepath:
            raise Exception("filepath not set")

        from .spoiler import main
        return main.load_level(self, context, self.filepath)


class ExportLevelJson(bpy.types.Operator, ExportHelper):
    bl_idname = "export_scene.json"
    bl_label = "Export JSON"
    bl_options = {"PRESET"}

    filename_ext = ".json"
    filter_glob: StringProperty(default="*.json", options={"HIDDEN"})

    @property
    def check_extension(self):
        return True

    def execute(self, context):
        if not self.filepath:
            raise Exception("filepath not set")

        from .spoiler import main
        return main.save_level(self, context, self.filepath)


def export_menu_func(self, context):
    self.layout.operator(ExportBMD.bl_idname, text="NDS Binary Model Format (.bmd)")


def import_menu_func(self, context):
    self.layout.operator(ImportBMD.bl_idname, text="NDS Binary Model Format (.bmd)")


def export_json_menu_func(self, context):
    self.layout.operator(ExportLevelJson.bl_idname, text="SM256 Level JSON (.json)")


def import_json_menu_func(self, context):
    self.layout.operator(ImportLevelJson.bl_idname, text="SM256 Level JSON (.json)")


bpy_classes = (ImportBMD, ExportLevelJson, ImportLevelJson)
bpy_class_register, bpy_class_unregister = bpy.utils.register_classes_factory(bpy_classes)

def register():
    bpy_class_register()

    # Suppress the BMD exporter until vertex color baking is supported
    bpy.types.TOPBAR_MT_file_import.append(import_menu_func)
    # bpy.types.TOPBAR_MT_file_export.append(export_menu_func)
    bpy.types.TOPBAR_MT_file_import.append(import_json_menu_func)
    bpy.types.TOPBAR_MT_file_export.append(export_json_menu_func)

    from .spoiler import main
    for c in main.bpy_classes:
        # Ensure that recursion works
        props = c.__annotations__ if hasattr(c, "__annotations__") else None
        if props is not None:
            delattr(c, "__annotations__")
        bpy.utils.register_class(c)
        if props is not None:
            c.__annotations__ = props

    # Now that all the classes have bl_rna attributes,
    # we can load in the properties
    for c in main.bpy_classes:
        bpy.utils.unregister_class(c)
        bpy.utils.register_class(c)

    bpy.types.Scene.level = bpy.props.PointerProperty(type=main.LevelSettings)

    for c in main.bpy_classes:
        match = re.match(r"ObjectStruct<(?P<arg>\w+)>", c.full_name) \
                if hasattr(c, "full_name") else None
        if match:
            setattr(bpy.types.Object, main.name_to_attr(match["arg"]),
                    bpy.props.PointerProperty(type=c))

    bpy.types.SpaceView3D.draw_handler_add(main.update_object_struct_arrays, (),
            "WINDOW", "POST_VIEW")


def unregister():
    bpy_class_unregister()

    bpy.types.TOPBAR_MT_file_import.remove(import_menu_func)
    # bpy.types.TOPBAR_MT_file_export.remove(export_menu_func)
    bpy.types.TOPBAR_MT_file_import.remove(import_json_menu_func)
    bpy.types.TOPBAR_MT_file_export.remove(export_json_menu_func)

    from .spoiler import main
    bpy.types.SpaceView3D.draw_handler_remove(main.update_object_struct_arrays, (),
            "WINDOW")

    for c in main.bpy_classes:
        bpy.utils.unregister_class(c)
    del bpy.types.Scene.level

    for c in main.bpy_classes:
        match = re.match(r"ObjectStruct<(?P<arg>\w+)>", c.full_name) \
                if hasattr(c, "full_name") else None
        if match:
            delattr(bpy.types.Object, main.name_to_attr(match["arg"]))

if __name__ == "__main__":
    register()
