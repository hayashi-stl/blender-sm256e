# blender-sm256e

Allows exporting of BMDs from Blender v2.79 (though it probably works with v2.78 as well, haven't checked).

## Installation

Download the zip file [here](https://github.com/josh65536/blender-sm256e/archive/master.zip), then open Blender, go to File > User Preferences > Addons and click "Install Addon from File". Install the downloaded zip file and enable the addon. Now you can export BMDs!

## Exporting Rules

* To export a rigged model, select the mesh and the armature. Make sure the armature is a parent of the mesh and that vertex groups have been set up properly.
    * Each vertex should have 1 group.
    * The limit is 32 bones.
* To export a stage model as multiple rooms, select up to 8 meshes. The order of the rooms is the alphabetical order of the *mesh*es' names (not the *object*s' names).
* To export a regular model, just select 1 mesh.
* Triangles and quadrilaterals only. No pentagons, hexagons, diacosiapentacontahexagons, etc.
* The *local* Y axis of the mesh (and armature) should point up.
* If a material has a texture, all texture slots that have textures should have image textures, and texture slot 0 should exist. Textures from other texture slots will be exported, but they will not be assigned to that material.
* **To make a material use backface culling (recommended!), change the engine to "Blender Game", then enable "Backface Culling" in the material's game settings, then change the engine back to "Blender Render" (last step optional).**
* To make a material use vertex colors instead of shading, enable "Vertex Color Paint" in the material's options.
* To make a material use regular (modulation) shading, set its texture blend type to "Multiply".
* To make a material use decal shading, set its texture blend type to "Mix".
* To make a material use equality as its depth test, add a custom property called "Depth Equal" to the material and set it to 1.
* To make a material use its texture as an environment map, add a custom property called "Environment Map" to the material and set it to 1.
* To avoid compressing a texture that would be compressed, add a custom property called "Uncompressed" to the texture and set it to 1.
