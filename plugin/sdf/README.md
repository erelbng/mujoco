<h1>
  <a href="#"><img alt="MuJoCo" src="../../banner.png" width="100%"/></a>
</h1>

## Signed distance function (SDF) plugins

These are first-party plugins that implement implicit geometries using SDFs. They can be applied to **geoms** and
**meshes** (in the **asset** section). Sample models can be found in [this folder](../../model/plugin/sdf/).

### Bolt

Implemented in [bolt.cc](bolt.cc). Example usage in [nutbolt.xml](../../model/plugin/sdf/nutbolt.xml).

This plugin implements a bolt with a hexagonal head, similar to https://www.shadertoy.com/view/XtffzX.

Parameters:

   - `radius` [m]: bolt radius (default `0.26`).

### Bowl

Implemented in [bowl.cc](bowl.cc). Example usage in [bowl.xml](../../model/plugin/sdf/bowl.xml).

The plugin implements a cut hollow sphere from https://www.shadertoy.com/view/7tVXRt.

Parameters:

   - `height` [m]: location of the cut plane (default `0.4`).
   - `radius` [m]: radius of the sphere (default `1`).
   - `thickness` [m]: thickness of the bowl (default `0.02`).

### Gear

Implemented in [gear.cc](gear.cc). Example usage in [gear.xml](../../model/plugin/sdf/gear.xml).

The plugin implements a 3D extrusion of the 2D gear geometry from https://www.shadertoy.com/view/3lG3WR.

Parameters:

   - `alpha` [m]: initial angle of rotation of the gear (default `0`).
   - `diameter` [m]: gear diameter (default `2.8`).
   - `teeth` []: number of teeth (default `25`).

### Nut

Implemented in [nut.cc](nut.cc). Example usage in [nutbolt.xml](../../model/plugin/sdf/nutbolt.xml).

This plugin implements a hexagonal nut identical to the bolt head from https://www.shadertoy.com/view/XtffzX.

Parameters:

   - `radius` [m]: nut radius (default `0.26`).

### Torus

Implemented in [torus.cc](torus.cc). Example usage in [torus.xml](../../model/plugin/sdf/torus.xml).

This plugin implements a torus.

Parameters:

  - `radius1` [m]: major radius (default `0.35`).
  - `radius1` [m]: minor radius (default `0.15`).

### EFlesh (SDF + tactile sensor)

Implemented in [eflesh.cc](eflesh.cc). Example usage in
[eflesh_test.xml](../../model/plugin/sdf/eflesh_test.xml) (SDF only) and
[eflesh_sensor.xml](../../model/plugin/sdf/eflesh_sensor.xml) (SDF + sensor).

This plugin simulates the [eFlesh](https://github.com/notvenky/eFlesh) magnetic
tactile sensor: a soft magnetized lattice on a magnetometer board that reads the
field change produced when the flesh deforms under contact. It is a **single
plugin with two capabilities**, `mjPLUGIN_SDF` and `mjPLUGIN_SENSOR`, so one
named `<instance>` is shared by the `<mesh>`/`<geom>` (shape) and the `<sensor>`
(tactile readout):

```xml
<extension>
  <plugin plugin="mujoco.sdf.eflesh">
    <instance name="flesh">
      <config key="size_x" value="0.15"/>
      <config key="size_y" value="0.15"/>
      <config key="size_z" value="0.04"/>
      <config key="e_module" value="2000"/>
      <config key="nx" value="8"/>
      <config key="ny" value="8"/>
    </instance>
  </plugin>
</extension>
...
<geom type="sdf" mesh="flesh" rgba="0.9 0.55 0.5 0.3"><plugin instance="flesh"/></geom>
<site name="eflesh"/>
...
<sensor>
  <plugin name="eflesh" instance="flesh" objtype="site" objname="eflesh"/>
</sensor>
```

Parameters:

  - `shape` []: eFlesh body shape: `0` = box (default), `1` = sphere. See
    [eflesh_sphere.xml](../../model/plugin/sdf/eflesh_sphere.xml).
  - `size_x`, `size_y`, `size_z` [m]: box half-extents (default `0.1`). For a
    **sphere**, `size_x` is the radius (`size_y`, `size_z` ignored).
  - `e_module` [Pa]: elastic (Young's) modulus of the flesh (default `1e5`). A
    softer flesh (smaller modulus) produces a stronger signal for the same force.
  - `nx`, `ny` []: taxel grid resolution (default `4`x`4`). Box: over the top
    face. Sphere: `nx` longitude x `ny` latitude cells over the whole surface.
  - `viz` []: `1` (default) draws the visualization, `0` disables it.

Adding a shape is localized: implement its cases in the `Shape*` helpers in
[eflesh.cc](eflesh.cc) (SDF distance/gradient/aabb, taxel layout, contact
binning, tile size); the rest of the plugin -- sensing, output, and the
surface-tangent tile + arrow visualization -- is generic.

The sensor is attached to a **site** whose parent body carries the eFlesh geom.
It is a **fixed `nx`x`ny` grid of taxels** tiling the top face, exactly like the
`touch_grid` sensor: each contact is binned into the single taxel cell it falls
in (no smearing), and each taxel sums the contact forces in its cell. A small
touch therefore lights up a single taxel; MuJoCo splitting one touch into many
micro-contacts is harmless because they all fall in the same cell and sum to the
true load.

For each taxel it reports, in the site (board) frame:

  - position  (3): the fixed taxel cell center;
  - direction (3): the SDF surface normal at the taxel (clean). The direction is
    taken from the SDF gradient rather than the raw contact normals, which are
    noisy when one touch splits into many contacts -- this keeps the reported
    direction stable;
  - magnitude (1): the compressive (normal) contact force in the cell divided by
    `e_module` -- the lateral/shear part is projected out so the pressure reading
    is robust.

The sensor output is a flat array of `7 * nx * ny` numbers laid out as
`[pos(3), dir(3), mag(1)]` per taxel (x-fastest).

When `viz` is on, the sensor draws (every frame, like `touch_grid`):

  - the whole taxel grid as flat, **semi-transparent** tiles over the surface,
    colored blue(cold) -> red(hot) by per-taxel magnitude (more opaque where
    pressed), so the array is always visible; and
  - a **force arrow** on each active taxel, along the reported direction with
    length proportional to the magnitude.

Give the flesh geom an `rgba` alpha < 1 so the tiles are visible inside it.
The visualization reads only from `sensordata`, so it renders correctly in the
passive viewer (which renders a copied `mjData`).

### How to make your own SDF

Create your `MySDF.h` and `MySDF.cc` files in the SDF folder, where this README is located. Implement your SDF using the
following interface:

```
struct MySDFAttribute {
  static constexpr int nattribute =
  /* insert the number of attributes */;
  static constexpr char const* names[nattribute] =
  /* an array of attributes with the same order as the attribute array in your SDF class */;
  static constexpr mjtNum defaults[nattribute] =
  /* an array of default values for your attributes */;
};

class MySDF {
 public:
  // creates a new MySDF instance or returns null on failure.
  static std::optional<MySDF> Create(const mjModel* m, mjData* d, int instance);
  MySDF(MySDF&&) = default;
  ~MySDF() = default;

  // functions that return the SDF and its gradient at a query point
  mjtNum Distance(const mjtNum point[3]) const;
  void Gradient(mjtNum grad[3], const mjtNum point[3]) const;

  // a call to this needs to be added to register.cc
  static void RegisterPlugin();

  // an array of attributes with the same order as in the struct above
  mjtNum attribute[MySDFAttribute::nattribute];

 private:
  MySDF(const mjModel* m, mjData* d, int instance);
};
```







