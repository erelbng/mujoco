// Copyright 2022 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef MUJOCO_PLUGIN_SDF_EFLESH_H_
#define MUJOCO_PLUGIN_SDF_EFLESH_H_

#include <optional>
#include <vector>

#include <mujoco/mjdata.h>
#include <mujoco/mjmodel.h>
#include <mujoco/mjtype.h>
#include <mujoco/mjvisualize.h>
#include "sdf.h"

namespace mujoco::plugin::sdf {

struct EFleshAttribute {
  static constexpr int nattribute = 8;
  // size_{x,y,z}: half-extents of the eFlesh box (meters). For a sphere, size_x
  //               is the radius (size_y, size_z are ignored).
  // e_module:     elastic (Young's) modulus of the soft flesh (Pa). Governs how
  //               much the flesh deforms -- and therefore how strong the
  //               tactile signal is -- for a given contact force.
  // nx, ny:       taxel grid resolution. Box: over the top face. Sphere: nx
  //               longitude x ny latitude cells over the whole surface.
  // viz:          1 = always draw the taxel grid + force arrows; 0 = off.
  // shape:        eFlesh body shape: 0 = box (default), 1 = sphere.
  static constexpr char const* names[nattribute] = {
      "size_x", "size_y", "size_z", "e_module", "nx", "ny", "viz", "shape"};
  static constexpr mjtNum defaults[nattribute] = {.1, .1, .1, 1.0e5, 4, 4, 1, 0};
};

// eFlesh body shapes. To add a new shape, add an enum value and implement its
// cases in the Shape* helpers in eflesh.cc (SDF, taxel layout, binning, tiles).
enum EFleshShape {
  kEFleshBox = 0,
  kEFleshSphere = 1,
};

// The eFlesh plugin (https://github.com/notvenky/eFlesh) is a single plugin
// with two capabilities:
//
//   * mjPLUGIN_SDF    -- a signed-distance function describing the soft flesh
//                        body (box or sphere, see `shape`), used for collisions.
//   * mjPLUGIN_SENSOR -- a tactile sensor simulating the real eFlesh device: a
//                        soft magnetized lattice on a magnetometer board.
//
// The sensor is a FIXED grid of nx*ny taxels tiling the sensing surface, exactly
// like the `touch_grid` sensor: each contact is binned into the single taxel
// cell it falls in (no smearing across the array), and each taxel sums the
// contact forces inside its cell. A single small touch therefore lights up a
// single taxel. For each taxel it reports, in the sensor-site frame:
//   - position  (3): the fixed taxel center on the surface;
//   - direction (3): the outward surface normal at the taxel (unit);
//   - magnitude (1): compressive contact force in the cell / e_module.
//
// The shape-specific geometry (SDF, taxel layout, contact binning, tile shape)
// lives in the Shape* helpers in eflesh.cc; the rest of the plugin is generic,
// so adding a shape only means adding cases there.
//
// A single plugin instance is shared by the <mesh>/<geom> (SDF) and the
// <sensor> (tactile) elements. The sensor output is a flat array of 7*nx*ny
// numbers laid out as [pos(3), dir(3), mag(1)] per taxel, in row-major
// (x-fastest) order.
class EFlesh {
 public:
  // Creates a new EFlesh instance or returns null on failure.
  static std::optional<EFlesh> Create(const mjModel* m, mjData* d, int instance);
  EFlesh(EFlesh&&) = default;
  ~EFlesh() = default;

  // SDF capability.
  mjtNum Distance(const mjtNum point[3]) const;
  void Gradient(mjtNum grad[3], const mjtNum point[3]) const;

  // Sensor capability.
  void Reset();
  void Compute(const mjModel* m, mjData* d, int instance);
  void Visualize(const mjModel* m, mjData* d, const mjvOption* opt,
                 mjvScene* scn, int instance);

  static void RegisterPlugin();

  mjtNum attribute[EFleshAttribute::nattribute];

 private:
  EFlesh(const mjModel* m, mjData* d, int instance);
};

}  // namespace mujoco::plugin::sdf

#endif  // MUJOCO_PLUGIN_SDF_EFLESH_H_
