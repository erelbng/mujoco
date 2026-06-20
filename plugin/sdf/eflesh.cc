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

#include <algorithm>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include <mujoco/mjplugin.h>
#include <mujoco/mjtype.h>
#include <mujoco/mjvisualize.h>
#include <mujoco/mujoco.h>
#include "sdf.h"
#include "eflesh.h"

namespace mujoco::plugin::sdf {
namespace {

// Numbers reported per taxel: position(3) + direction(3) + magnitude(1).
constexpr int kValuesPerTaxel = 7;

// Indices into the attribute array.
enum { kSizeX = 0, kSizeY, kSizeZ, kEModule, kNx, kNy, kViz, kShape };

// =========================================================================
// Shape-specific geometry. Everything that differs between shapes lives here;
// to add a shape, add an enum value (eflesh.h) and a case in each helper.
// All points/normals are in the geom-local frame.
// =========================================================================

int ShapeOf(const mjtNum* a) { return static_cast<int>(a[kShape]); }

// ---- Box (centered at origin, half-extents a[0..2]) ----
mjtNum BoxDistance(const mjtNum p[3], const mjtNum b[3]) {
  mjtNum dx = mju_abs(p[0]) - b[0];
  mjtNum dy = mju_abs(p[1]) - b[1];
  mjtNum dz = mju_abs(p[2]) - b[2];
  mjtNum qx = mju_max(dx, 0.0), qy = mju_max(dy, 0.0), qz = mju_max(dz, 0.0);
  return mju_sqrt(qx*qx + qy*qy + qz*qz) +
         mju_min(mju_max(dx, mju_max(dy, dz)), 0.0);
}

void BoxGradient(mjtNum g[3], const mjtNum p[3], const mjtNum b[3]) {
  mjtNum dx = mju_abs(p[0]) - b[0];
  mjtNum dy = mju_abs(p[1]) - b[1];
  mjtNum dz = mju_abs(p[2]) - b[2];
  mjtNum qx = mju_max(dx, 0.0), qy = mju_max(dy, 0.0), qz = mju_max(dz, 0.0);
  mjtNum len = mju_sqrt(qx*qx + qy*qy + qz*qz);
  if (len > 0) {
    g[0] = (p[0] >= 0 ? 1.0 : -1.0) * qx / len;
    g[1] = (p[1] >= 0 ? 1.0 : -1.0) * qy / len;
    g[2] = (p[2] >= 0 ? 1.0 : -1.0) * qz / len;
  } else {
    g[0] = g[1] = g[2] = 0;
    if (dx > dy && dx > dz)      g[0] = (p[0] >= 0 ? 1.0 : -1.0);
    else if (dy > dz)            g[1] = (p[1] >= 0 ? 1.0 : -1.0);
    else                         g[2] = (p[2] >= 0 ? 1.0 : -1.0);
  }
}

// ---- Sphere (centered at origin, radius a[0]) ----
mjtNum SphereDistance(const mjtNum p[3], mjtNum r) {
  return mju_norm3(p) - r;
}

void SphereGradient(mjtNum g[3], const mjtNum p[3]) {
  g[0] = p[0]; g[1] = p[1]; g[2] = p[2];
  if (mju_normalize3(g) < mjMINVAL) { g[0] = g[1] = 0; g[2] = 1; }
}

// ---- Generic SDF dispatch ----
mjtNum ShapeDistance(const mjtNum p[3], const mjtNum* a) {
  if (ShapeOf(a) == kEFleshSphere) return SphereDistance(p, a[kSizeX]);
  return BoxDistance(p, a);
}

void ShapeGradient(mjtNum g[3], const mjtNum p[3], const mjtNum* a) {
  if (ShapeOf(a) == kEFleshSphere) SphereGradient(g, p);
  else BoxGradient(g, p, a);
}

void ShapeAabb(mjtNum aabb[6], const mjtNum* a) {
  aabb[0] = aabb[1] = aabb[2] = 0;
  if (ShapeOf(a) == kEFleshSphere) {
    aabb[3] = aabb[4] = aabb[5] = a[kSizeX];
  } else {
    aabb[3] = a[kSizeX]; aabb[4] = a[kSizeY]; aabb[5] = a[kSizeZ];
  }
}

// Center and outward normal (local frame) of taxel (i, j).
void ShapeTaxel(int i, int j, const mjtNum* a, int nx, int ny,
                mjtNum pos[3], mjtNum nrm[3]) {
  if (ShapeOf(a) == kEFleshSphere) {
    mjtNum r = a[kSizeX];
    mjtNum phi = (i + 0.5) / nx * 2*mjPI;       // longitude
    mjtNum th  = (j + 0.5) / ny * mjPI;         // latitude (0 = +z pole)
    mjtNum st = mju_sin(th), ct = mju_cos(th);
    nrm[0] = st*mju_cos(phi); nrm[1] = st*mju_sin(phi); nrm[2] = ct;
    pos[0] = r*nrm[0]; pos[1] = r*nrm[1]; pos[2] = r*nrm[2];
  } else {  // box: top face (+z)
    pos[0] = -a[kSizeX] + (i + 0.5) * (2*a[kSizeX] / nx);
    pos[1] = -a[kSizeY] + (j + 0.5) * (2*a[kSizeY] / ny);
    pos[2] = a[kSizeZ];
    nrm[0] = 0; nrm[1] = 0; nrm[2] = 1;
  }
}

// Maps a local surface point (with outward normal `nrm`) to a taxel cell.
// Returns false if the point is not on a sensed part of the surface.
bool ShapeBin(const mjtNum p[3], const mjtNum nrm[3], const mjtNum* a,
              int nx, int ny, int* bi, int* bj) {
  if (ShapeOf(a) == kEFleshSphere) {
    mjtNum phi = mju_atan2(p[1], p[0]);
    if (phi < 0) phi += 2*mjPI;
    mjtNum r = mju_norm3(p);
    mjtNum th = mju_acos(mju_clip(r > 0 ? p[2]/r : 1, -1, 1));
    *bi = mju_clip(static_cast<int>(phi / (2*mjPI) * nx), 0, nx-1);
    *bj = mju_clip(static_cast<int>(th  /     mjPI * ny), 0, ny-1);
    return true;
  }
  // box: only the top face (+z) is sensed.
  if (nrm[2] < 0.7) return false;
  *bi = static_cast<int>((p[0] + a[kSizeX]) / (2*a[kSizeX]) * nx);
  *bj = static_cast<int>((p[1] + a[kSizeY]) / (2*a[kSizeY]) * ny);
  if (*bi < 0 || *bi >= nx || *bj < 0 || *bj >= ny) return false;
  return true;
}

// Representative visualization tile half-extents (in the tangent plane).
void ShapeTileSize(const mjtNum* a, int nx, int ny, mjtNum* hx, mjtNum* hy) {
  if (ShapeOf(a) == kEFleshSphere) {
    mjtNum r = a[kSizeX];
    *hx = 0.45 * r * (2*mjPI / nx);
    *hy = 0.45 * r * (mjPI / ny);
  } else {
    *hx = 0.95 * a[kSizeX] / nx;
    *hy = 0.95 * a[kSizeY] / ny;
  }
}

// Characteristic size, used to scale force arrows.
mjtNum ShapeSize(const mjtNum* a) {
  if (ShapeOf(a) == kEFleshSphere) return a[kSizeX];
  return mju_max(a[kSizeX], a[kSizeY]);
}

// =========================================================================
// Generic helpers
// =========================================================================

void GridSize(const mjtNum* a, int* nx, int* ny) {
  *nx = mju_max(1, static_cast<int>(a[kNx]));
  *ny = mju_max(1, static_cast<int>(a[kNy]));
}

int SensorId(const mjModel* m, int instance) {
  for (int id = 0; id < m->nsensor; ++id) {
    if (m->sensor_type[id] == mjSENS_PLUGIN &&
        m->sensor_plugin[id] == instance) {
      return id;
    }
  }
  return -1;
}

int GeomForInstance(const mjModel* m, int instance) {
  for (int g = 0; g < m->ngeom; ++g) {
    if (m->geom_plugin[g] == instance) return g;
  }
  return -1;
}

// Row-major rotation whose z-axis is `n`; arbitrary tangent x, y axes.
void FrameFromNormal(const mjtNum n[3], mjtNum mat[9]) {
  mjtNum z[3] = {n[0], n[1], n[2]};
  if (mju_normalize3(z) < mjMINVAL) { z[0] = z[1] = 0; z[2] = 1; }
  mjtNum ref[3] = {0, 0, 1};
  if (mju_abs(z[2]) > 0.9) { ref[0] = 1; ref[1] = 0; ref[2] = 0; }
  mjtNum x[3], y[3];
  mju_cross(x, ref, z); mju_normalize3(x);
  mju_cross(y, z, x);
  mat[0] = x[0]; mat[1] = y[0]; mat[2] = z[0];
  mat[3] = x[1]; mat[4] = y[1]; mat[5] = z[1];
  mat[6] = x[2]; mat[7] = y[2]; mat[8] = z[2];
}

mjvGeom* NextGeom(mjvScene* scn) {
  if (scn->ngeom >= scn->maxgeom) {
    if (!scn->status) {
      mju_warning("Pre-allocated visual geom buffer is full. "
                  "Increase maxgeom above %d.", scn->maxgeom);
      scn->status = 1;
    }
    return nullptr;
  }
  return scn->geoms + scn->ngeom;
}

void TagDecor(mjvGeom* g, int instance, int segid) {
  g->objtype = mjOBJ_UNKNOWN;
  g->objid = instance;
  g->category = mjCAT_DECOR;
  g->segid = segid;
}

// Blue -> green -> red heatmap for a value t in [0, 1].
void Heatmap(mjtNum t, float rgba[4]) {
  t = mju_clip(t, 0, 1);
  if (t < 0.5) {
    mjtNum f = t / 0.5;
    rgba[0] = 0; rgba[1] = static_cast<float>(f); rgba[2] = static_cast<float>(1 - f);
  } else {
    mjtNum f = (t - 0.5) / 0.5;
    rgba[0] = static_cast<float>(f); rgba[1] = static_cast<float>(1 - f); rgba[2] = 0;
  }
  rgba[3] = 1.0f;
}

// Flat-tile thickness, relative to the taxel cell size.
const mjtNum kRelativeThickness = 0.2;

}  // namespace

// factory function
std::optional<EFlesh> EFlesh::Create(
    const mjModel* m, mjData* d, int instance) {
  if (CheckAttr("size_x", m, instance) &&
      CheckAttr("size_y", m, instance) &&
      CheckAttr("size_z", m, instance)) {
    return EFlesh(m, d, instance);
  } else {
    mju_warning("Invalid size_x, size_y or size_z parameters in EFlesh plugin");
    return std::nullopt;
  }
}

// plugin constructor
EFlesh::EFlesh(const mjModel* m, mjData* d, int instance) {
  SdfDefault<EFleshAttribute> defattribute;
  for (int i = 0; i < EFleshAttribute::nattribute; i++) {
    attribute[i] = defattribute.GetDefault(
        EFleshAttribute::names[i],
        mj_getPluginConfig(m, instance, EFleshAttribute::names[i]));
  }
}

// sdf
mjtNum EFlesh::Distance(const mjtNum point[3]) const {
  return ShapeDistance(point, attribute);
}

// gradient of sdf
void EFlesh::Gradient(mjtNum grad[3], const mjtNum p[3]) const {
  ShapeGradient(grad, p, attribute);
}

// sensor: stateless, nothing to reset (readings live in d->sensordata)
void EFlesh::Reset() {}

// sensor: bin contact forces into a fixed taxel grid (like touch_grid)
void EFlesh::Compute(const mjModel* m, mjData* d, int instance) {
  // This callback fires for every instance during the sensor stage; if this
  // instance is not bound to a sensor, there is nothing to do.
  int id = SensorId(m, instance);
  if (id < 0) return;

  mjtNum* sensordata = d->sensordata + m->sensor_adr[id];
  mju_zero(sensordata, m->sensor_dim[id]);

  // The flesh geom shares this instance; it provides the SDF frame.
  int flesh_geom = GeomForInstance(m, instance);
  if (flesh_geom < 0) return;
  mjtNum* geom_pos = d->geom_xpos + 3*flesh_geom;
  mjtNum* geom_mat = d->geom_xmat + 9*flesh_geom;

  // Site frame: the rigid magnetometer board.
  int site_id = m->sensor_objid[id];
  mjtNum* site_pos = d->site_xpos + 3*site_id;
  mjtNum* site_mat = d->site_xmat + 9*site_id;
  int parent_weld = m->body_weldid[m->site_bodyid[site_id]];

  mjtNum* a = attribute;
  mjtNum e_module = a[kEModule] <= 0 ? 1.0 : a[kEModule];
  int nx, ny;
  GridSize(a, &nx, &ny);
  int ntaxel = nx*ny;

  // Per-taxel compressive (normal) force.
  std::vector<mjtNum> fnsum(ntaxel, 0);

  for (int i = 0; i < d->ncon; ++i) {
    const mjContact* con = d->contact + i;
    int weld1 = m->body_weldid[m->geom_bodyid[con->geom1]];
    int weld2 = m->body_weldid[m->geom_bodyid[con->geom2]];
    if (weld1 != parent_weld && weld2 != parent_weld) continue;

    // Contact force in the world frame (contact.frame maps contact->world).
    mjtNum f6[6], force[3];
    mj_contactForce(m, d, i, f6);
    mju_mulMatTVec3(force, con->frame, f6);

    // Contact point and outward surface normal in the geom-local frame.
    mjtNum rel[3], lpos[3], grad[3], surf_local[3];
    mju_sub3(rel, con->pos, geom_pos);
    mju_mulMatTVec3(lpos, geom_mat, rel);
    mjtNum dist = ShapeDistance(lpos, a);
    ShapeGradient(grad, lpos, a);
    mju_addScl3(surf_local, lpos, grad, -dist);  // surface = lpos - dist*grad

    // Which taxel cell does this contact fall in?
    int bi, bj;
    if (!ShapeBin(surf_local, grad, a, nx, ny, &bi, &bj)) continue;

    // Compressive normal force: project the contact force onto the (world)
    // outward normal. Using the SDF normal (not the noisy per-contact normal)
    // keeps the reading clean.
    mjtNum normal_world[3];
    mju_mulMatVec3(normal_world, geom_mat, grad);
    fnsum[bj*nx + bi] += mju_abs(mju_dot3(force, normal_world));
  }

  // Write taxel outputs (site frame): [pos(3), normal(3), magnitude(1)].
  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      int k = j*nx + i;
      mjtNum lpos[3], lnrm[3];
      ShapeTaxel(i, j, a, nx, ny, lpos, lnrm);

      // local -> world -> site frame.
      mjtNum wpos[3], wnrm[3], rel[3], psite[3], nsite[3];
      mju_mulMatVec3(wpos, geom_mat, lpos);
      mju_addTo3(wpos, geom_pos);
      mju_mulMatVec3(wnrm, geom_mat, lnrm);
      mju_sub3(rel, wpos, site_pos);
      mju_mulMatTVec3(psite, site_mat, rel);
      mju_mulMatTVec3(nsite, site_mat, wnrm);

      mjtNum* out = sensordata + kValuesPerTaxel*k;
      mju_copy3(out, psite);
      mju_copy3(out + 3, nsite);
      out[6] = fnsum[k] / e_module;
    }
  }
}

// sensor: draw the taxel array as (transparent) colored tiles tangent to the
// surface, plus a force arrow on each active taxel. Works for any shape because
// each taxel carries its own surface normal in sensordata. Controlled by `viz`.
void EFlesh::Visualize(const mjModel* m, mjData* d, const mjvOption* opt,
                       mjvScene* scn, int instance) {
  if (attribute[kViz] == 0) return;

  int id = SensorId(m, instance);
  if (id < 0) return;

  // Read only from d->sensordata + the site frame (no cached state) so this
  // renders correctly even on the viewer's copied mjData.
  const mjtNum* sensordata = d->sensordata + m->sensor_adr[id];
  int ntaxel = m->sensor_dim[id] / kValuesPerTaxel;

  mjtNum maxmag = 0;
  for (int k = 0; k < ntaxel; ++k) {
    maxmag = mju_max(maxmag, sensordata[kValuesPerTaxel*k + 6]);
  }

  int site_id = m->sensor_objid[id];
  mjtNum* site_pos = d->site_xpos + 3*site_id;
  mjtNum* site_mat = d->site_xmat + 9*site_id;

  mjtNum* a = attribute;
  int nx, ny;
  GridSize(a, &nx, &ny);
  mjtNum hx, hy;
  ShapeTileSize(a, nx, ny, &hx, &hy);
  mjtNum thick = kRelativeThickness * mju_min(hx, hy);
  mjtNum bsize[3] = {hx, hy, thick};
  mjtNum arrow_len = ShapeSize(a);
  mjtNum arrow_w = 0.3 * mju_min(hx, hy);

  for (int k = 0; k < ntaxel; ++k) {
    const mjtNum* psite = sensordata + kValuesPerTaxel*k;
    mjtNum mag = psite[6];
    mjtNum scale = maxmag > 0 ? mag / maxmag : 0;

    // Taxel center + normal -> world frame.
    mjtNum tpos[3], tnrm[3];
    mju_mulMatVec3(tpos, site_mat, psite);
    mju_addTo3(tpos, site_pos);
    mju_mulMatVec3(tnrm, site_mat, psite + 3);

    // Tile tangent to the surface (z-axis = normal), lifted off the surface.
    mjtNum mat[9];
    FrameFromNormal(tnrm, mat);
    mjtNum bpos[3];
    mju_addScl3(bpos, tpos, tnrm, thick);

    float rgba[4];
    Heatmap(scale, rgba);
    rgba[3] = static_cast<float>(0.15 + 0.55*scale);

    mjvGeom* g = NextGeom(scn);
    if (!g) return;
    mjv_initGeom(g, mjGEOM_BOX, bsize, bpos, mat, rgba);
    TagDecor(g, instance, scn->ngeom);
    scn->ngeom++;

    // Force arrow along the surface normal, length scaled by magnitude.
    if (mag <= 0) continue;
    mjtNum to[3];
    mju_addScl3(to, tpos, tnrm, scale*arrow_len);
    mjvGeom* arrow = NextGeom(scn);
    if (!arrow) return;
    float arrgba[4] = {rgba[0], rgba[1], rgba[2], 1.0f};
    mjv_initGeom(arrow, mjGEOM_NONE, NULL, NULL, NULL, arrgba);
    TagDecor(arrow, instance, scn->ngeom);
    mjv_connector(arrow, mjGEOM_ARROW, arrow_w, tpos, to);
    scn->ngeom++;
  }
}

// plugin registration
void EFlesh::RegisterPlugin() {
  mjpPlugin plugin;
  mjp_defaultPlugin(&plugin);

  plugin.name = "mujoco.sdf.eflesh";
  // Dual capability: signed-distance shape + tactile sensor.
  plugin.capabilityflags |= mjPLUGIN_SDF | mjPLUGIN_SENSOR;

  plugin.nattribute = EFleshAttribute::nattribute;
  plugin.attributes = EFleshAttribute::names;
  plugin.nstate = +[](const mjModel* m, int instance) { return 0; };

  // Sensor dimension = 7 * nx * ny (queried only for sensor-bound instances).
  plugin.nsensordata = +[](const mjModel* m, int instance, int sensor_id) {
    const char* xval = mj_getPluginConfig(m, instance, "nx");
    const char* yval = mj_getPluginConfig(m, instance, "ny");
    int nx = (xval && xval[0]) ? static_cast<int>(strtod(xval, nullptr))
                               : static_cast<int>(EFleshAttribute::defaults[kNx]);
    int ny = (yval && yval[0]) ? static_cast<int>(strtod(yval, nullptr))
                               : static_cast<int>(EFleshAttribute::defaults[kNy]);
    if (nx < 1) nx = 1;
    if (ny < 1) ny = 1;
    return kValuesPerTaxel * nx * ny;
  };

  // The tactile sensor needs contact forces, available at the acceleration stage.
  plugin.needstage = mjSTAGE_ACC;

  plugin.init = +[](const mjModel* m, mjData* d, int instance) {
    auto sdf_or_null = EFlesh::Create(m, d, instance);
    if (!sdf_or_null.has_value()) {
      return -1;
    }
    d->plugin_data[instance] = reinterpret_cast<uintptr_t>(
        new EFlesh(std::move(*sdf_or_null)));
    return 0;
  };
  plugin.destroy = +[](mjData* d, int instance) {
    delete reinterpret_cast<EFlesh*>(d->plugin_data[instance]);
    d->plugin_data[instance] = 0;
  };
  plugin.reset = +[](const mjModel* m, mjtNum* plugin_state, void* plugin_data,
                     int instance) {
    auto* eflesh = reinterpret_cast<EFlesh*>(plugin_data);
    eflesh->Reset();
  };
  plugin.compute =
      +[](const mjModel* m, mjData* d, int instance, int capability_bit) {
        auto* eflesh = reinterpret_cast<EFlesh*>(d->plugin_data[instance]);
        eflesh->Compute(m, d, instance);
      };
  plugin.visualize = +[](const mjModel* m, mjData* d, const mjvOption* opt,
                         mjvScene* scn, int instance) {
    auto* eflesh = reinterpret_cast<EFlesh*>(d->plugin_data[instance]);
    eflesh->Visualize(m, d, opt, scn, instance);
  };
  plugin.sdf_distance =
      +[](const mjtNum point[3], const mjData* d, int instance) {
        auto* sdf = reinterpret_cast<EFlesh*>(d->plugin_data[instance]);
        return sdf->Distance(point);
      };
  plugin.sdf_gradient = +[](mjtNum gradient[3], const mjtNum point[3],
                        const mjData* d, int instance) {
    auto* sdf = reinterpret_cast<EFlesh*>(d->plugin_data[instance]);
    sdf->Gradient(gradient, point);
  };
  plugin.sdf_staticdistance =
      +[](const mjtNum point[3], const mjtNum* attributes) {
        return ShapeDistance(point, attributes);
      };
  plugin.sdf_aabb =
      +[](mjtNum aabb[6], const mjtNum* attributes) {
        ShapeAabb(aabb, attributes);
      };
  plugin.sdf_attribute =
      +[](mjtNum attribute[], const char* name[], const char* value[]) {
        SdfDefault<EFleshAttribute> defattribute;
        defattribute.GetDefaults(attribute, name, value);
      };

  mjp_registerPlugin(&plugin);
}

}  // namespace mujoco::plugin::sdf
