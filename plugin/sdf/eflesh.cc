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

#include <cstdint>
#include <optional>
#include <utility>
#include <algorithm>

#include <mujoco/mjplugin.h>
#include <mujoco/mjtype.h>
#include <mujoco/mujoco.h>
#include "sdf.h"
#include "eflesh.h"

namespace mujoco::plugin::sdf {
namespace {

static mjtNum distance(const mjtNum p[3], const mjtNum b[3]) {
  mjtNum dx = mju_abs(p[0]) - b[0];
  mjtNum dy = mju_abs(p[1]) - b[1];
  mjtNum dz = mju_abs(p[2]) - b[2];
  
  mjtNum qx = mju_max(dx, 0.0);
  mjtNum qy = mju_max(dy, 0.0);
  mjtNum qz = mju_max(dz, 0.0);
  
  return mju_sqrt(qx*qx + qy*qy + qz*qz) + mju_min(mju_max(dx, mju_max(dy, dz)), 0.0);
}

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

  for (int i=0; i < EFleshAttribute::nattribute; i++) {
    attribute[i] = defattribute.GetDefault(
        EFleshAttribute::names[i],
        mj_getPluginConfig(m, instance, EFleshAttribute::names[i]));
  }
}

// sdf
mjtNum EFlesh::Distance(const mjtNum point[3]) const {
  return distance(point, attribute);
}

// gradient of sdf
void EFlesh::Gradient(mjtNum grad[3], const mjtNum p[3]) const {
  mjtNum dx = mju_abs(p[0]) - attribute[0];
  mjtNum dy = mju_abs(p[1]) - attribute[1];
  mjtNum dz = mju_abs(p[2]) - attribute[2];
  
  mjtNum qx = mju_max(dx, 0.0);
  mjtNum qy = mju_max(dy, 0.0);
  mjtNum qz = mju_max(dz, 0.0);
  
  mjtNum len_q = mju_sqrt(qx*qx + qy*qy + qz*qz);
  
  if (len_q > 0) {
    grad[0] = (p[0] >= 0 ? 1.0 : -1.0) * qx / len_q;
    grad[1] = (p[1] >= 0 ? 1.0 : -1.0) * qy / len_q;
    grad[2] = (p[2] >= 0 ? 1.0 : -1.0) * qz / len_q;
  } else {
    if (dx > dy && dx > dz) {
      grad[0] = (p[0] >= 0 ? 1.0 : -1.0);
      grad[1] = 0;
      grad[2] = 0;
    } else if (dy > dz) {
      grad[0] = 0;
      grad[1] = (p[1] >= 0 ? 1.0 : -1.0);
      grad[2] = 0;
    } else {
      grad[0] = 0;
      grad[1] = 0;
      grad[2] = (p[2] >= 0 ? 1.0 : -1.0);
    }
  }
}

// plugin registration
void EFlesh::RegisterPlugin() {
  mjpPlugin plugin;
  mjp_defaultPlugin(&plugin);

  plugin.name = "mujoco.sdf.eflesh";
  plugin.capabilityflags |= mjPLUGIN_SDF;

  plugin.nattribute = EFleshAttribute::nattribute;
  plugin.attributes = EFleshAttribute::names;
  plugin.nstate = +[](const mjModel* m, int instance) { return 0; };

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
    // do nothing;
  };
  plugin.compute =
      +[](const mjModel* m, mjData* d, int instance, int capability_bit) {
        // do nothing;
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
        return distance(point, attributes);
      };
  plugin.sdf_aabb =
      +[](mjtNum aabb[6], const mjtNum* attributes) {
        aabb[0] = aabb[1] = aabb[2] = 0;
        aabb[3] = attributes[0];
        aabb[4] = attributes[1];
        aabb[5] = attributes[2];
      };
  plugin.sdf_attribute =
      +[](mjtNum attribute[], const char* name[], const char* value[]) {
        SdfDefault<EFleshAttribute> defattribute;
        defattribute.GetDefaults(attribute, name, value);
      };

  mjp_registerPlugin(&plugin);
}

}  // namespace mujoco::plugin::sdf
