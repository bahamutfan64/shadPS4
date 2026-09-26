// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <map>
#include <span>
#include <utility>
#include <vector>

#include "common/types.h"
#include "shader_recompiler/ir/attribute.h"

namespace Shader {

struct CopyShaderData {
    // A single GSVS ring slot can be exported to more than one target. For example, a layer
    // index is commonly exported both to POS1 (render target index) and to a PARAM so the
    // pixel shader can read it. Keep every target so none of them is lost.
    std::map<u32, std::vector<std::pair<Shader::IR::Attribute, u32>>> attr_map;
    u32 num_attrs{0};
    u32 output_vertices{0};
    u32 num_comps{0};
};

CopyShaderData ParseCopyShader(std::span<const u32> code);

} // namespace Shader
