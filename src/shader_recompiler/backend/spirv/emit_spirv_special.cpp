// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "shader_recompiler/backend/spirv/emit_spirv_instructions.h"
#include "shader_recompiler/backend/spirv/spirv_emit_context.h"
#include "shader_recompiler/ir/debug_print.h"
#include "shader_recompiler/ir/microinstruction.h"

namespace Shader::Backend::SPIRV {

void EmitPrologue(EmitContext& ctx) {
    if (ctx.hw_stage == HwStage::Fragment) {
        ctx.DefineAmdPerVertexAttribs();
    }
    if (ctx.info.loads.Get(IR::Attribute::WorkgroupIndex)) {
        ctx.DefineWorkgroupIndex();
    }
    ctx.DefineBufferProperties();
}

void ConvertDepthMode(EmitContext& ctx) {
    const Id type{ctx.F32[1]};
    const Id position{ctx.OpLoad(ctx.F32[4], ctx.output_position)};
    const Id z{ctx.OpCompositeExtract(type, position, 2u)};
    const Id w{ctx.OpCompositeExtract(type, position, 3u)};
    const Id screen_depth{ctx.OpFMul(type, ctx.OpFAdd(type, z, w), ctx.Constant(type, 0.5f))};
    const Id vector{ctx.OpCompositeInsert(ctx.F32[4], screen_depth, position, 2u)};
    ctx.OpStore(ctx.output_position, vector);
}

void ConvertPositionToClipSpace(EmitContext& ctx) {
    ASSERT_MSG(!ctx.info.stores.GetAny(IR::Attribute::ViewportIndex),
               "Multi-viewport with shader clip space conversion not yet implemented.");

    const Id type{ctx.F32[1]};
    Id position{ctx.OpLoad(ctx.F32[4], ctx.output_position)};
    const Id x{ctx.OpCompositeExtract(type, position, 0u)};
    const Id y{ctx.OpCompositeExtract(type, position, 1u)};
    const Id z{ctx.OpCompositeExtract(type, position, 2u)};
    const Id w{ctx.OpCompositeExtract(type, position, 3u)};
    const Id xoffset_ptr{ctx.OpAccessChain(ctx.TypePointer(spv::StorageClass::PushConstant, type),
                                           ctx.push_data_block,
                                           ctx.ConstU32(PushData::XOffsetIndex))};
    const Id xoffset{ctx.OpLoad(type, xoffset_ptr)};
    const Id yoffset_ptr{ctx.OpAccessChain(ctx.TypePointer(spv::StorageClass::PushConstant, type),
                                           ctx.push_data_block,
                                           ctx.ConstU32(PushData::YOffsetIndex))};
    const Id yoffset{ctx.OpLoad(type, yoffset_ptr)};
    const Id xscale_ptr{ctx.OpAccessChain(ctx.TypePointer(spv::StorageClass::PushConstant, type),
                                          ctx.push_data_block,
                                          ctx.ConstU32(PushData::XScaleIndex))};
    const Id xscale{ctx.OpLoad(type, xscale_ptr)};
    const Id yscale_ptr{ctx.OpAccessChain(ctx.TypePointer(spv::StorageClass::PushConstant, type),
                                          ctx.push_data_block,
                                          ctx.ConstU32(PushData::YScaleIndex))};
    const Id yscale{ctx.OpLoad(type, yscale_ptr)};
    const Id vport_w =
        ctx.Constant(type, float(std::min<u32>(ctx.profile.max_viewport_width / 2, 8_KB)));
    const Id wnd_x = ctx.OpFAdd(type, ctx.OpFMul(type, x, xscale), xoffset);
    const Id ndc_x = ctx.OpFSub(type, ctx.OpFDiv(type, wnd_x, vport_w), ctx.Constant(type, 1.f));
    const Id vport_h =
        ctx.Constant(type, float(std::min<u32>(ctx.profile.max_viewport_height / 2, 8_KB)));
    const Id wnd_y = ctx.OpFAdd(type, ctx.OpFMul(type, y, yscale), yoffset);
    const Id ndc_y = ctx.OpFSub(type, ctx.OpFDiv(type, wnd_y, vport_h), ctx.Constant(type, 1.f));
    const Id vector{ctx.OpCompositeConstruct(ctx.F32[4], std::array<Id, 4>({ndc_x, ndc_y, z, w}))};
    ctx.OpStore(ctx.output_position, vector);
}

void EmitEpilogue(EmitContext& ctx) {
    if (ctx.hw_stage == HwStage::Vertex &&
        ctx.runtime_info.hw.vs.emulate_depth_negative_one_to_one) {
        ConvertDepthMode(ctx);
    }
    if (ctx.hw_stage == HwStage::Vertex && ctx.runtime_info.hw.vs.clip_disable) {
        ConvertPositionToClipSpace(ctx);
    }
}

void EmitDiscard(EmitContext& ctx) {
    ctx.OpDemoteToHelperInvocationEXT();
}

void EmitDiscardCond(EmitContext& ctx, Id condition) {
    const Id kill_label{ctx.OpLabel()};
    const Id merge_label{ctx.OpLabel()};
    ctx.OpSelectionMerge(merge_label, spv::SelectionControlMask::MaskNone);
    ctx.OpBranchConditional(condition, kill_label, merge_label);
    ctx.AddLabel(kill_label);
    ctx.OpDemoteToHelperInvocationEXT();
    ctx.OpBranch(merge_label);
    ctx.AddLabel(merge_label);
}

void EmitEmitVertex(EmitContext& ctx) {
    if (ctx.gs_split_invocations == 0) {
        ctx.OpEmitVertex();
        return;
    }
    // This invocation owns vertices [lo, hi) of the stream. It also emits the two vertices
    // before lo so its first triangle joins the previous window, and a third one when that is
    // needed to start on an even strip index (which keeps the winding). The previous invocation
    // then drops its last vertex so the shared triangle is not drawn twice.
    const Id u32{ctx.U32[1]};
    const Id u1{ctx.U1[1]};
    const Id count{ctx.OpLoad(u32, ctx.gs_vertex_count)};
    const Id strip{ctx.OpLoad(u32, ctx.gs_strip_index)};
    const Id invocation{ctx.OpLoad(u32, ctx.invocation_id)};
    const Id lo{ctx.OpIMul(u32, invocation, ctx.ConstU32(ctx.gs_split_window))};
    const Id hi{ctx.OpIAdd(u32, lo, ctx.ConstU32(ctx.gs_split_window))};
    const Id strip_even{
        ctx.OpIEqual(u1, ctx.OpBitwiseAnd(u32, strip, ctx.ConstU32(1u)), ctx.ConstU32(0u))};
    const Id joins{ctx.OpUGreaterThanEqual(u1, ctx.OpIAdd(u32, count, ctx.ConstU32(2u)), lo)};
    const Id fixes_parity{ctx.OpLogicalAnd(
        u1, ctx.OpIEqual(u1, ctx.OpIAdd(u32, count, ctx.ConstU32(3u)), lo), strip_even)};
    const Id started{ctx.OpLogicalOr(u1, joins, fixes_parity)};
    const Id is_last_of_window{ctx.OpIEqual(u1, ctx.OpIAdd(u32, count, ctx.ConstU32(1u)), hi)};
    const Id has_next{ctx.OpULessThan(u1, ctx.OpIAdd(u32, invocation, ctx.ConstU32(1u)),
                                      ctx.ConstU32(ctx.gs_split_invocations))};
    const Id next_takes_it{ctx.OpLogicalAnd(
        u1, ctx.OpLogicalAnd(u1, is_last_of_window, has_next),
        ctx.OpLogicalAnd(u1, strip_even, ctx.OpUGreaterThanEqual(u1, strip, ctx.ConstU32(2u))))};
    const Id in_window{ctx.OpLogicalAnd(u1, ctx.OpULessThan(u1, count, hi),
                                        ctx.OpLogicalNot(u1, next_takes_it))};
    const Id emit{ctx.OpLogicalAnd(u1, started, in_window)};
    ctx.OpStore(ctx.gs_vertex_count, ctx.OpIAdd(u32, count, ctx.ConstU32(1u)));
    ctx.OpStore(ctx.gs_strip_index, ctx.OpIAdd(u32, strip, ctx.ConstU32(1u)));
    const Id emit_label{ctx.OpLabel()};
    const Id merge_label{ctx.OpLabel()};
    ctx.OpSelectionMerge(merge_label, spv::SelectionControlMask::MaskNone);
    ctx.OpBranchConditional(emit, emit_label, merge_label);
    ctx.AddLabel(emit_label);
    ctx.OpEmitVertex();
    ctx.OpBranch(merge_label);
    ctx.AddLabel(merge_label);
}

void EmitEmitPrimitive(EmitContext& ctx) {
    ctx.OpEndPrimitive();
    if (ctx.gs_split_invocations != 0) {
        ctx.OpStore(ctx.gs_strip_index, ctx.ConstU32(0u));
    }
}

void EmitEmitVertex(EmitContext& ctx, const IR::Value& stream) {
    UNREACHABLE_MSG("Geometry streams");
}

void EmitEndPrimitive(EmitContext& ctx, const IR::Value& stream) {
    UNREACHABLE_MSG("Geometry streams");
}

void EmitDebugPrint(EmitContext& ctx, IR::Inst* inst, Id fmt, Id arg0, Id arg1, Id arg2, Id arg3) {
    IR::DebugPrintFlags flags = inst->Flags<IR::DebugPrintFlags>();
    std::array<Id, IR::DEBUGPRINT_NUM_FORMAT_ARGS> fmt_args = {arg0, arg1, arg2, arg3};
    auto fmt_args_span = std::span<Id>(fmt_args.begin(), fmt_args.begin() + flags.num_args);
    ctx.OpDebugPrintf(fmt, fmt_args_span);
}

Id EmitMemtime(EmitContext& ctx) {
    if (ctx.profile.supports_shader_subgroup_clock) {
        return ctx.OpReadClockKHR(ctx.U64, ctx.ConstU32(std::to_underlying(spv::Scope::Subgroup)));
    } else {
        return ctx.Constant(ctx.U64, 1U);
    }
}

} // namespace Shader::Backend::SPIRV
