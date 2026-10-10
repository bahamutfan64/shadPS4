// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>

#include "common/types.h"
#include "core/libraries/system/commondialog.h"

namespace Core::Loader {
class SymbolsResolver;
}
namespace Libraries::ErrorDialog {

using OrbisUserServiceUserId = s32;

struct Param;

/// Set when the "signed out of PSN" error dialog was skipped, so that the sign-in retry
/// question a game shows right after it can be answered "No" without user input.
extern std::atomic_bool g_skipped_np_signed_out_dialog;

CommonDialog::Error PS4_SYSV_ABI sceErrorDialogClose();
CommonDialog::Status PS4_SYSV_ABI sceErrorDialogGetStatus();
CommonDialog::Error PS4_SYSV_ABI sceErrorDialogInitialize();
CommonDialog::Error PS4_SYSV_ABI sceErrorDialogOpen(const Param* param);
int PS4_SYSV_ABI sceErrorDialogOpenDetail();
int PS4_SYSV_ABI sceErrorDialogOpenWithReport();
CommonDialog::Error PS4_SYSV_ABI sceErrorDialogTerminate();
CommonDialog::Status PS4_SYSV_ABI sceErrorDialogUpdateStatus();

void RegisterLib(Core::Loader::SymbolsResolver* sym);
} // namespace Libraries::ErrorDialog