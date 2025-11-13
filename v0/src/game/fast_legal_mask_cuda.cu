#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <tuple>

#include "fast_legal_mask_common.hpp"
#include "v0/game_state.hpp"

namespace v0 {
namespace {

constexpr int kMaxCells = kCellCount;

struct IndexList {
    int count;
    int data[kMaxCells];

    __device__ __forceinline__ void Clear() {
        count = 0;
    }

    __device__ __forceinline__ void Push(int value) {
        if (count < kMaxCells) {
            data[count++] = value;
        }
    }

    __device__ __forceinline__ void CopyFrom(const IndexList& other) {
        count = other.count;
        for (int i = 0; i < count; ++i) {
            data[i] = other.data[i];
        }
    }
};

__device__ __forceinline__ bool DeviceIsMarked(const bool* marked, int idx) {
    return marked != nullptr && marked[idx];
}

__device__ __forceinline__ void SetMetadata(
    int32_t* meta_state,
    int64_t local_index,
    ActionKind kind,
    int32_t primary,
    int32_t secondary,
    int32_t extra) {
    const int64_t offset = local_index * kMetaFields;
    meta_state[offset + 0] = static_cast<int32_t>(kind);
    meta_state[offset + 1] = primary;
    meta_state[offset + 2] = secondary;
    meta_state[offset + 3] = extra;
}

__device__ bool CheckSquares(
    const int8_t* board,
    const bool* marked,
    int size,
    int r,
    int c,
    int player_value) {
    const int offsets[2] = {0, -1};
    for (int oi = 0; oi < 2; ++oi) {
        for (int oj = 0; oj < 2; ++oj) {
            const int rr = r + offsets[oi];
            const int cc = c + offsets[oj];
            if (rr >= 0 && rr < size - 1 && cc >= 0 && cc < size - 1) {
                bool ok = true;
                const int cells[4][2] = {{rr, cc}, {rr, cc + 1}, {rr + 1, cc}, {rr + 1, cc + 1}};
                for (int k = 0; k < 4; ++k) {
                    const int idx = flat_index(cells[k][0], cells[k][1], size);
                    if (board[idx] != player_value || DeviceIsMarked(marked, idx)) {
                        ok = false;
                        break;
                    }
                }
                if (ok) {
                    return true;
                }
            }
        }
    }
    return false;
}

__device__ bool CheckLines(
    const int8_t* board,
    const bool* marked,
    int size,
    int r,
    int c,
    int player_value) {
    int count = 1;
    for (int dc = c - 1; dc >= 0; --dc) {
        const int idx = flat_index(r, dc, size);
        if (board[idx] == player_value && !DeviceIsMarked(marked, idx)) {
            ++count;
        } else {
            break;
        }
    }
    for (int dc = c + 1; dc < size; ++dc) {
        const int idx = flat_index(r, dc, size);
        if (board[idx] == player_value && !DeviceIsMarked(marked, idx)) {
            ++count;
        } else {
            break;
        }
    }
    if (count >= 6) {
        return true;
    }

    count = 1;
    for (int dr = r - 1; dr >= 0; --dr) {
        const int idx = flat_index(dr, c, size);
        if (board[idx] == player_value && !DeviceIsMarked(marked, idx)) {
            ++count;
        } else {
            break;
        }
    }
    for (int dr = r + 1; dr < size; ++dr) {
        const int idx = flat_index(dr, c, size);
        if (board[idx] == player_value && !DeviceIsMarked(marked, idx)) {
            ++count;
        } else {
            break;
        }
    }
    return count >= 6;
}

__device__ bool IsPieceInShape(
    const int8_t* board,
    int size,
    int r,
    int c,
    int player_value,
    const bool* marked) {
    if (board[flat_index(r, c, size)] != player_value) {
        return false;
    }
    return CheckSquares(board, marked, size, r, c, player_value) ||
           CheckLines(board, marked, size, r, c, player_value);
}

__device__ void PreferNormalPieces(
    const int8_t* board,
    const bool* marked,
    int size,
    const IndexList& candidates,
    int player_value,
    IndexList* out) {
    out->Clear();
    for (int i = 0; i < candidates.count; ++i) {
        const int idx = candidates.data[i];
        const int r = idx / size;
        const int c = idx % size;
        if (!IsPieceInShape(board, size, r, c, player_value, marked)) {
            out->Push(idx);
        }
    }
    if (out->count == 0) {
        out->CopyFrom(candidates);
    }
}

__device__ void GenerateMarkTargets(
    const int8_t* board,
    const bool* opponent_marked,
    int size,
    int pending_mark_rem,
    int current_value,
    IndexList* out) {
    const int opponent_value = -current_value;
    IndexList candidates;
    candidates.Clear();
    for (int idx = 0; idx < size * size; ++idx) {
        if (board[idx] == opponent_value && !DeviceIsMarked(opponent_marked, idx)) {
            candidates.Push(idx);
        }
    }
    IndexList preferred;
    preferred.Clear();
    PreferNormalPieces(board, opponent_marked, size, candidates, opponent_value, &preferred);
    if (pending_mark_rem > 0) {
        out->CopyFrom(preferred);
    } else {
        out->Clear();
    }
}

__device__ void GenerateCaptureTargets(
    const int8_t* board,
    const bool* opponent_marked,
    int size,
    int pending_capture_rem,
    int current_value,
    IndexList* out) {
    const int opponent_value = -current_value;
    IndexList candidates;
    candidates.Clear();
    for (int idx = 0; idx < size * size; ++idx) {
        if (board[idx] == opponent_value && DeviceIsMarked(opponent_marked, idx)) {
            candidates.Push(idx);
        }
    }
    if (pending_capture_rem > 0) {
        IndexList preferred;
        preferred.Clear();
        PreferNormalPieces(board, opponent_marked, size, candidates, opponent_value, &preferred);
        out->CopyFrom(preferred);
    } else {
        out->Clear();
    }
}

__device__ void ForcedRemovalTargets(
    const int8_t* board,
    int size,
    int forced_done,
    IndexList* out) {
    out->Clear();
    if (forced_done >= 2) {
        return;
    }
    const int value = forced_done == 0 ? kPlayerBlack : kPlayerWhite;
    IndexList candidates;
    candidates.Clear();
    for (int idx = 0; idx < size * size; ++idx) {
        if (board[idx] == value) {
            candidates.Push(idx);
        }
    }
    IndexList preferred;
    preferred.Clear();
    PreferNormalPieces(board, nullptr, size, candidates, value, &preferred);
    out->CopyFrom(preferred);
}

__device__ void CounterRemovalTargets(
    const int8_t* board,
    int size,
    int current_value,
    IndexList* out) {
    const int player_value = -current_value;
    IndexList candidates;
    candidates.Clear();
    for (int idx = 0; idx < size * size; ++idx) {
        if (board[idx] == player_value) {
            candidates.Push(idx);
        }
    }
    IndexList preferred;
    preferred.Clear();
    PreferNormalPieces(board, nullptr, size, candidates, player_value, &preferred);
    out->CopyFrom(preferred);
}

__device__ void EmitSelection(
    const IndexList& indices,
    ActionKind kind,
    int64_t selection_dim,
    int64_t selection_offset,
    bool* mask_state,
    int32_t* meta_state) {
    if (selection_dim <= 0) {
        return;
    }
    for (int i = 0; i < indices.count; ++i) {
        const int idx = indices.data[i];
        if (idx >= 0 && idx < selection_dim) {
            const int64_t local_index = selection_offset + idx;
            mask_state[local_index] = true;
            SetMetadata(meta_state, local_index, kind, idx, -1, -1);
        }
    }
}

__global__ void EncodeActionsKernel(
    const int8_t* board,
    const bool* marks_black,
    const bool* marks_white,
    const int64_t* phase,
    const int64_t* current_player,
    const int64_t* pending_marks_remaining,
    const int64_t* pending_captures_remaining,
    const int64_t* forced_removals_done,
    int64_t batch_size,
    int size,
    int cell_count,
    int64_t placement_dim,
    int64_t movement_dim,
    int64_t selection_dim,
    int64_t auxiliary_dim,
    int64_t total_dim,
    bool* mask,
    int32_t* metadata) {
    const int64_t b = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (b >= batch_size) {
        return;
    }

    const int8_t* board_state = board + b * cell_count;
    const bool* marks_black_state = marks_black + b * cell_count;
    const bool* marks_white_state = marks_white + b * cell_count;
    const int phase_val = static_cast<int>(phase[b]);
    const int current_val = static_cast<int>(current_player[b]);
    const int pending_mark_rem = static_cast<int>(pending_marks_remaining[b]);
    const int pending_capture_rem = static_cast<int>(pending_captures_remaining[b]);
    const int forced_done = static_cast<int>(forced_removals_done[b]);

    bool* mask_state = mask + b * total_dim;
    int32_t* meta_state = metadata + b * total_dim * kMetaFields;

    if (phase_val == kPhasePlacement) {
        for (int idx = 0; idx < cell_count; ++idx) {
            if (board_state[idx] == 0) {
                mask_state[idx] = true;
                SetMetadata(meta_state, idx, kActionPlacement, idx, -1, -1);
            }
        }
    }

    bool has_movement = false;
    if (phase_val == kPhaseMovement) {
        const Directions dirs[4] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
        for (int r = 0; r < size; ++r) {
            for (int c = 0; c < size; ++c) {
                const int base_idx = flat_index(r, c, size);
                if (board_state[base_idx] != current_val) {
                    continue;
                }
                for (int dir = 0; dir < 4; ++dir) {
                    const int nr = r + dirs[dir].dr;
                    const int nc = c + dirs[dir].dc;
                    if (nr >= 0 && nr < size && nc >= 0 && nc < size) {
                        const int dest_idx = flat_index(nr, nc, size);
                        if (board_state[dest_idx] == 0) {
                            const int move_offset = base_idx * 4 + dir;
                            const int64_t local_index = placement_dim + move_offset;
                            if (local_index < placement_dim + movement_dim) {
                                mask_state[local_index] = true;
                                SetMetadata(
                                    meta_state,
                                    local_index,
                                    kActionMovement,
                                    base_idx,
                                    dir,
                                    dest_idx);
                                has_movement = true;
                            }
                        }
                    }
                }
            }
        }
    }

    const int64_t selection_offset = placement_dim + movement_dim;
    IndexList selection;
    selection.Clear();

    if (phase_val == kPhaseMarkSelection) {
        const bool* opponent_marked = current_val == kPlayerBlack ? marks_white_state : marks_black_state;
        GenerateMarkTargets(
            board_state,
            opponent_marked,
            size,
            pending_mark_rem,
            current_val,
            &selection);
        EmitSelection(selection, kActionMarkSelection, selection_dim, selection_offset, mask_state, meta_state);
    } else if (phase_val == kPhaseCaptureSelection) {
        const bool* opponent_marked = current_val == kPlayerBlack ? marks_white_state : marks_black_state;
        GenerateCaptureTargets(
            board_state,
            opponent_marked,
            size,
            pending_capture_rem,
            current_val,
            &selection);
        EmitSelection(selection, kActionCaptureSelection, selection_dim, selection_offset, mask_state, meta_state);
    } else if (phase_val == kPhaseForcedRemoval) {
        ForcedRemovalTargets(board_state, size, forced_done, &selection);
        EmitSelection(selection, kActionForcedRemovalSelection, selection_dim, selection_offset, mask_state, meta_state);
    } else if (phase_val == kPhaseCounterRemoval) {
        CounterRemovalTargets(board_state, size, current_val, &selection);
        EmitSelection(selection, kActionCounterRemovalSelection, selection_dim, selection_offset, mask_state, meta_state);
    } else if (phase_val == kPhaseMovement && !has_movement) {
        CounterRemovalTargets(board_state, size, current_val, &selection);
        EmitSelection(selection, kActionNoMovesRemovalSelection, selection_dim, selection_offset, mask_state, meta_state);
    }

    if (phase_val == kPhaseRemoval && auxiliary_dim > 0) {
        const int64_t removal_index = placement_dim + movement_dim + selection_dim;
        if (removal_index < total_dim) {
            mask_state[removal_index] = true;
            SetMetadata(meta_state, removal_index, kActionProcessRemoval, -1, -1, -1);
        }
    }
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor> encode_actions_fast_cuda(
    torch::Tensor board,
    torch::Tensor marks_black,
    torch::Tensor marks_white,
    torch::Tensor phase,
    torch::Tensor current_player,
    torch::Tensor pending_marks_required,
    torch::Tensor pending_marks_remaining,
    torch::Tensor pending_captures_required,
    torch::Tensor pending_captures_remaining,
    torch::Tensor forced_removals_done,
    int64_t placement_dim,
    int64_t movement_dim,
    int64_t selection_dim,
    int64_t auxiliary_dim) {
    TORCH_CHECK(board.device().is_cuda(), "encode_actions_fast_cuda requires CUDA tensors.");

    at::cuda::OptionalCUDAGuard guard(board.device());

    (void)pending_marks_required;
    (void)pending_captures_required;

    board = board.contiguous();
    marks_black = marks_black.contiguous();
    marks_white = marks_white.contiguous();
    phase = phase.contiguous();
    current_player = current_player.contiguous();
    pending_marks_remaining = pending_marks_remaining.contiguous();
    pending_captures_remaining = pending_captures_remaining.contiguous();
    forced_removals_done = forced_removals_done.contiguous();

    const auto B = board.size(0);
    const auto H = board.size(1);
    const auto W = board.size(2);
    TORCH_CHECK(H == W, "Board must be square.");
    TORCH_CHECK(
        H <= kBoardSize,
        "encode_actions_fast_cuda only supports board size <= ",
        kBoardSize);

    const int size = static_cast<int>(H);
    const int cell_count = size * size;
    const int64_t total_dim = placement_dim + movement_dim + selection_dim + auxiliary_dim;

    auto mask_options = torch::TensorOptions().dtype(torch::kBool).device(board.device());
    auto meta_options = torch::TensorOptions().dtype(torch::kInt32).device(board.device());
    auto mask = torch::zeros({B, total_dim}, mask_options);
    auto metadata = torch::full({B, total_dim, kMetaFields}, -1, meta_options);

    if (B == 0 || total_dim == 0) {
        return {mask, metadata};
    }

    const int threads = 128;
    const int blocks = static_cast<int>((B + threads - 1) / threads);
    EncodeActionsKernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        board.data_ptr<int8_t>(),
        marks_black.data_ptr<bool>(),
        marks_white.data_ptr<bool>(),
        phase.data_ptr<int64_t>(),
        current_player.data_ptr<int64_t>(),
        pending_marks_remaining.data_ptr<int64_t>(),
        pending_captures_remaining.data_ptr<int64_t>(),
        forced_removals_done.data_ptr<int64_t>(),
        B,
        size,
        cell_count,
        placement_dim,
        movement_dim,
        selection_dim,
        auxiliary_dim,
        total_dim,
        mask.data_ptr<bool>(),
        metadata.data_ptr<int32_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {mask, metadata};
}

}  // namespace v0
