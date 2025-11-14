#include <torch/extension.h>

#include <algorithm>
#include <cstdint>
#include <vector>

#include "fast_legal_mask_common.hpp"

namespace v0 {

namespace {

inline bool is_marked(const bool* marked, int idx) {
    return marked != nullptr && marked[idx];
}

bool check_squares(
    const int8_t* board,
    const bool* marked,
    int size,
    int r,
    int c,
    int player_value) {
    static const int offsets[2] = {0, -1};
    for (int dr : offsets) {
        for (int dc : offsets) {
            int rr = r + dr;
            int cc = c + dc;
            if (rr >= 0 && rr < size - 1 && cc >= 0 && cc < size - 1) {
                bool ok = true;
                const int cells[4][2] = {{rr, cc}, {rr, cc + 1}, {rr + 1, cc}, {rr + 1, cc + 1}};
                for (const auto& cell : cells) {
                    int idx = flat_index(cell[0], cell[1], size);
                    if (board[idx] != player_value || is_marked(marked, idx)) {
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

bool check_lines(
    const int8_t* board,
    const bool* marked,
    int size,
    int r,
    int c,
    int player_value) {
    int count = 1;
    for (int dc = c - 1; dc >= 0; --dc) {
        int idx = flat_index(r, dc, size);
        if (board[idx] == player_value && !is_marked(marked, idx)) {
            ++count;
        } else {
            break;
        }
    }
    for (int dc = c + 1; dc < size; ++dc) {
        int idx = flat_index(r, dc, size);
        if (board[idx] == player_value && !is_marked(marked, idx)) {
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
        int idx = flat_index(dr, c, size);
        if (board[idx] == player_value && !is_marked(marked, idx)) {
            ++count;
        } else {
            break;
        }
    }
    for (int dr = r + 1; dr < size; ++dr) {
        int idx = flat_index(dr, c, size);
        if (board[idx] == player_value && !is_marked(marked, idx)) {
            ++count;
        } else {
            break;
        }
    }
    return count >= 6;
}

bool is_piece_in_shape(
    const int8_t* board,
    int size,
    int r,
    int c,
    int player_value,
    const bool* marked) {
    if (board[flat_index(r, c, size)] != player_value) {
        return false;
    }
    return check_squares(board, marked, size, r, c, player_value) ||
           check_lines(board, marked, size, r, c, player_value);
}

std::vector<int> prefer_normal_pieces(
    const int8_t* board,
    const bool* marked,
    int size,
    const std::vector<int>& candidates,
    int player_value) {
    std::vector<int> normal;
    normal.reserve(candidates.size());
    for (int idx : candidates) {
        int r = idx / size;
        int c = idx % size;
        if (!is_piece_in_shape(board, size, r, c, player_value, marked)) {
            normal.push_back(idx);
        }
    }
    if (!normal.empty()) {
        return normal;
    }
    return candidates;
}

std::vector<int> forced_removal_targets(const int8_t* board, int size, int forced_done) {
    if (forced_done >= 2) {
        return {};
    }
    int value = forced_done == 0 ? kPlayerBlack : kPlayerWhite;
    std::vector<int> candidates;
    candidates.reserve(size * size);
    for (int r = 0; r < size; ++r) {
        for (int c = 0; c < size; ++c) {
            int idx = flat_index(r, c, size);
            if (board[idx] == value) {
                candidates.push_back(idx);
            }
        }
    }
    return prefer_normal_pieces(board, nullptr, size, candidates, value);
}

std::vector<int> counter_removal_targets(const int8_t* board, int size, int current_value) {
    int player_value = -current_value;
    std::vector<int> candidates;
    candidates.reserve(size * size);
    for (int r = 0; r < size; ++r) {
        for (int c = 0; c < size; ++c) {
            int idx = flat_index(r, c, size);
            if (board[idx] == player_value) {
                candidates.push_back(idx);
            }
        }
    }
    return prefer_normal_pieces(board, nullptr, size, candidates, player_value);
}

std::vector<int> no_moves_removal_targets(const int8_t* board, int size, int current_value) {
    int player_value = -current_value;
    std::vector<int> candidates;
    candidates.reserve(size * size);
    for (int r = 0; r < size; ++r) {
        for (int c = 0; c < size; ++c) {
            int idx = flat_index(r, c, size);
            if (board[idx] == player_value) {
                candidates.push_back(idx);
            }
        }
    }
    return prefer_normal_pieces(board, nullptr, size, candidates, player_value);
}

bool check_no_moves_phase3(
    const int8_t* board,
    int size,
    int current_value) {
    for (int r = 0; r < size; ++r) {
        for (int c = 0; c < size; ++c) {
            int idx = flat_index(r, c, size);
            if (board[idx] != current_value) {
                continue;
            }
            for (int dir = 0; dir < 4; ++dir) {
                int nr = r + kDirections[dir].dr;
                int nc = c + kDirections[dir].dc;
                if (nr >= 0 && nr < size && nc >= 0 && nc < size) {
                    int dest_idx = flat_index(nr, nc, size);
                    if (board[dest_idx] == 0) {
                        return false;
                    }
                }
            }
        }
    }
    return true;
}

std::vector<int> generate_mark_targets(
    const int8_t* board,
    const bool* opponent_marked,
    int size,
    int pending_mark_rem,
    int current_value) {
    int opponent_value = -current_value;
    std::vector<int> candidates;
    candidates.reserve(size * size);
    for (int r = 0; r < size; ++r) {
        for (int c = 0; c < size; ++c) {
            int idx = flat_index(r, c, size);
            if (board[idx] == opponent_value && !is_marked(opponent_marked, idx)) {
                candidates.push_back(idx);
            }
        }
    }
    auto preferred = prefer_normal_pieces(board, opponent_marked, size, candidates, opponent_value);
    if (pending_mark_rem > 0) {
        return preferred;
    }
    return {};
}

std::vector<int> generate_capture_targets(
    const int8_t* board,
    const bool* opponent_marked,
    int size,
    int pending_capture_rem,
    int current_value) {
    int opponent_value = -current_value;
    std::vector<int> candidates;
    candidates.reserve(size * size);
    for (int r = 0; r < size; ++r) {
        for (int c = 0; c < size; ++c) {
            int idx = flat_index(r, c, size);
            if (board[idx] == opponent_value) {
                candidates.push_back(idx);
            }
        }
    }
    if (pending_capture_rem > 0) {
        return prefer_normal_pieces(board, opponent_marked, size, candidates, opponent_value);
    }
    return {};
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor> encode_actions_fast_cpu(
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
    TORCH_CHECK(board.device().is_cpu(), "fast legal mask currently supports CPU tensors only.");

    board = board.contiguous();
    marks_black = marks_black.contiguous();
    marks_white = marks_white.contiguous();
    phase = phase.contiguous();
    current_player = current_player.contiguous();
    pending_marks_required = pending_marks_required.contiguous();
    pending_marks_remaining = pending_marks_remaining.contiguous();
    pending_captures_required = pending_captures_required.contiguous();
    pending_captures_remaining = pending_captures_remaining.contiguous();
    forced_removals_done = forced_removals_done.contiguous();

    const auto B = board.size(0);
    const auto H = board.size(1);
    const auto W = board.size(2);
    const int size = static_cast<int>(H);
    TORCH_CHECK(H == W, "Board must be square.");

    const int64_t total_dim = placement_dim + movement_dim + selection_dim + auxiliary_dim;

    auto mask_options = torch::TensorOptions().dtype(torch::kBool).device(board.device());
    auto meta_options = torch::TensorOptions().dtype(torch::kInt32).device(board.device());
    torch::Tensor mask = torch::zeros({B, total_dim}, mask_options);
    torch::Tensor metadata = torch::full({B, total_dim, kMetaFields}, -1, meta_options);

    const int8_t* board_ptr = board.data_ptr<int8_t>();
    const bool* marks_black_ptr = marks_black.data_ptr<bool>();
    const bool* marks_white_ptr = marks_white.data_ptr<bool>();
    const int64_t* phase_ptr = phase.data_ptr<int64_t>();
    const int64_t* current_ptr = current_player.data_ptr<int64_t>();
    const int64_t* pending_marks_remaining_ptr = pending_marks_remaining.data_ptr<int64_t>();
    const int64_t* pending_captures_remaining_ptr = pending_captures_remaining.data_ptr<int64_t>();
    const int64_t* forced_removals_ptr = forced_removals_done.data_ptr<int64_t>();

    bool* mask_ptr = mask.data_ptr<bool>();
    int32_t* meta_ptr = metadata.data_ptr<int32_t>();

    const int cell_count = size * size;

    auto set_metadata = [&](int64_t global_index, ActionKind kind, int32_t primary, int32_t secondary, int32_t extra) {
        int64_t offset = global_index * kMetaFields;
        meta_ptr[offset + 0] = static_cast<int32_t>(kind);
        meta_ptr[offset + 1] = primary;
        meta_ptr[offset + 2] = secondary;
        meta_ptr[offset + 3] = extra;
    };

    for (int64_t b = 0; b < B; ++b) {
        const int8_t* board_state = board_ptr + b * cell_count;
        const bool* marks_black_state = marks_black_ptr + b * cell_count;
        const bool* marks_white_state = marks_white_ptr + b * cell_count;
        int phase_val = static_cast<int>(phase_ptr[b]);
        int current_val = static_cast<int>(current_ptr[b]);
        int pending_mark_rem = static_cast<int>(pending_marks_remaining_ptr[b]);
        int pending_capture_rem = static_cast<int>(pending_captures_remaining_ptr[b]);
        int forced_done = static_cast<int>(forced_removals_ptr[b]);

        if (phase_val == kPhasePlacement) {
            for (int r = 0; r < size; ++r) {
                for (int c = 0; c < size; ++c) {
                    int idx = flat_index(r, c, size);
                    if (board_state[idx] == 0) {
                        int64_t global_index = b * total_dim + idx;
                        mask_ptr[global_index] = true;
                        set_metadata(global_index, kActionPlacement, idx, -1, -1);
                    }
                }
            }
        }

        bool has_movement = false;
        if (phase_val == kPhaseMovement) {
            for (int r = 0; r < size; ++r) {
                for (int c = 0; c < size; ++c) {
                    int base_idx = flat_index(r, c, size);
                    if (board_state[base_idx] != current_val) {
                        continue;
                    }
                    for (int dir = 0; dir < 4; ++dir) {
                        int nr = r + kDirections[dir].dr;
                        int nc = c + kDirections[dir].dc;
                        if (nr >= 0 && nr < size && nc >= 0 && nc < size) {
                            int dest_idx = flat_index(nr, nc, size);
                            if (board_state[dest_idx] == 0) {
                                int move_offset = base_idx * 4 + dir;
                                int64_t global_index = b * total_dim + placement_dim + move_offset;
                                mask_ptr[global_index] = true;
                                set_metadata(
                                    global_index,
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

        auto emit_selection = [&](const std::vector<int>& indices, ActionKind kind) {
            int64_t base = b * total_dim + placement_dim + movement_dim;
            for (int idx : indices) {
                if (idx >= 0 && idx < selection_dim) {
                    int64_t global_index = base + idx;
                    mask_ptr[global_index] = true;
                    set_metadata(global_index, kind, idx, -1, -1);
                }
            }
        };

        if (phase_val == kPhaseMarkSelection) {
            const bool* opponent_marked = (current_val == kPlayerBlack) ? marks_white_state : marks_black_state;
            auto targets = generate_mark_targets(
                board_state,
                opponent_marked,
                size,
                pending_mark_rem,
                current_val);
            emit_selection(targets, kActionMarkSelection);
        } else if (phase_val == kPhaseCaptureSelection) {
            const bool* opponent_marked = (current_val == kPlayerBlack) ? marks_white_state : marks_black_state;
            auto targets = generate_capture_targets(
                board_state,
                opponent_marked,
                size,
                pending_capture_rem,
                current_val);
            emit_selection(targets, kActionCaptureSelection);
        } else if (phase_val == kPhaseForcedRemoval) {
            auto targets = forced_removal_targets(board_state, size, forced_done);
            emit_selection(targets, kActionForcedRemovalSelection);
        } else if (phase_val == kPhaseCounterRemoval) {
            auto targets = counter_removal_targets(board_state, size, current_val);
            emit_selection(targets, kActionCounterRemovalSelection);
        } else if (phase_val == kPhaseMovement && !has_movement) {
            auto targets = no_moves_removal_targets(board_state, size, current_val);
            emit_selection(targets, kActionNoMovesRemovalSelection);
        }

        if (phase_val == kPhaseRemoval && auxiliary_dim > 0) {
            int64_t global_index = b * total_dim + placement_dim + movement_dim + selection_dim;
            mask_ptr[global_index] = true;
            set_metadata(global_index, kActionProcessRemoval, -1, -1, -1);
        }
    }

    return {mask, metadata};
}

#if defined(V0_HAS_CUDA_LEGAL_MASK)
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
    int64_t auxiliary_dim);
#endif

std::tuple<torch::Tensor, torch::Tensor> encode_actions_fast(
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
    if (board.device().is_cuda()) {
#if defined(V0_HAS_CUDA_LEGAL_MASK)
        return encode_actions_fast_cuda(
            board.contiguous(),
            marks_black.contiguous(),
            marks_white.contiguous(),
            phase.contiguous(),
            current_player.contiguous(),
            pending_marks_required.contiguous(),
            pending_marks_remaining.contiguous(),
            pending_captures_required.contiguous(),
            pending_captures_remaining.contiguous(),
            forced_removals_done.contiguous(),
            placement_dim,
            movement_dim,
            selection_dim,
            auxiliary_dim);
#else
        TORCH_CHECK(
            false,
            "encode_actions_fast: requested CUDA tensors but CUDA kernels were not built. "
            "Please rebuild with -DBUILD_CUDA_KERNELS=ON.");
#endif
    }

    return encode_actions_fast_cpu(
        board.contiguous(),
        marks_black.contiguous(),
        marks_white.contiguous(),
        phase.contiguous(),
        current_player.contiguous(),
        pending_marks_required.contiguous(),
        pending_marks_remaining.contiguous(),
        pending_captures_required.contiguous(),
        pending_captures_remaining.contiguous(),
        forced_removals_done.contiguous(),
        placement_dim,
        movement_dim,
        selection_dim,
        auxiliary_dim);
}

}// namespace v0

#ifndef FAST_LEGAL_MASK_NO_MODULE
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "encode_actions_fast",
        &v0::encode_actions_fast,
        "Fast legal action mask encoder");
}
#endif
