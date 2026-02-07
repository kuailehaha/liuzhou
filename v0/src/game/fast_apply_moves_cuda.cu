#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include "fast_legal_mask_common.hpp"
#include "v0/game_state.hpp"

namespace v0 {
namespace {

enum class ShapeResult : int {
    kNone,
    kLine,
    kSquare,
};

__device__ inline int cell_from_rc(int r, int c, int size) {
    return r * size + c;
}

__device__ inline void rc_from_cell(int cell, int size, int& r, int& c) {
    r = cell / size;
    c = cell % size;
}

__device__ inline bool is_marked(const bool* marked, int idx) {
    return marked && marked[idx];
}

__device__ bool check_squares(
    const int8_t* board,
    int size,
    int r,
    int c,
    int player_value,
    const bool* marked) {
    const int offsets[2] = {0, -1};
    for (int dr : offsets) {
        for (int dc : offsets) {
            int rr = r + dr;
            int cc = c + dc;
            if (rr >= 0 && rr < size - 1 && cc >= 0 && cc < size - 1) {
                bool ok = true;
                const int cells[4][2] = {{rr, cc}, {rr, cc + 1}, {rr + 1, cc}, {rr + 1, cc + 1}};
                for (int k = 0; k < 4; ++k) {
                    int idx = cell_from_rc(cells[k][0], cells[k][1], size);
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

__device__ bool check_lines(
    const int8_t* board,
    int size,
    int r,
    int c,
    int player_value,
    const bool* marked) {
    int count = 1;
    for (int dc = c - 1; dc >= 0; --dc) {
        int idx = cell_from_rc(r, dc, size);
        if (board[idx] == player_value && !is_marked(marked, idx)) {
            ++count;
        } else {
            break;
        }
    }
    for (int dc = c + 1; dc < size; ++dc) {
        int idx = cell_from_rc(r, dc, size);
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
        int idx = cell_from_rc(dr, c, size);
        if (board[idx] == player_value && !is_marked(marked, idx)) {
            ++count;
        } else {
            break;
        }
    }
    for (int dr = r + 1; dr < size; ++dr) {
        int idx = cell_from_rc(dr, c, size);
        if (board[idx] == player_value && !is_marked(marked, idx)) {
            ++count;
        } else {
            break;
        }
    }
    return count >= 6;
}

__device__ bool is_piece_in_shape(
    const int8_t* board,
    int size,
    int r,
    int c,
    int player_value,
    const bool* marked) {
    if (board[cell_from_rc(r, c, size)] != player_value) {
        return false;
    }
    return check_squares(board, size, r, c, player_value, marked) ||
           check_lines(board, size, r, c, player_value, marked);
}

__device__ int count_player_pieces(const int8_t* board, int size, int player_value) {
    const int cell_count = size * size;
    int count = 0;
    for (int idx = 0; idx < cell_count; ++idx) {
        if (board[idx] == player_value) {
            ++count;
        }
    }
    return count;
}

__device__ bool has_any_marked(const bool* marked, int cell_count) {
    for (int idx = 0; idx < cell_count; ++idx) {
        if (marked[idx]) {
            return true;
        }
    }
    return false;
}

__device__ void clear_marks(bool* marked, int cell_count) {
    for (int idx = 0; idx < cell_count; ++idx) {
        marked[idx] = false;
    }
}

__device__ bool is_board_full(const int8_t* board, int cell_count) {
    for (int idx = 0; idx < cell_count; ++idx) {
        if (board[idx] == 0) {
            return false;
        }
    }
    return true;
}

__device__ inline void set_pending_marks(int64_t* required, int64_t* remaining, int value) {
    *required = value;
    *remaining = value;
}

__device__ inline void clear_pending_marks(int64_t* required, int64_t* remaining) {
    *required = 0;
    *remaining = 0;
}

__device__ inline void set_pending_captures(int64_t* required, int64_t* remaining, int value) {
    *required = value;
    *remaining = value;
}

__device__ inline void clear_pending_captures(int64_t* required, int64_t* remaining) {
    *required = 0;
    *remaining = 0;
}

__device__ inline int64_t switch_player(int64_t current_player) {
    return -current_player;
}

__device__ ShapeResult detect_shape(
    const int8_t* board,
    int size,
    int r,
    int c,
    int player_value,
    const bool* marked) {
    bool found_square = check_squares(board, size, r, c, player_value, marked);
    bool found_line = check_lines(board, size, r, c, player_value, marked);
    if (found_line) {
        return ShapeResult::kLine;
    }
    if (found_square) {
        return ShapeResult::kSquare;
    }
    return ShapeResult::kNone;
}

__device__ void process_removal_phase(
    int8_t* board,
    bool* marks_black,
    bool* marks_white,
    int64_t* phase,
    int64_t* current_player,
    int64_t* forced_removals_done,
    int size) {
    const int cell_count = size * size;
    bool any_black = has_any_marked(marks_black, cell_count);
    bool any_white = has_any_marked(marks_white, cell_count);

    if (!any_black && !any_white) {
        *phase = kPhaseForcedRemoval;
        *current_player = -1;
        *forced_removals_done = 0;
        return;
    }

    int removed = 0;
    for (int idx = 0; idx < cell_count; ++idx) {
        if (marks_black[idx]) {
            board[idx] = 0;
            ++removed;
        } else if (marks_white[idx]) {
            board[idx] = 0;
            ++removed;
        }
    }
    clear_marks(marks_black, cell_count);
    clear_marks(marks_white, cell_count);

    if (removed > 0) {
        *phase = kPhaseMovement;
        *current_player = -1;
    }
}

__device__ void apply_placement(
    int8_t* board,
    bool* marks_black,
    bool* marks_white,
    int64_t* phase,
    int64_t* current_player,
    int64_t* pending_marks_required,
    int64_t* pending_marks_remaining,
    int64_t* pending_captures_required,
    int64_t* pending_captures_remaining,
    int64_t* forced_removals_done,
    int64_t* move_count,
    int size,
    int cell) {
    (void)pending_captures_required;
    (void)pending_captures_remaining;
    (void)forced_removals_done;

    if (*phase != kPhasePlacement) {
        return;
    }
    if (cell < 0 || cell >= size * size) {
        return;
    }
    if (board[cell] != 0) {
        return;
    }

    const bool* opponent_marked = *current_player == 1 ? marks_white : marks_black;
    if (is_marked(opponent_marked, cell)) {
        return;
    }

    board[cell] = static_cast<int8_t>(*current_player);

    bool* own_marked = *current_player == 1 ? marks_black : marks_white;
    if (!is_marked(own_marked, cell)) {
        int r, c;
        rc_from_cell(cell, size, r, c);
        ShapeResult shape = detect_shape(board, size, r, c, static_cast<int>(*current_player), own_marked);
        if (shape == ShapeResult::kLine) {
            set_pending_marks(pending_marks_required, pending_marks_remaining, 2);
            *phase = kPhaseMarkSelection;
            *move_count += 1;
            return;
        }
        if (shape == ShapeResult::kSquare) {
            set_pending_marks(pending_marks_required, pending_marks_remaining, 1);
            *phase = kPhaseMarkSelection;
            *move_count += 1;
            return;
        }
    }

    clear_pending_marks(pending_marks_required, pending_marks_remaining);
    if (is_board_full(board, size * size)) {
        *phase = kPhaseRemoval;
    } else {
        *current_player = switch_player(*current_player);
        *phase = kPhasePlacement;
    }
    *move_count += 1;
}

__device__ bool has_unmarked_normal_piece(
    const int8_t* board,
    int size,
    int player_value,
    const bool* marked) {
    const int cell_count = size * size;
    for (int idx = 0; idx < cell_count; ++idx) {
        if (board[idx] == player_value) {
            int r, c;
            rc_from_cell(idx, size, r, c);
            if (!is_piece_in_shape(board, size, r, c, player_value, marked) && !is_marked(marked, idx)) {
                return true;
            }
        }
    }
    return false;
}

__device__ void apply_mark_selection(
    int8_t* board,
    bool* marks_black,
    bool* marks_white,
    int64_t* phase,
    int64_t* current_player,
    int64_t* pending_marks_required,
    int64_t* pending_marks_remaining,
    int size,
    int cell) {
    if (*phase != kPhaseMarkSelection || *pending_marks_remaining <= 0) {
        return;
    }
    if (cell < 0 || cell >= size * size) {
        return;
    }

    int opponent_value = static_cast<int>(-*current_player);
    bool* opponent_marked = opponent_value == -1 ? marks_white : marks_black;
    if (board[cell] != opponent_value || is_marked(opponent_marked, cell)) {
        return;
    }

    int r, c;
    rc_from_cell(cell, size, r, c);
    bool selected_in_shape = is_piece_in_shape(board, size, r, c, opponent_value, opponent_marked);
    if (selected_in_shape && has_unmarked_normal_piece(board, size, opponent_value, opponent_marked)) {
        return;
    }

    opponent_marked[cell] = true;
    *pending_marks_remaining -= 1;
    if (*pending_marks_remaining > 0) {
        return;
    }

    clear_pending_marks(pending_marks_required, pending_marks_remaining);
    if (is_board_full(board, size * size)) {
        *phase = kPhaseRemoval;
    } else {
        *current_player = switch_player(*current_player);
        *phase = kPhasePlacement;
    }
}

__device__ void apply_forced_removal(
    int8_t* board,
    int64_t* phase,
    int64_t* current_player,
    int64_t* forced_removals_done,
    int size,
    int cell) {
    if (*phase != kPhaseForcedRemoval || cell < 0 || cell >= size * size) {
        return;
    }
    int r, c;
    rc_from_cell(cell, size, r, c);
    int idx = cell_from_rc(r, c, size);

    if (*forced_removals_done == 0) {
        if (*current_player != -1 || board[idx] != 1) {
            return;
        }
        if (is_piece_in_shape(board, size, r, c, 1, nullptr)) {
            return;
        }
        board[idx] = 0;
        *forced_removals_done = 1;
        *current_player = 1;
    } else if (*forced_removals_done == 1) {
        if (*current_player != 1 || board[idx] != -1) {
            return;
        }
        if (is_piece_in_shape(board, size, r, c, -1, nullptr)) {
            return;
        }
        board[idx] = 0;
        *forced_removals_done = 2;
        *phase = kPhaseMovement;
        *current_player = -1;
    }
}

__device__ void apply_no_moves_removal(
    int8_t* board,
    int64_t* phase,
    int64_t* current_player,
    int size,
    int cell) {
    if (*phase != kPhaseMovement || cell < 0 || cell >= size * size) {
        return;
    }
    int opponent_value = static_cast<int>(-*current_player);
    if (board[cell] != opponent_value) {
        return;
    }

    int r, c;
    rc_from_cell(cell, size, r, c);
    bool selected_in_shape = is_piece_in_shape(board, size, r, c, opponent_value, nullptr);
    if (selected_in_shape && has_unmarked_normal_piece(board, size, opponent_value, nullptr)) {
        return;
    }

    board[cell] = 0;
    if (count_player_pieces(board, size, opponent_value) == 0) {
        return;
    }

    *phase = kPhaseCounterRemoval;
    *current_player = switch_player(*current_player);
}

__device__ void apply_capture_selection(
    int8_t* board,
    bool* marks_black,
    bool* marks_white,
    int64_t* phase,
    int64_t* current_player,
    int64_t* pending_captures_remaining,
    int64_t* pending_captures_required,
    int size,
    int cell) {
    if (*phase != kPhaseCaptureSelection || *pending_captures_remaining <= 0 || cell < 0 || cell >= size * size) {
        return;
    }

    int opponent_value = static_cast<int>(-*current_player);
    bool* opponent_marked = opponent_value == -1 ? marks_white : marks_black;
    if (board[cell] != opponent_value) {
        return;
    }

    int r, c;
    rc_from_cell(cell, size, r, c);
    bool selected_in_shape = is_piece_in_shape(board, size, r, c, opponent_value, opponent_marked);
    if (selected_in_shape && has_unmarked_normal_piece(board, size, opponent_value, opponent_marked)) {
        return;
    }

    board[cell] = 0;
    *pending_captures_remaining -= 1;
    if (count_player_pieces(board, size, opponent_value) == 0 || *pending_captures_remaining > 0) {
        return;
    }

    clear_pending_captures(pending_captures_required, pending_captures_remaining);
    *current_player = switch_player(*current_player);
    *phase = kPhaseMovement;
}

__device__ void apply_counter_removal(
    int8_t* board,
    int64_t* phase,
    int64_t* current_player,
    int size,
    int cell) {
    if (*phase != kPhaseCounterRemoval || cell < 0 || cell >= size * size) {
        return;
    }
    int stuck_player_value = static_cast<int>(-*current_player);
    if (board[cell] != stuck_player_value) {
        return;
    }

    int r, c;
    rc_from_cell(cell, size, r, c);
    bool selected_in_shape = is_piece_in_shape(board, size, r, c, stuck_player_value, nullptr);
    if (selected_in_shape && has_unmarked_normal_piece(board, size, stuck_player_value, nullptr)) {
        return;
    }

    board[cell] = 0;
    if (count_player_pieces(board, size, stuck_player_value) == 0) {
        return;
    }

    *phase = kPhaseMovement;
    *current_player = switch_player(*current_player);
}

__device__ void apply_movement(
    int8_t* board,
    int64_t* phase,
    int64_t* current_player,
    int64_t* pending_captures_required,
    int64_t* pending_captures_remaining,
    int size,
    int from_cell,
    int dir_idx) {
    if (*phase != kPhaseMovement || dir_idx < 0 || dir_idx >= 4) {
        return;
    }

    int r_from, c_from;
    rc_from_cell(from_cell, size, r_from, c_from);
    const Directions dirs[4] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
    int r_to = r_from + dirs[dir_idx].dr;
    int c_to = c_from + dirs[dir_idx].dc;
    if (r_to < 0 || r_to >= size || c_to < 0 || c_to >= size) {
        return;
    }
    int from_idx = cell_from_rc(r_from, c_from, size);
    int to_idx = cell_from_rc(r_to, c_to, size);
    if (board[from_idx] != *current_player || board[to_idx] != 0) {
        return;
    }

    board[to_idx] = board[from_idx];
    board[from_idx] = 0;

    ShapeResult shape = detect_shape(board, size, r_to, c_to, static_cast<int>(*current_player), nullptr);
    if (shape == ShapeResult::kLine) {
        set_pending_captures(pending_captures_required, pending_captures_remaining, 2);
        *phase = kPhaseCaptureSelection;
        return;
    }
    if (shape == ShapeResult::kSquare) {
        set_pending_captures(pending_captures_required, pending_captures_remaining, 1);
        *phase = kPhaseCaptureSelection;
        return;
    }

    clear_pending_captures(pending_captures_required, pending_captures_remaining);
    *current_player = switch_player(*current_player);
}

__global__ void BatchApplyMovesKernel(
    const int8_t* board,
    const bool* marks_black,
    const bool* marks_white,
    const int64_t* phase,
    const int64_t* current_player,
    const int64_t* pending_marks_required,
    const int64_t* pending_marks_remaining,
    const int64_t* pending_captures_required,
    const int64_t* pending_captures_remaining,
    const int64_t* forced_removals_done,
    const int64_t* move_count,
    const int64_t* moves_since_capture,
    const int32_t* action_codes,
    const int64_t* parent_indices,
    int64_t batch_size,
    int64_t num_actions,
    int size,
    int cell_count,
    int8_t* out_board,
    bool* out_marks_black,
    bool* out_marks_white,
    int64_t* out_phase,
    int64_t* out_current_player,
    int64_t* out_pending_marks_required,
    int64_t* out_pending_marks_remaining,
    int64_t* out_pending_captures_required,
    int64_t* out_pending_captures_remaining,
    int64_t* out_forced_removals_done,
    int64_t* out_move_count,
    int64_t* out_moves_since_capture) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_actions) {
        return;
    }

    int64_t parent = parent_indices[idx];
    if (parent < 0 || parent >= batch_size) {
        return;
    }

    const int8_t* parent_board = board + parent * cell_count;
    int8_t* child_board = out_board + idx * cell_count;
    for (int i = 0; i < cell_count; ++i) {
        child_board[i] = parent_board[i];
    }

    const bool* parent_marks_black = marks_black + parent * cell_count;
    bool* child_marks_black = out_marks_black + idx * cell_count;
    for (int i = 0; i < cell_count; ++i) {
        child_marks_black[i] = parent_marks_black[i];
    }
    const bool* parent_marks_white = marks_white + parent * cell_count;
    bool* child_marks_white = out_marks_white + idx * cell_count;
    for (int i = 0; i < cell_count; ++i) {
        child_marks_white[i] = parent_marks_white[i];
    }

    out_phase[idx] = phase[parent];
    out_current_player[idx] = current_player[parent];
    out_pending_marks_required[idx] = pending_marks_required[parent];
    out_pending_marks_remaining[idx] = pending_marks_remaining[parent];
    out_pending_captures_required[idx] = pending_captures_required[parent];
    out_pending_captures_remaining[idx] = pending_captures_remaining[parent];
    out_forced_removals_done[idx] = forced_removals_done[parent];
    out_move_count[idx] = move_count[parent];
    out_moves_since_capture[idx] = moves_since_capture[parent];

    int64_t* phase_ptr = out_phase + idx;
    int64_t* current_ptr = out_current_player + idx;
    int64_t* marks_req_ptr = out_pending_marks_required + idx;
    int64_t* marks_rem_ptr = out_pending_marks_remaining + idx;
    int64_t* captures_req_ptr = out_pending_captures_required + idx;
    int64_t* captures_rem_ptr = out_pending_captures_remaining + idx;
    int64_t* forced_ptr = out_forced_removals_done + idx;
    int64_t* move_count_ptr = out_move_count + idx;
    int64_t phase_before = *phase_ptr;

    int32_t kind = action_codes[idx * 4 + 0];
    int32_t primary = action_codes[idx * 4 + 1];
    int32_t secondary = action_codes[idx * 4 + 2];

    switch (kind) {
        case kActionPlacement:
            apply_placement(
                child_board,
                child_marks_black,
                child_marks_white,
                phase_ptr,
                current_ptr,
                marks_req_ptr,
                marks_rem_ptr,
                captures_req_ptr,
                captures_rem_ptr,
                forced_ptr,
                move_count_ptr,
                size,
                primary);
            break;
        case kActionMarkSelection:
            apply_mark_selection(
                child_board,
                child_marks_black,
                child_marks_white,
                phase_ptr,
                current_ptr,
                marks_req_ptr,
                marks_rem_ptr,
                size,
                primary);
            *move_count_ptr += 1;
            break;
        case kActionProcessRemoval:
            process_removal_phase(
                child_board,
                child_marks_black,
                child_marks_white,
                phase_ptr,
                current_ptr,
                forced_ptr,
                size);
            *move_count_ptr += 1;
            break;
        case kActionForcedRemovalSelection:
            apply_forced_removal(
                child_board,
                phase_ptr,
                current_ptr,
                forced_ptr,
                size,
                primary);
            *move_count_ptr += 1;
            break;
        case kActionMovement:
            apply_movement(
                child_board,
                phase_ptr,
                current_ptr,
                captures_req_ptr,
                captures_rem_ptr,
                size,
                primary,
                secondary);
            *move_count_ptr += 1;
            break;
        case kActionNoMovesRemovalSelection:
            apply_no_moves_removal(
                child_board,
                phase_ptr,
                current_ptr,
                size,
                primary);
            *move_count_ptr += 1;
            break;
        case kActionCaptureSelection:
            apply_capture_selection(
                child_board,
                child_marks_black,
                child_marks_white,
                phase_ptr,
                current_ptr,
                captures_rem_ptr,
                captures_req_ptr,
                size,
                primary);
            *move_count_ptr += 1;
            break;
        case kActionCounterRemovalSelection:
            apply_counter_removal(
                child_board,
                phase_ptr,
                current_ptr,
                size,
                primary);
            *move_count_ptr += 1;
            break;
        default:
            break;
    }

    // Track moves_since_capture for no-capture draw detection
    if (phase_before == kPhasePlacement || phase_before == kPhaseMarkSelection) {
        out_moves_since_capture[idx] = 0;
    } else {
        const int8_t* parent_board_ptr = board + parent * cell_count;
        int old_total = 0, new_total = 0;
        for (int i = 0; i < cell_count; ++i) {
            if (parent_board_ptr[i] != 0) ++old_total;
            if (child_board[i] != 0) ++new_total;
        }
        if (new_total < old_total) {
            out_moves_since_capture[idx] = 0;
        } else {
            out_moves_since_capture[idx] = moves_since_capture[parent] + 1;
        }
    }
}

}  // namespace

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
batch_apply_moves_cuda(
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
    torch::Tensor move_count,
    torch::Tensor moves_since_capture,
    torch::Tensor action_codes,
    torch::Tensor parent_indices) {
    TORCH_CHECK(board.device().is_cuda(), "batch_apply_moves_cuda requires CUDA tensors.");

    at::cuda::OptionalCUDAGuard guard(board.device());

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
    move_count = move_count.contiguous();
    moves_since_capture = moves_since_capture.contiguous();
    action_codes = action_codes.contiguous();
    parent_indices = parent_indices.contiguous();

    const auto B = board.size(0);
    const auto H = board.size(1);
    const auto W = board.size(2);
    TORCH_CHECK(H == W, "Board must be square.");
    const int size = static_cast<int>(H);
    const int cell_count = size * size;

    TORCH_CHECK(action_codes.dim() == 2 && action_codes.size(1) == 4, "action_codes must be (N, 4).");
    const int64_t num_actions = action_codes.size(0);
    TORCH_CHECK(parent_indices.numel() == num_actions, "parent_indices must align with action_codes.");

    auto options_board = board.options();
    auto options_bool = marks_black.options();
    auto options_long = phase.options();

    torch::Tensor out_board = torch::empty({num_actions, size, size}, options_board);
    torch::Tensor out_marks_black = torch::empty({num_actions, size, size}, options_bool);
    torch::Tensor out_marks_white = torch::empty({num_actions, size, size}, options_bool);
    torch::Tensor out_phase = torch::empty({num_actions}, options_long);
    torch::Tensor out_current_player = torch::empty({num_actions}, options_long);
    torch::Tensor out_pending_marks_required = torch::empty({num_actions}, options_long);
    torch::Tensor out_pending_marks_remaining = torch::empty({num_actions}, options_long);
    torch::Tensor out_pending_captures_required = torch::empty({num_actions}, options_long);
    torch::Tensor out_pending_captures_remaining = torch::empty({num_actions}, options_long);
    torch::Tensor out_forced_removals_done = torch::empty({num_actions}, options_long);
    torch::Tensor out_move_count = torch::empty({num_actions}, options_long);
    torch::Tensor out_moves_since_capture = torch::empty({num_actions}, options_long);

    if (num_actions == 0) {
        return {
            out_board,
            out_marks_black,
            out_marks_white,
            out_phase,
            out_current_player,
            out_pending_marks_required,
            out_pending_marks_remaining,
            out_pending_captures_required,
            out_pending_captures_remaining,
            out_forced_removals_done,
            out_move_count,
            out_moves_since_capture,
        };
    }

    const int threads = 128;
    const int blocks = static_cast<int>((num_actions + threads - 1) / threads);
    BatchApplyMovesKernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        board.data_ptr<int8_t>(),
        marks_black.data_ptr<bool>(),
        marks_white.data_ptr<bool>(),
        phase.data_ptr<int64_t>(),
        current_player.data_ptr<int64_t>(),
        pending_marks_required.data_ptr<int64_t>(),
        pending_marks_remaining.data_ptr<int64_t>(),
        pending_captures_required.data_ptr<int64_t>(),
        pending_captures_remaining.data_ptr<int64_t>(),
        forced_removals_done.data_ptr<int64_t>(),
        move_count.data_ptr<int64_t>(),
        moves_since_capture.data_ptr<int64_t>(),
        action_codes.data_ptr<int32_t>(),
        parent_indices.data_ptr<int64_t>(),
        B,
        num_actions,
        size,
        cell_count,
        out_board.data_ptr<int8_t>(),
        out_marks_black.data_ptr<bool>(),
        out_marks_white.data_ptr<bool>(),
        out_phase.data_ptr<int64_t>(),
        out_current_player.data_ptr<int64_t>(),
        out_pending_marks_required.data_ptr<int64_t>(),
        out_pending_marks_remaining.data_ptr<int64_t>(),
        out_pending_captures_required.data_ptr<int64_t>(),
        out_pending_captures_remaining.data_ptr<int64_t>(),
        out_forced_removals_done.data_ptr<int64_t>(),
        out_move_count.data_ptr<int64_t>(),
        out_moves_since_capture.data_ptr<int64_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {
        out_board,
        out_marks_black,
        out_marks_white,
        out_phase,
        out_current_player,
        out_pending_marks_required,
        out_pending_marks_remaining,
        out_pending_captures_required,
        out_pending_captures_remaining,
        out_forced_removals_done,
        out_move_count,
        out_moves_since_capture,
    };
}

}  // namespace v0
