#include <torch/extension.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <tuple>
#include <vector>

namespace v0 {

namespace {

enum Phase : int {
    kPhasePlacement = 1,
    kPhaseMarkSelection = 2,
    kPhaseRemoval = 3,
    kPhaseMovement = 4,
    kPhaseCaptureSelection = 5,
    kPhaseForcedRemoval = 6,
    kPhaseCounterRemoval = 7,
};

enum ActionKind : int32_t {
    kActionInvalid = 0,
    kActionPlacement = 1,
    kActionMovement = 2,
    kActionMarkSelection = 3,
    kActionCaptureSelection = 4,
    kActionForcedRemovalSelection = 5,
    kActionCounterRemovalSelection = 6,
    kActionNoMovesRemovalSelection = 7,
    kActionProcessRemoval = 8,
};

struct Directions {
    int dr;
    int dc;
};

constexpr Directions kDirections[4] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};

struct BatchInputs {
    const int8_t* board;
    const bool* marks_black;
    const bool* marks_white;
    const int64_t* phase;
    const int64_t* current_player;
    const int64_t* pending_marks_required;
    const int64_t* pending_marks_remaining;
    const int64_t* pending_captures_required;
    const int64_t* pending_captures_remaining;
    const int64_t* forced_removals_done;
    const int64_t* move_count;
};

struct BatchOutputs {
    int8_t* board;
    bool* marks_black;
    bool* marks_white;
    int64_t* phase;
    int64_t* current_player;
    int64_t* pending_marks_required;
    int64_t* pending_marks_remaining;
    int64_t* pending_captures_required;
    int64_t* pending_captures_remaining;
    int64_t* forced_removals_done;
    int64_t* move_count;
};

struct ActionEntry {
    int32_t kind;
    int32_t primary;
    int32_t secondary;
    int32_t extra;
    int64_t parent;
};

inline int cell_from_rc(int r, int c, int size) {
    return r * size + c;
}

inline void rc_from_cell(int cell, int size, int& r, int& c) {
    r = cell / size;
    c = cell % size;
}

inline bool is_marked(const bool* marked, int idx) {
    return marked && marked[idx];
}

bool check_squares(
    const int8_t* board,
    int size,
    int r,
    int c,
    int player_value,
    const bool* marked) {
    for (int dr : {0, -1}) {
        for (int dc : {0, -1}) {
            int rr = r + dr;
            int cc = c + dc;
            if (rr >= 0 && rr < size - 1 && cc >= 0 && cc < size - 1) {
                bool ok = true;
                const std::array<std::pair<int, int>, 4> cells = {{
                    {rr, cc},
                    {rr, cc + 1},
                    {rr + 1, cc},
                    {rr + 1, cc + 1},
                }};
                for (const auto& cell : cells) {
                    int idx = cell_from_rc(cell.first, cell.second, size);
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

enum class ShapeResult {
    kNone,
    kLine,
    kSquare,
};

ShapeResult detect_shape(
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

bool is_piece_in_shape(
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

int count_player_pieces(const int8_t* board, int size, int player_value) {
    int count = 0;
    const int cell_count = size * size;
    for (int idx = 0; idx < cell_count; ++idx) {
        if (board[idx] == player_value) {
            ++count;
        }
    }
    return count;
}

bool has_any_marked(const bool* marked, int cell_count) {
    for (int idx = 0; idx < cell_count; ++idx) {
        if (marked[idx]) {
            return true;
        }
    }
    return false;
}

void clear_marks(bool* marked, int cell_count) {
    std::fill(marked, marked + cell_count, false);
}

bool is_board_full(const int8_t* board, int cell_count) {
    for (int idx = 0; idx < cell_count; ++idx) {
        if (board[idx] == 0) {
            return false;
        }
    }
    return true;
}

void set_pending_marks(int64_t* required, int64_t* remaining, int value) {
    *required = value;
    *remaining = value;
}

void clear_pending_marks(int64_t* required, int64_t* remaining) {
    *required = 0;
    *remaining = 0;
}

void set_pending_captures(int64_t* required, int64_t* remaining, int value) {
    *required = value;
    *remaining = value;
}

void clear_pending_captures(int64_t* required, int64_t* remaining) {
    *required = 0;
    *remaining = 0;
}

int64_t switch_player(int64_t current_player) {
    return -current_player;
}

void apply_placement(
    int8_t* board,
    bool* marks_black,
    bool* marks_white,
    int64_t& phase,
    int64_t& current_player,
    int64_t& pending_marks_required,
    int64_t& pending_marks_remaining,
    int64_t& pending_captures_required,
    int64_t& pending_captures_remaining,
    int64_t& forced_removals_done,
    int64_t& move_count,
    int size,
    int cell) {
    (void)pending_captures_required;
    (void)pending_captures_remaining;
    (void)forced_removals_done;

    TORCH_CHECK(phase == kPhasePlacement, "apply_placement called outside placement phase.");
    int r, c;
    rc_from_cell(cell, size, r, c);
    int idx = cell_from_rc(r, c, size);
    TORCH_CHECK(board[idx] == 0, "Placement cell is not empty.");

    const bool* opponent_marked = current_player == 1 ? marks_white : marks_black;
    TORCH_CHECK(!is_marked(opponent_marked, idx), "Placement cell is marked by opponent.");

    board[idx] = static_cast<int8_t>(current_player);

    bool* own_marked = current_player == 1 ? marks_black : marks_white;
    bool already_marked = own_marked[idx];

    if (!already_marked) {
        auto shape = detect_shape(
            board,
            size,
            r,
            c,
            static_cast<int>(current_player),
            own_marked);
        if (shape == ShapeResult::kLine) {
            set_pending_marks(&pending_marks_required, &pending_marks_remaining, 2);
            phase = kPhaseMarkSelection;
            move_count += 1;
            return;
        }
        if (shape == ShapeResult::kSquare) {
            set_pending_marks(&pending_marks_required, &pending_marks_remaining, 1);
            phase = kPhaseMarkSelection;
            move_count += 1;
            return;
        }
    }

    clear_pending_marks(&pending_marks_required, &pending_marks_remaining);
    if (is_board_full(board, size * size)) {
        phase = kPhaseRemoval;
    } else {
        current_player = switch_player(current_player);
        phase = kPhasePlacement;
    }
    move_count += 1;
}

void apply_mark_selection(
    int8_t* board,
    bool* marks_black,
    bool* marks_white,
    int64_t& phase,
    int64_t& current_player,
    int64_t& pending_marks_required,
    int64_t& pending_marks_remaining,
    int size,
    int cell) {
    TORCH_CHECK(phase == kPhaseMarkSelection, "apply_mark_selection outside mark phase.");
    TORCH_CHECK(pending_marks_remaining > 0, "No pending marks remaining.");

    int r, c;
    rc_from_cell(cell, size, r, c);
    int idx = cell_from_rc(r, c, size);

    int opponent_value = static_cast<int>(-current_player);
    bool* opponent_marked = opponent_value == -1 ? marks_white : marks_black;

    TORCH_CHECK(board[idx] == opponent_value, "Selected cell is not opponent piece.");
    TORCH_CHECK(!opponent_marked[idx], "Selected opponent piece already marked.");

    std::vector<int> opponent_normal_positions;
    std::vector<int> opponent_normal_unmarked;
    std::vector<int> opponent_pieces;
    const int cell_count = size * size;
    for (int pos = 0; pos < cell_count; ++pos) {
        if (board[pos] == opponent_value) {
            opponent_pieces.push_back(pos);
            if (!is_piece_in_shape(board, size, pos / size, pos % size, opponent_value, opponent_marked)) {
                opponent_normal_positions.push_back(pos);
                if (!opponent_marked[pos]) {
                    opponent_normal_unmarked.push_back(pos);
                }
            }
        }
    }

    if (!opponent_normal_unmarked.empty() &&
        is_piece_in_shape(board, size, r, c, opponent_value, opponent_marked)) {
        TORCH_CHECK(false, "Cannot mark piece in shape while unmarked normal pieces remain.");
    }

    opponent_marked[idx] = true;
    pending_marks_remaining -= 1;
    if (pending_marks_remaining > 0) {
        return;
    }

    clear_pending_marks(&pending_marks_required, &pending_marks_remaining);
    if (is_board_full(board, cell_count)) {
        phase = kPhaseRemoval;
    } else {
        current_player = switch_player(current_player);
        phase = kPhasePlacement;
    }
}

void process_removal_phase(
    int8_t* board,
    bool* marks_black,
    bool* marks_white,
    int64_t& phase,
    int64_t& current_player,
    int64_t& forced_removals_done,
    int size) {
    TORCH_CHECK(phase == kPhaseRemoval, "process_removal called outside removal phase.");
    const int cell_count = size * size;
    bool any_black = has_any_marked(marks_black, cell_count);
    bool any_white = has_any_marked(marks_white, cell_count);

    if (!any_black && !any_white) {
        phase = kPhaseForcedRemoval;
        current_player = -1;  // WHITE
        forced_removals_done = 0;
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
        phase = kPhaseMovement;
        current_player = -1;  // WHITE moves first
    }
}

void apply_forced_removal(
    int8_t* board,
    int64_t& phase,
    int64_t& current_player,
    int64_t& forced_removals_done,
    int size,
    int cell) {
    TORCH_CHECK(phase == kPhaseForcedRemoval, "forced removal outside phase.");
    int r, c;
    rc_from_cell(cell, size, r, c);
    int idx = cell_from_rc(r, c, size);

    if (forced_removals_done == 0) {
        TORCH_CHECK(current_player == -1, "Forced removal order mismatch (expected WHITE).");
        TORCH_CHECK(board[idx] == 1, "Must remove black piece first.");
        TORCH_CHECK(!is_piece_in_shape(board, size, r, c, 1, nullptr), "Cannot remove black piece in shape.");
        board[idx] = 0;
        forced_removals_done = 1;
        current_player = 1;
    } else if (forced_removals_done == 1) {
        TORCH_CHECK(current_player == 1, "Forced removal order mismatch (expected BLACK).");
        TORCH_CHECK(board[idx] == -1, "Must remove white piece second.");
        TORCH_CHECK(!is_piece_in_shape(board, size, r, c, -1, nullptr), "Cannot remove white piece in shape.");
        board[idx] = 0;
        forced_removals_done = 2;
        phase = kPhaseMovement;
        current_player = -1;  // WHITE to move after forced removal
    } else {
        TORCH_CHECK(false, "Invalid forced removal state.");
    }
}

void apply_no_moves_removal(
    int8_t* board,
    int64_t& phase,
    int64_t& current_player,
    int size,
    int cell) {
    TORCH_CHECK(phase == kPhaseMovement, "no-moves removal only valid during movement.");
    int r, c;
    rc_from_cell(cell, size, r, c);
    int idx = cell_from_rc(r, c, size);
    int opponent_value = static_cast<int>(-current_player);
    TORCH_CHECK(board[idx] == opponent_value, "Must remove opponent piece.");

    std::vector<int> opponent_normal;
    for (int pos = 0; pos < size * size; ++pos) {
        if (board[pos] == opponent_value &&
            !is_piece_in_shape(board, size, pos / size, pos % size, opponent_value, nullptr)) {
            opponent_normal.push_back(pos);
        }
    }
    if (!opponent_normal.empty() &&
        is_piece_in_shape(board, size, r, c, opponent_value, nullptr)) {
        TORCH_CHECK(false, "Cannot remove structural piece while normal pieces remain.");
    }

    board[idx] = 0;
    if (count_player_pieces(board, size, opponent_value) == 0) {
        return;
    }

    phase = kPhaseCounterRemoval;
    current_player = switch_player(current_player);
}

void apply_capture_selection(
    int8_t* board,
    bool* marks_black,
    bool* marks_white,
    int64_t& phase,
    int64_t& current_player,
    int64_t& pending_captures_remaining,
    int64_t& pending_captures_required,
    int size,
    int cell) {
    TORCH_CHECK(phase == kPhaseCaptureSelection, "capture selection outside phase.");
    TORCH_CHECK(pending_captures_remaining > 0, "No pending captures.");
    int r, c;
    rc_from_cell(cell, size, r, c);
    int idx = cell_from_rc(r, c, size);

    int opponent_value = static_cast<int>(-current_player);
    bool* opponent_marked = opponent_value == -1 ? marks_white : marks_black;
    TORCH_CHECK(board[idx] == opponent_value, "Capture target must be opponent piece.");

    std::vector<int> opponent_normal;
    for (int pos = 0; pos < size * size; ++pos) {
        if (board[pos] == opponent_value &&
            !is_piece_in_shape(board, size, pos / size, pos % size, opponent_value, opponent_marked)) {
            opponent_normal.push_back(pos);
        }
    }
    if (!opponent_normal.empty() &&
        is_piece_in_shape(board, size, r, c, opponent_value, opponent_marked)) {
        TORCH_CHECK(false, "Cannot capture structural piece while normal pieces remain.");
    }

    board[idx] = 0;
    pending_captures_remaining -= 1;
    if (count_player_pieces(board, size, opponent_value) == 0) {
        return;
    }
    if (pending_captures_remaining > 0) {
        return;
    }

    clear_pending_captures(&pending_captures_required, &pending_captures_remaining);
    current_player = switch_player(current_player);
    phase = kPhaseMovement;
}

void apply_counter_removal(
    int8_t* board,
    int64_t& phase,
    int64_t& current_player,
    int size,
    int cell) {
    TORCH_CHECK(phase == kPhaseCounterRemoval, "Counter removal outside phase.");
    int r, c;
    rc_from_cell(cell, size, r, c);
    int idx = cell_from_rc(r, c, size);
    int stuck_player_value = static_cast<int>(-current_player);
    TORCH_CHECK(board[idx] == stuck_player_value, "Counter removal must target stuck player.");

    std::vector<int> stuck_normal;
    for (int pos = 0; pos < size * size; ++pos) {
        if (board[pos] == stuck_player_value &&
            !is_piece_in_shape(board, size, pos / size, pos % size, stuck_player_value, nullptr)) {
            stuck_normal.push_back(pos);
        }
    }
    if (!stuck_normal.empty() &&
        is_piece_in_shape(board, size, r, c, stuck_player_value, nullptr)) {
        TORCH_CHECK(false, "Cannot remove structural piece while normal pieces remain.");
    }

    board[idx] = 0;
    if (count_player_pieces(board, size, stuck_player_value) == 0) {
        return;
    }

    phase = kPhaseMovement;
    current_player = switch_player(current_player);
}

void apply_movement(
    int8_t* board,
    int64_t& phase,
    int64_t& current_player,
    int64_t& pending_captures_required,
    int64_t& pending_captures_remaining,
    int size,
    int from_cell,
    int dir_idx) {
    TORCH_CHECK(phase == kPhaseMovement, "Movement action outside movement phase.");
    TORCH_CHECK(dir_idx >= 0 && dir_idx < 4, "Invalid movement direction.");
    int r_from, c_from;
    rc_from_cell(from_cell, size, r_from, c_from);
    int r_to = r_from + kDirections[dir_idx].dr;
    int c_to = c_from + kDirections[dir_idx].dc;
    TORCH_CHECK(r_to >= 0 && r_to < size && c_to >= 0 && c_to < size, "Movement destination out of bounds.");
    int from_idx = cell_from_rc(r_from, c_from, size);
    int to_idx = cell_from_rc(r_to, c_to, size);

    TORCH_CHECK(board[from_idx] == current_player, "Movement source must be current player's piece.");
    TORCH_CHECK(board[to_idx] == 0, "Movement destination must be empty.");

    board[to_idx] = board[from_idx];
    board[from_idx] = 0;

    ShapeResult shape = detect_shape(board, size, r_to, c_to, static_cast<int>(current_player), nullptr);
    if (shape == ShapeResult::kLine) {
        set_pending_captures(&pending_captures_required, &pending_captures_remaining, 2);
        phase = kPhaseCaptureSelection;
        return;
    }
    if (shape == ShapeResult::kSquare) {
        set_pending_captures(&pending_captures_required, &pending_captures_remaining, 1);
        phase = kPhaseCaptureSelection;
        return;
    }

    clear_pending_captures(&pending_captures_required, &pending_captures_remaining);
    current_player = switch_player(current_player);
}

void apply_action(
    const ActionEntry& action,
    const BatchInputs& in,
    BatchOutputs& out,
    int size,
    int cell_count,
    int64_t out_index) {
    int64_t parent = action.parent;

    auto copy_state = [&](auto* dst, const auto* src, size_t count) {
        std::copy(src, src + count, dst);
    };

    int8_t* board_out = out.board + out_index * cell_count;
    copy_state(board_out, in.board + parent * cell_count, cell_count);

    bool* marks_black_out = out.marks_black + out_index * cell_count;
    copy_state(marks_black_out, in.marks_black + parent * cell_count, cell_count);

    bool* marks_white_out = out.marks_white + out_index * cell_count;
    copy_state(marks_white_out, in.marks_white + parent * cell_count, cell_count);

    out.phase[out_index] = in.phase[parent];
    out.current_player[out_index] = in.current_player[parent];
    out.pending_marks_required[out_index] = in.pending_marks_required[parent];
    out.pending_marks_remaining[out_index] = in.pending_marks_remaining[parent];
    out.pending_captures_required[out_index] = in.pending_captures_required[parent];
    out.pending_captures_remaining[out_index] = in.pending_captures_remaining[parent];
    out.forced_removals_done[out_index] = in.forced_removals_done[parent];
    out.move_count[out_index] = in.move_count[parent];

    int64_t& phase = out.phase[out_index];
    int64_t& current_player = out.current_player[out_index];
    int64_t& pending_marks_required = out.pending_marks_required[out_index];
    int64_t& pending_marks_remaining = out.pending_marks_remaining[out_index];
    int64_t& pending_captures_required = out.pending_captures_required[out_index];
    int64_t& pending_captures_remaining = out.pending_captures_remaining[out_index];
    int64_t& forced_removals_done = out.forced_removals_done[out_index];
    int64_t& move_count = out.move_count[out_index];

    switch (action.kind) {
        case kActionPlacement:
            apply_placement(
                board_out,
                marks_black_out,
                marks_white_out,
                phase,
                current_player,
                pending_marks_required,
                pending_marks_remaining,
                pending_captures_required,
                pending_captures_remaining,
                forced_removals_done,
                move_count,
                size,
                action.primary);
            break;
        case kActionMarkSelection:
            apply_mark_selection(
                board_out,
                marks_black_out,
                marks_white_out,
                phase,
                current_player,
                pending_marks_required,
                pending_marks_remaining,
                size,
                action.primary);
            move_count += 1;
            break;
        case kActionProcessRemoval:
            process_removal_phase(
                board_out,
                marks_black_out,
                marks_white_out,
                phase,
                current_player,
                forced_removals_done,
                size);
            move_count += 1;
            break;
        case kActionForcedRemovalSelection:
            apply_forced_removal(
                board_out,
                phase,
                current_player,
                forced_removals_done,
                size,
                action.primary);
            move_count += 1;
            break;
        case kActionMovement:
            apply_movement(
                board_out,
                phase,
                current_player,
                pending_captures_required,
                pending_captures_remaining,
                size,
                action.primary,
                action.secondary);
            move_count += 1;
            break;
        case kActionNoMovesRemovalSelection:
            apply_no_moves_removal(
                board_out,
                phase,
                current_player,
                size,
                action.primary);
            move_count += 1;
            break;
        case kActionCaptureSelection:
            apply_capture_selection(
                board_out,
                marks_black_out,
                marks_white_out,
                phase,
                current_player,
                pending_captures_remaining,
                pending_captures_required,
                size,
                action.primary);
            move_count += 1;
            break;
        case kActionCounterRemovalSelection:
            apply_counter_removal(
                board_out,
                phase,
                current_player,
                size,
                action.primary);
            move_count += 1;
            break;
        default:
            TORCH_CHECK(false, "Unsupported action kind: ", action.kind);
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
    torch::Tensor>
batch_apply_moves(
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
    torch::Tensor action_codes,
    torch::Tensor parent_indices) {
    TORCH_CHECK(board.device().is_cpu(), "batch_apply_moves currently supports CPU tensors only.");
    TORCH_CHECK(action_codes.device().is_cpu(), "action_codes must be on CPU.");
    TORCH_CHECK(parent_indices.device().is_cpu(), "parent_indices must be on CPU.");

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

    BatchInputs in{
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
    };

    BatchOutputs out{
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
    };

    const auto* action_ptr = action_codes.data_ptr<int32_t>();
    const auto* parent_ptr = parent_indices.data_ptr<int64_t>();

    for (int64_t i = 0; i < num_actions; ++i) {
        ActionEntry action{
            action_ptr[i * 4 + 0],
            action_ptr[i * 4 + 1],
            action_ptr[i * 4 + 2],
            action_ptr[i * 4 + 3],
            parent_ptr[i],
        };
        TORCH_CHECK(action.parent >= 0 && action.parent < B, "parent index out of range.");
        apply_action(action, in, out, size, cell_count, i);
    }

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
    };
}

}  // namespace v0

#ifndef FAST_APPLY_MOVES_NO_MODULE
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "batch_apply_moves",
        &v0::batch_apply_moves,
        "Apply encoded actions to tensorized game states (CPU)");
}
#endif
