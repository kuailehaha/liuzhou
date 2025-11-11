#include "v0/tensor_state_batch.hpp"

#include <cstring>
#include <stdexcept>
#include <string>

namespace v0 {

namespace {

torch::TensorOptions TensorOptionsForDevice(
    torch::ScalarType dtype,
    const torch::Device& device) {
    return torch::TensorOptions().dtype(dtype).device(device);
}

torch::Device NormalizeDevice(const torch::Device& device) {
    if (!device.has_index() && device.type() == torch::kCUDA) {
        return torch::Device(torch::kCUDA, 0);
    }
    return device;
}

}  // namespace

TensorStateBatch TensorStateBatch::To(const torch::Device& device) const {
    auto norm_device = NormalizeDevice(device);
    TensorStateBatch out;
    out.board = board.to(norm_device);
    out.marks_black = marks_black.to(norm_device);
    out.marks_white = marks_white.to(norm_device);
    out.phase = phase.to(norm_device);
    out.current_player = current_player.to(norm_device);
    out.pending_marks_required = pending_marks_required.to(norm_device);
    out.pending_marks_remaining = pending_marks_remaining.to(norm_device);
    out.pending_captures_required = pending_captures_required.to(norm_device);
    out.pending_captures_remaining = pending_captures_remaining.to(norm_device);
    out.forced_removals_done = forced_removals_done.to(norm_device);
    out.move_count = move_count.to(norm_device);
    out.mask_alive = mask_alive.to(norm_device);
    out.board_size = board_size;
    return out;
}

TensorStateBatch TensorStateBatch::Clone() const {
    TensorStateBatch out;
    out.board = board.clone();
    out.marks_black = marks_black.clone();
    out.marks_white = marks_white.clone();
    out.phase = phase.clone();
    out.current_player = current_player.clone();
    out.pending_marks_required = pending_marks_required.clone();
    out.pending_marks_remaining = pending_marks_remaining.clone();
    out.pending_captures_required = pending_captures_required.clone();
    out.pending_captures_remaining = pending_captures_remaining.clone();
    out.forced_removals_done = forced_removals_done.clone();
    out.move_count = move_count.clone();
    out.mask_alive = mask_alive.clone();
    out.board_size = board_size;
    return out;
}

TensorStateBatch FromGameStates(
    const std::vector<GameState>& states,
    const torch::Device& device) {
    if (states.empty()) {
        throw std::runtime_error("from_game_states requires at least one GameState instance.");
    }

    const auto norm_device = NormalizeDevice(device);
    const int64_t batch_size = static_cast<int64_t>(states.size());
    const int64_t cell_count = kCellCount;

    auto board = torch::zeros(
        {batch_size, kBoardSize, kBoardSize},
        TensorOptionsForDevice(torch::kInt8, norm_device));
    auto marks_black = torch::zeros_like(board, TensorOptionsForDevice(torch::kBool, norm_device));
    auto marks_white = torch::zeros_like(board, TensorOptionsForDevice(torch::kBool, norm_device));

    auto phase = torch::empty(batch_size, TensorOptionsForDevice(torch::kInt64, norm_device));
    auto current_player = torch::empty(batch_size, TensorOptionsForDevice(torch::kInt64, norm_device));
    auto pending_marks_required = torch::empty(batch_size, TensorOptionsForDevice(torch::kInt64, norm_device));
    auto pending_marks_remaining = torch::empty(batch_size, TensorOptionsForDevice(torch::kInt64, norm_device));
    auto pending_captures_required = torch::empty(batch_size, TensorOptionsForDevice(torch::kInt64, norm_device));
    auto pending_captures_remaining = torch::empty(batch_size, TensorOptionsForDevice(torch::kInt64, norm_device));
    auto forced_removals_done = torch::empty(batch_size, TensorOptionsForDevice(torch::kInt64, norm_device));
    auto move_count = torch::empty(batch_size, TensorOptionsForDevice(torch::kInt64, norm_device));
    auto mask_alive = torch::ones(batch_size, TensorOptionsForDevice(torch::kBool, norm_device));

    auto board_ptr = board.data_ptr<int8_t>();
    auto marks_black_ptr = marks_black.data_ptr<uint8_t>();
    auto marks_white_ptr = marks_white.data_ptr<uint8_t>();
    auto phase_ptr = phase.data_ptr<int64_t>();
    auto current_ptr = current_player.data_ptr<int64_t>();
    auto pending_marks_required_ptr = pending_marks_required.data_ptr<int64_t>();
    auto pending_marks_remaining_ptr = pending_marks_remaining.data_ptr<int64_t>();
    auto pending_captures_required_ptr = pending_captures_required.data_ptr<int64_t>();
    auto pending_captures_remaining_ptr = pending_captures_remaining.data_ptr<int64_t>();
    auto forced_removals_ptr = forced_removals_done.data_ptr<int64_t>();
    auto move_count_ptr = move_count.data_ptr<int64_t>();

    for (int64_t b = 0; b < batch_size; ++b) {
        const GameState& state = states[b];
        std::memcpy(board_ptr + b * cell_count, state.board.data(), cell_count);

        for (int idx = 0; idx < cell_count; ++idx) {
            marks_black_ptr[b * cell_count + idx] =
                state.marked_black.ContainsIndex(idx) ? 1 : 0;
            marks_white_ptr[b * cell_count + idx] =
                state.marked_white.ContainsIndex(idx) ? 1 : 0;
        }

        phase_ptr[b] = static_cast<int64_t>(state.phase);
        current_ptr[b] = static_cast<int64_t>(state.current_player == Player::kBlack ? 1 : -1);
        pending_marks_required_ptr[b] = state.pending_marks_required;
        pending_marks_remaining_ptr[b] = state.pending_marks_remaining;
        pending_captures_required_ptr[b] = state.pending_captures_required;
        pending_captures_remaining_ptr[b] = state.pending_captures_remaining;
        forced_removals_ptr[b] = state.forced_removals_done;
        move_count_ptr[b] = state.move_count;
    }

    TensorStateBatch batch{
        board,
        marks_black,
        marks_white,
        phase,
        current_player,
        pending_marks_required,
        pending_marks_remaining,
        pending_captures_required,
        pending_captures_remaining,
        forced_removals_done,
        move_count,
        mask_alive,
        kBoardSize};
    return batch;
}

std::vector<GameState> ToGameStates(const TensorStateBatch& batch) {
    const int64_t batch_size = batch.board.size(0);
    const int64_t cell_count = batch.board.size(1) * batch.board.size(2);

    auto board_cpu = batch.board.to(torch::kCPU, torch::kInt8);
    auto marks_black_cpu = batch.marks_black.to(torch::kCPU, torch::kBool);
    auto marks_white_cpu = batch.marks_white.to(torch::kCPU, torch::kBool);
    auto phase_cpu = batch.phase.to(torch::kCPU, torch::kInt64);
    auto current_cpu = batch.current_player.to(torch::kCPU, torch::kInt64);
    auto pending_marks_required_cpu = batch.pending_marks_required.to(torch::kCPU, torch::kInt64);
    auto pending_marks_remaining_cpu = batch.pending_marks_remaining.to(torch::kCPU, torch::kInt64);
    auto pending_captures_required_cpu = batch.pending_captures_required.to(torch::kCPU, torch::kInt64);
    auto pending_captures_remaining_cpu = batch.pending_captures_remaining.to(torch::kCPU, torch::kInt64);
    auto forced_removals_cpu = batch.forced_removals_done.to(torch::kCPU, torch::kInt64);
    auto move_count_cpu = batch.move_count.to(torch::kCPU, torch::kInt64);

    const int8_t* board_ptr = board_cpu.data_ptr<int8_t>();
    const uint8_t* marks_black_ptr = marks_black_cpu.data_ptr<uint8_t>();
    const uint8_t* marks_white_ptr = marks_white_cpu.data_ptr<uint8_t>();
    const int64_t* phase_ptr = phase_cpu.data_ptr<int64_t>();
    const int64_t* current_ptr = current_cpu.data_ptr<int64_t>();
    const int64_t* pending_marks_required_ptr = pending_marks_required_cpu.data_ptr<int64_t>();
    const int64_t* pending_marks_remaining_ptr = pending_marks_remaining_cpu.data_ptr<int64_t>();
    const int64_t* pending_captures_required_ptr = pending_captures_required_cpu.data_ptr<int64_t>();
    const int64_t* pending_captures_remaining_ptr = pending_captures_remaining_cpu.data_ptr<int64_t>();
    const int64_t* forced_removals_ptr = forced_removals_cpu.data_ptr<int64_t>();
    const int64_t* move_count_ptr = move_count_cpu.data_ptr<int64_t>();

    std::vector<GameState> states;
    states.reserve(batch_size);

    for (int64_t b = 0; b < batch_size; ++b) {
        GameState state;
        std::memcpy(state.board.data(), board_ptr + b * cell_count, cell_count);

        for (int idx = 0; idx < cell_count; ++idx) {
            if (marks_black_ptr[b * cell_count + idx]) {
                int r = idx / kBoardSize;
                int c = idx % kBoardSize;
                state.marked_black.Add(r, c);
            }
            if (marks_white_ptr[b * cell_count + idx]) {
                int r = idx / kBoardSize;
                int c = idx % kBoardSize;
                state.marked_white.Add(r, c);
            }
        }

        state.phase = static_cast<Phase>(phase_ptr[b]);
        state.current_player = current_ptr[b] >= 0 ? Player::kBlack : Player::kWhite;
        state.pending_marks_required = static_cast<int>(pending_marks_required_ptr[b]);
        state.pending_marks_remaining = static_cast<int>(pending_marks_remaining_ptr[b]);
        state.pending_captures_required = static_cast<int>(pending_captures_required_ptr[b]);
        state.pending_captures_remaining = static_cast<int>(pending_captures_remaining_ptr[b]);
        state.forced_removals_done = static_cast<int>(forced_removals_ptr[b]);
        state.move_count = static_cast<int>(move_count_ptr[b]);

        states.push_back(state);
    }

    return states;
}

}  // namespace v0
