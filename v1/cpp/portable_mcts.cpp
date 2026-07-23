#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "v0/game_state.hpp"
#include "v0/move_generator.hpp"

namespace py = pybind11;

namespace {

constexpr int kInputChannels = 11;
constexpr int kActionCount = 220;
constexpr int kPlacementCount = 36;
constexpr int kMovementCount = 144;
constexpr int kSelectionOffset = kPlacementCount + kMovementCount;
constexpr int kProcessRemovalAction = kSelectionOffset + 36;
constexpr int kInputSize = kInputChannels * v0::kCellCount;

const std::array<v0::Coord, 4> kDirections = {
    v0::Coord{-1, 0},
    v0::Coord{1, 0},
    v0::Coord{0, -1},
    v0::Coord{0, 1},
};

bool InBounds(const v0::Coord& coord) {
    return coord.first >= 0 && coord.first < v0::kBoardSize &&
        coord.second >= 0 && coord.second < v0::kBoardSize;
}

int ToInt(const py::handle& value, const char* name) {
    py::object object = py::reinterpret_borrow<py::object>(value);
    if (py::hasattr(object, "value")) {
        object = object.attr("value");
    }
    try {
        return object.cast<int>();
    } catch (const py::cast_error&) {
        throw std::runtime_error(std::string("Could not convert ") + name + " to int.");
    }
}

v0::GameState StateFromPython(const py::object& object) {
    if (!object || object.is_none()) {
        throw std::runtime_error("state must not be None");
    }
    auto require = [&](const char* name) -> py::object {
        if (!py::hasattr(object, name)) {
            throw std::runtime_error(std::string("state is missing attribute ") + name);
        }
        return object.attr(name);
    };

    v0::GameState state;
    const auto board = py::list(require("board")).cast<std::vector<std::vector<int>>>();
    if (board.size() != static_cast<std::size_t>(v0::kBoardSize)) {
        throw std::runtime_error("board must be 6x6");
    }
    for (int row = 0; row < v0::kBoardSize; ++row) {
        if (board[static_cast<std::size_t>(row)].size() !=
            static_cast<std::size_t>(v0::kBoardSize)) {
            throw std::runtime_error("board must be 6x6");
        }
        for (int col = 0; col < v0::kBoardSize; ++col) {
            const int value =
                board[static_cast<std::size_t>(row)][static_cast<std::size_t>(col)];
            if (value < -1 || value > 1) {
                throw std::runtime_error("board cells must be in [-1, 1]");
            }
            state.SetBoard(row, col, static_cast<int8_t>(value));
        }
    }

    const int phase = ToInt(require("phase"), "phase");
    if (phase < static_cast<int>(v0::Phase::kPlacement) ||
        phase > static_cast<int>(v0::Phase::kCounterRemoval)) {
        throw std::runtime_error("phase is outside [1, 7]");
    }
    state.phase = static_cast<v0::Phase>(phase);

    const int player = ToInt(require("current_player"), "current_player");
    if (player != 1 && player != -1) {
        throw std::runtime_error("current_player must be 1 or -1");
    }
    state.current_player =
        player == 1 ? v0::Player::kBlack : v0::Player::kWhite;

    auto assign_marks = [&](const char* name, v0::MarkSet& marks) {
        marks.Clear();
        if (!py::hasattr(object, name)) {
            return;
        }
        const py::object values = object.attr(name);
        if (values.is_none()) {
            return;
        }
        for (const py::handle item : py::iterable(values)) {
            const auto coord = py::cast<v0::Coord>(item);
            if (!InBounds(coord)) {
                throw std::runtime_error(std::string(name) + " contains an invalid coordinate");
            }
            marks.Add(coord);
        }
    };
    assign_marks("marked_black", state.marked_black);
    assign_marks("marked_white", state.marked_white);

    auto assign_optional = [&](const char* name, int32_t& destination) {
        if (py::hasattr(object, name)) {
            const py::object value = object.attr(name);
            if (!value.is_none()) {
                destination = value.cast<int32_t>();
            }
        }
    };
    assign_optional("forced_removals_done", state.forced_removals_done);
    assign_optional("move_count", state.move_count);
    assign_optional("pending_marks_required", state.pending_marks_required);
    assign_optional("pending_marks_remaining", state.pending_marks_remaining);
    assign_optional("pending_captures_required", state.pending_captures_required);
    assign_optional("pending_captures_remaining", state.pending_captures_remaining);
    assign_optional("moves_since_capture", state.moves_since_capture);
    return state;
}

py::dict StateToPython(const v0::GameState& state) {
    std::vector<std::vector<int>> board(
        v0::kBoardSize, std::vector<int>(v0::kBoardSize, 0));
    for (int row = 0; row < v0::kBoardSize; ++row) {
        for (int col = 0; col < v0::kBoardSize; ++col) {
            board[static_cast<std::size_t>(row)][static_cast<std::size_t>(col)] =
                state.BoardAt(row, col);
        }
    }
    py::dict result;
    result["board"] = std::move(board);
    result["phase"] = static_cast<int>(state.phase);
    result["current_player"] = v0::PlayerValue(state.current_player);
    result["marked_black"] = state.marked_black.ToVector();
    result["marked_white"] = state.marked_white.ToVector();
    result["forced_removals_done"] = state.forced_removals_done;
    result["move_count"] = state.move_count;
    result["pending_marks_required"] = state.pending_marks_required;
    result["pending_marks_remaining"] = state.pending_marks_remaining;
    result["pending_captures_required"] = state.pending_captures_required;
    result["pending_captures_remaining"] = state.pending_captures_remaining;
    result["moves_since_capture"] = state.moves_since_capture;
    return result;
}

int ActionIndex(const v0::MoveRecord& move) {
    switch (move.action_type) {
        case v0::ActionType::kPlace:
            return v0::CellIndex(move.Position().first, move.Position().second);
        case v0::ActionType::kMove: {
            const auto from = move.From();
            const auto to = move.To();
            const v0::Coord delta{to.first - from.first, to.second - from.second};
            const auto found = std::find(kDirections.begin(), kDirections.end(), delta);
            if (found == kDirections.end()) {
                throw std::runtime_error("movement is not an orthogonal one-cell move");
            }
            const int direction = static_cast<int>(found - kDirections.begin());
            return kPlacementCount + v0::CellIndex(from.first, from.second) * 4 +
                direction;
        }
        case v0::ActionType::kMark:
        case v0::ActionType::kCapture:
        case v0::ActionType::kForcedRemoval:
        case v0::ActionType::kCounterRemoval:
        case v0::ActionType::kNoMovesRemoval:
            return kSelectionOffset +
                v0::CellIndex(move.Position().first, move.Position().second);
        case v0::ActionType::kProcessRemoval:
            return kProcessRemovalAction;
        default:
            throw std::runtime_error("unsupported action type");
    }
}

struct LegalMoves {
    std::vector<v0::MoveRecord> moves;
    std::vector<int> indices;
    std::array<uint8_t, kActionCount> mask{};
};

LegalMoves BuildLegalMoves(const v0::GameState& state) {
    LegalMoves result;
    result.moves = v0::GenerateAllLegalMoves(state);
    result.indices.reserve(result.moves.size());
    for (const auto& move : result.moves) {
        const int index = ActionIndex(move);
        if (index < 0 || index >= kActionCount) {
            throw std::runtime_error("legal move encoded outside the 220-d action space");
        }
        if (result.mask[static_cast<std::size_t>(index)] != 0) {
            throw std::runtime_error("two legal moves share one 220-d action index");
        }
        result.mask[static_cast<std::size_t>(index)] = 1;
        result.indices.push_back(index);
    }
    std::vector<std::size_t> order(result.indices.size());
    for (std::size_t offset = 0; offset < order.size(); ++offset) {
        order[offset] = offset;
    }
    std::sort(order.begin(), order.end(), [&](std::size_t lhs, std::size_t rhs) {
        return result.indices[lhs] < result.indices[rhs];
    });
    std::vector<v0::MoveRecord> sorted_moves;
    std::vector<int> sorted_indices;
    sorted_moves.reserve(order.size());
    sorted_indices.reserve(order.size());
    for (const std::size_t offset : order) {
        sorted_moves.push_back(result.moves[offset]);
        sorted_indices.push_back(result.indices[offset]);
    }
    result.moves = std::move(sorted_moves);
    result.indices = std::move(sorted_indices);
    return result;
}

std::array<float, kInputSize> EncodeModelInput(const v0::GameState& state) {
    std::array<float, kInputSize> result{};
    const int player = v0::PlayerValue(state.current_player);
    const v0::MarkSet& self_marks = state.Marks(state.current_player);
    const v0::MarkSet& opponent_marks = state.Marks(v0::Opponent(state.current_player));
    for (int cell = 0; cell < v0::kCellCount; ++cell) {
        const int value = state.board[static_cast<std::size_t>(cell)];
        result[static_cast<std::size_t>(cell)] = value == player ? 1.0F : 0.0F;
        result[static_cast<std::size_t>(v0::kCellCount + cell)] =
            value == -player ? 1.0F : 0.0F;
        result[static_cast<std::size_t>(2 * v0::kCellCount + cell)] =
            self_marks.ContainsIndex(cell) ? 1.0F : 0.0F;
        result[static_cast<std::size_t>(3 * v0::kCellCount + cell)] =
            opponent_marks.ContainsIndex(cell) ? 1.0F : 0.0F;
    }
    const int phase_channel = 3 + static_cast<int>(state.phase);
    if (phase_channel >= 4 && phase_channel < kInputChannels) {
        const int offset = phase_channel * v0::kCellCount;
        std::fill(
            result.begin() + offset,
            result.begin() + offset + v0::kCellCount,
            1.0F);
    }
    return result;
}

double TerminalValue(const v0::GameState& state) {
    const auto winner = state.GetWinner();
    if (!winner.has_value()) {
        return 0.0;
    }
    return winner.value() == state.current_player ? 1.0 : -1.0;
}

class ThreadPool {
   public:
    explicit ThreadPool(int threads) : thread_count_(std::max(1, threads)) {
        if (thread_count_ <= 1) {
            return;
        }
        workers_.reserve(static_cast<std::size_t>(thread_count_));
        for (int index = 0; index < thread_count_; ++index) {
            workers_.emplace_back([this]() { WorkerLoop(); });
        }
    }

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    ~ThreadPool() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
            ++generation_;
        }
        work_cv_.notify_all();
        for (auto& worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }

    int ThreadCount() const {
        return thread_count_;
    }

    void ParallelFor(
        std::size_t count,
        const std::function<void(std::size_t)>& function) {
        if (count == 0) {
            return;
        }
        if (workers_.empty() || count == 1) {
            for (std::size_t index = 0; index < count; ++index) {
                function(index);
            }
            return;
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            function_ = function;
            item_count_ = count;
            next_.store(0);
            completed_workers_ = 0;
            error_ = nullptr;
            ++generation_;
        }
        work_cv_.notify_all();
        std::unique_lock<std::mutex> lock(mutex_);
        done_cv_.wait(lock, [&]() {
            return completed_workers_ == workers_.size();
        });
        function_ = {};
        if (error_) {
            std::rethrow_exception(error_);
        }
    }

   private:
    void WorkerLoop() {
        std::size_t observed_generation = 0;
        while (true) {
            std::function<void(std::size_t)> function;
            std::size_t count = 0;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                work_cv_.wait(lock, [&]() {
                    return stop_ || generation_ != observed_generation;
                });
                if (stop_) {
                    return;
                }
                observed_generation = generation_;
                function = function_;
                count = item_count_;
            }
            while (true) {
                const std::size_t index = next_.fetch_add(1);
                if (index >= count) {
                    break;
                }
                try {
                    function(index);
                } catch (...) {
                    std::lock_guard<std::mutex> lock(mutex_);
                    if (!error_) {
                        error_ = std::current_exception();
                    }
                    next_.store(count);
                    break;
                }
            }
            {
                std::lock_guard<std::mutex> lock(mutex_);
                ++completed_workers_;
                if (completed_workers_ == workers_.size()) {
                    done_cv_.notify_one();
                }
            }
        }
    }

    int thread_count_{1};
    std::vector<std::thread> workers_;
    std::mutex mutex_;
    std::condition_variable work_cv_;
    std::condition_variable done_cv_;
    bool stop_{false};
    std::size_t generation_{0};
    std::size_t item_count_{0};
    std::size_t completed_workers_{0};
    std::atomic<std::size_t> next_{0};
    std::function<void(std::size_t)> function_;
    std::exception_ptr error_;
};

struct Node {
    explicit Node(
        const v0::GameState& initial_state,
        Node* parent_node = nullptr,
        double initial_prior = 1.0,
        int initial_action = -1)
        : state(initial_state),
          parent(parent_node),
          prior(initial_prior),
          action_index(initial_action),
          terminal(initial_state.IsGameOver()) {}

    v0::GameState state;
    Node* parent{nullptr};
    std::vector<std::unique_ptr<Node>> children;
    double prior{1.0};
    int action_index{-1};
    int visit_count{0};
    double value_sum{0.0};
    bool terminal{false};
    bool expanded{false};
    bool no_legal_terminal{false};
    double initial_value{0.0};

    double MeanValue() const {
        return visit_count > 0 ? value_sum / static_cast<double>(visit_count) : 0.0;
    }
};

struct Tree {
    explicit Tree(const v0::GameState& state)
        : root(std::make_unique<Node>(state)) {}
    std::unique_ptr<Node> root;
    bool active{true};
};

struct PendingEvaluation {
    int tree_index{-1};
    Node* node{nullptr};
    std::vector<Node*> path;
    LegalMoves legal;
    std::array<float, kInputSize> model_input{};
};

enum class PendingKind {
    kNone,
    kRoots,
    kLeaves,
};

class PortableTreeBatch {
   public:
    PortableTreeBatch(
        const std::vector<v0::GameState>& states,
        double exploration_weight,
        int num_threads)
        : exploration_weight_(exploration_weight), pool_(num_threads) {
        if (states.empty()) {
            throw std::runtime_error("PortableTreeBatch requires at least one state");
        }
        if (!std::isfinite(exploration_weight_) || exploration_weight_ < 0.0) {
            throw std::runtime_error("exploration_weight must be finite and non-negative");
        }
        trees_.reserve(states.size());
        for (const auto& state : states) {
            trees_.emplace_back(state);
        }
    }

    int NumTrees() const {
        return static_cast<int>(trees_.size());
    }

    int NumThreads() const {
        return pool_.ThreadCount();
    }

    int IllegalActionCount() const {
        return illegal_action_count_;
    }

    int NonFiniteCount() const {
        return non_finite_count_;
    }

    py::dict PrepareRoots() {
        RequireNoPending();
        std::vector<std::unique_ptr<PendingEvaluation>> rows(trees_.size());
        pool_.ParallelFor(trees_.size(), [&](std::size_t index) {
            Tree& tree = trees_[index];
            Node* root = tree.root.get();
            if (!tree.active || root->terminal) {
                return;
            }
            if (root->state.IsGameOver()) {
                root->terminal = true;
                return;
            }
            if (!root->expanded) {
                auto row = std::make_unique<PendingEvaluation>();
                row->tree_index = static_cast<int>(index);
                row->node = root;
                row->path = {root};
                row->legal = BuildLegalMoves(root->state);
                row->model_input = EncodeModelInput(root->state);
                rows[index] = std::move(row);
            }
        });
        for (auto& row : rows) {
            if (row) {
                pending_.push_back(std::move(*row));
            }
        }
        pending_kind_ = PendingKind::kRoots;
        return PendingBatchToPython();
    }

    py::dict SelectLeaves() {
        RequireNoPending();
        std::vector<std::unique_ptr<PendingEvaluation>> rows(trees_.size());
        pool_.ParallelFor(trees_.size(), [&](std::size_t index) {
            Tree& tree = trees_[index];
            if (!tree.active || tree.root->terminal) {
                return;
            }
            std::vector<Node*> path = SelectPath(tree.root.get());
            Node* leaf = path.back();
            if (leaf->terminal) {
                const double value =
                    leaf->no_legal_terminal ? -1.0 : TerminalValue(leaf->state);
                Backup(path, value);
                return;
            }
            if (leaf->expanded && leaf->children.empty()) {
                leaf->terminal = true;
                leaf->no_legal_terminal = true;
                Backup(path, -1.0);
                return;
            }
            auto row = std::make_unique<PendingEvaluation>();
            row->tree_index = static_cast<int>(index);
            row->node = leaf;
            row->path = std::move(path);
            row->legal = BuildLegalMoves(leaf->state);
            row->model_input = EncodeModelInput(leaf->state);
            rows[index] = std::move(row);
        });
        for (auto& row : rows) {
            if (row) {
                pending_.push_back(std::move(*row));
            }
        }
        pending_kind_ = PendingKind::kLeaves;
        return PendingBatchToPython();
    }

    void CompletePending(
        py::array_t<float, py::array::c_style | py::array::forcecast> priors,
        py::array_t<float, py::array::c_style | py::array::forcecast> values) {
        if (pending_kind_ == PendingKind::kNone) {
            throw std::runtime_error("there is no pending evaluation");
        }
        const auto prior_info = priors.request();
        const auto value_info = values.request();
        if (
            prior_info.ndim != 2 ||
            prior_info.shape[0] != static_cast<py::ssize_t>(pending_.size()) ||
            prior_info.shape[1] != kActionCount) {
            throw std::runtime_error("priors must have shape [pending, 220]");
        }
        if (
            value_info.ndim != 1 ||
            value_info.shape[0] != static_cast<py::ssize_t>(pending_.size())) {
            throw std::runtime_error("values must have shape [pending]");
        }
        const float* prior_data = static_cast<const float*>(prior_info.ptr);
        const float* value_data = static_cast<const float*>(value_info.ptr);
        const bool backup = pending_kind_ == PendingKind::kLeaves;
        pool_.ParallelFor(pending_.size(), [&](std::size_t row) {
            const float* row_priors = prior_data + row * kActionCount;
            const double value = static_cast<double>(value_data[row]);
            if (!std::isfinite(value)) {
                ++non_finite_count_;
                throw std::runtime_error("model value is NaN or Inf");
            }
            const double expanded_value = Expand(pending_[row], row_priors, value);
            if (backup) {
                Backup(pending_[row].path, expanded_value);
            }
        });
        pending_.clear();
        pending_kind_ = PendingKind::kNone;
    }

    py::dict RootPriors() const {
        const py::ssize_t count = static_cast<py::ssize_t>(trees_.size());
        py::array_t<float> priors({count, static_cast<py::ssize_t>(kActionCount)});
        py::array_t<uint8_t> masks({count, static_cast<py::ssize_t>(kActionCount)});
        py::array_t<uint8_t> active(count);
        auto priors_mutable = priors.mutable_unchecked<2>();
        auto masks_mutable = masks.mutable_unchecked<2>();
        auto active_mutable = active.mutable_unchecked<1>();
        for (py::ssize_t row = 0; row < count; ++row) {
            for (int action = 0; action < kActionCount; ++action) {
                priors_mutable(row, action) = 0.0F;
                masks_mutable(row, action) = 0;
            }
            const Tree& tree = trees_[static_cast<std::size_t>(row)];
            const Node& root = *tree.root;
            const bool usable =
                tree.active && root.expanded && !root.terminal && !root.children.empty();
            active_mutable(row) = usable ? 1 : 0;
            if (!usable) {
                continue;
            }
            for (const auto& child : root.children) {
                priors_mutable(row, child->action_index) =
                    static_cast<float>(child->prior);
                masks_mutable(row, child->action_index) = 1;
            }
        }
        py::dict result;
        result["priors"] = std::move(priors);
        result["legal_masks"] = std::move(masks);
        result["active"] = std::move(active);
        return result;
    }

    void SetRootPriors(
        py::array_t<float, py::array::c_style | py::array::forcecast> priors) {
        const auto info = priors.request();
        if (
            info.ndim != 2 ||
            info.shape[0] != static_cast<py::ssize_t>(trees_.size()) ||
            info.shape[1] != kActionCount) {
            throw std::runtime_error("root priors must have shape [trees, 220]");
        }
        const float* data = static_cast<const float*>(info.ptr);
        pool_.ParallelFor(trees_.size(), [&](std::size_t row) {
            Node& root = *trees_[row].root;
            if (!trees_[row].active || root.terminal || root.children.empty()) {
                return;
            }
            double sum = 0.0;
            for (const auto& child : root.children) {
                const double value =
                    static_cast<double>(data[row * kActionCount + child->action_index]);
                if (!std::isfinite(value) || value < 0.0) {
                    ++non_finite_count_;
                    throw std::runtime_error("root prior is negative, NaN, or Inf");
                }
                sum += value;
            }
            if (!std::isfinite(sum) || sum <= 0.0) {
                ++non_finite_count_;
                throw std::runtime_error("root priors have a non-positive sum");
            }
            for (auto& child : root.children) {
                child->prior =
                    static_cast<double>(
                        data[row * kActionCount + child->action_index]) /
                    sum;
            }
        });
    }

    py::dict RootOutputs() const {
        const py::ssize_t count = static_cast<py::ssize_t>(trees_.size());
        py::array_t<float> inputs(
            {count, static_cast<py::ssize_t>(kInputChannels),
             static_cast<py::ssize_t>(v0::kBoardSize),
             static_cast<py::ssize_t>(v0::kBoardSize)});
        py::array_t<uint8_t> masks({count, static_cast<py::ssize_t>(kActionCount)});
        py::array_t<int32_t> visits({count, static_cast<py::ssize_t>(kActionCount)});
        py::array_t<float> root_values(count);
        py::array_t<int32_t> players(count);
        py::array_t<uint8_t> terminals(count);
        py::array_t<uint8_t> active(count);
        auto* input_data = static_cast<float*>(inputs.request().ptr);
        auto* mask_data = static_cast<uint8_t*>(masks.request().ptr);
        auto* visit_data = static_cast<int32_t*>(visits.request().ptr);
        auto* value_data = static_cast<float*>(root_values.request().ptr);
        auto* player_data = static_cast<int32_t*>(players.request().ptr);
        auto* terminal_data = static_cast<uint8_t*>(terminals.request().ptr);
        auto* active_data = static_cast<uint8_t*>(active.request().ptr);
        std::fill(input_data, input_data + count * kInputSize, 0.0F);
        std::fill(mask_data, mask_data + count * kActionCount, 0);
        std::fill(visit_data, visit_data + count * kActionCount, 0);

        for (py::ssize_t row = 0; row < count; ++row) {
            const Tree& tree = trees_[static_cast<std::size_t>(row)];
            const Node& root = *tree.root;
            const auto encoded = EncodeModelInput(root.state);
            std::copy(
                encoded.begin(),
                encoded.end(),
                input_data + row * kInputSize);
            player_data[row] = v0::PlayerValue(root.state.current_player);
            terminal_data[row] =
                (root.terminal || root.children.empty()) ? 1 : 0;
            active_data[row] = tree.active ? 1 : 0;
            value_data[row] = static_cast<float>(
                root.visit_count > 0 ? root.MeanValue() :
                (root.no_legal_terminal ? -1.0 :
                 (root.terminal ? TerminalValue(root.state) : root.initial_value)));
            for (const auto& child : root.children) {
                mask_data[row * kActionCount + child->action_index] = 1;
                visit_data[row * kActionCount + child->action_index] =
                    child->visit_count;
            }
        }
        py::dict result;
        result["model_inputs"] = std::move(inputs);
        result["legal_masks"] = std::move(masks);
        result["visit_counts"] = std::move(visits);
        result["root_values"] = std::move(root_values);
        result["current_players"] = std::move(players);
        result["terminal"] = std::move(terminals);
        result["active"] = std::move(active);
        return result;
    }

    void AdvanceRoots(const std::vector<int>& actions) {
        RequireNoPending();
        if (actions.size() != trees_.size()) {
            throw std::runtime_error("actions must contain one entry per tree");
        }
        pool_.ParallelFor(trees_.size(), [&](std::size_t row) {
            Tree& tree = trees_[row];
            if (!tree.active) {
                return;
            }
            const int action = actions[row];
            if (action < 0) {
                return;
            }
            Node& root = *tree.root;
            auto found = std::find_if(
                root.children.begin(),
                root.children.end(),
                [&](const std::unique_ptr<Node>& child) {
                    return child && child->action_index == action;
                });
            if (found == root.children.end()) {
                ++illegal_action_count_;
                throw std::runtime_error(
                    "selected action is not a child of the current root");
            }
            std::unique_ptr<Node> next = std::move(*found);
            next->parent = nullptr;
            tree.root = std::move(next);
        });
    }

    void Deactivate(const std::vector<int>& tree_indices) {
        RequireNoPending();
        for (const int index : tree_indices) {
            if (index < 0 || index >= static_cast<int>(trees_.size())) {
                throw std::runtime_error("deactivate tree index is out of range");
            }
            trees_[static_cast<std::size_t>(index)].active = false;
        }
    }

    py::dict RootStatus() const {
        const py::ssize_t count = static_cast<py::ssize_t>(trees_.size());
        py::array_t<int32_t> winners(count);
        py::array_t<int32_t> black_pieces(count);
        py::array_t<int32_t> white_pieces(count);
        py::array_t<int32_t> players(count);
        py::array_t<int32_t> phases(count);
        py::array_t<int32_t> move_counts(count);
        py::array_t<int32_t> moves_since_capture(count);
        py::array_t<uint8_t> game_over(count);
        auto winner_mutable = winners.mutable_unchecked<1>();
        auto black_mutable = black_pieces.mutable_unchecked<1>();
        auto white_mutable = white_pieces.mutable_unchecked<1>();
        auto player_mutable = players.mutable_unchecked<1>();
        auto phase_mutable = phases.mutable_unchecked<1>();
        auto move_mutable = move_counts.mutable_unchecked<1>();
        auto capture_mutable = moves_since_capture.mutable_unchecked<1>();
        auto over_mutable = game_over.mutable_unchecked<1>();
        for (py::ssize_t row = 0; row < count; ++row) {
            const auto& state = trees_[static_cast<std::size_t>(row)].root->state;
            const auto winner = state.GetWinner();
            winner_mutable(row) =
                winner.has_value() ? v0::PlayerValue(winner.value()) : 0;
            black_mutable(row) = state.CountPlayerPieces(v0::Player::kBlack);
            white_mutable(row) = state.CountPlayerPieces(v0::Player::kWhite);
            player_mutable(row) = v0::PlayerValue(state.current_player);
            phase_mutable(row) = static_cast<int>(state.phase);
            move_mutable(row) = state.move_count;
            capture_mutable(row) = state.moves_since_capture;
            over_mutable(row) = state.IsGameOver() ? 1 : 0;
        }
        py::dict result;
        result["winner"] = std::move(winners);
        result["black_pieces"] = std::move(black_pieces);
        result["white_pieces"] = std::move(white_pieces);
        result["current_players"] = std::move(players);
        result["phases"] = std::move(phases);
        result["move_counts"] = std::move(move_counts);
        result["moves_since_capture"] = std::move(moves_since_capture);
        result["game_over"] = std::move(game_over);
        return result;
    }

   private:
    void RequireNoPending() const {
        if (pending_kind_ != PendingKind::kNone) {
            throw std::runtime_error(
                "complete the current model evaluation before starting another operation");
        }
    }

    Node* SelectChild(Node* node) const {
        const double sqrt_total =
            std::sqrt(static_cast<double>(std::max(1, node->visit_count)));
        double best_score = -std::numeric_limits<double>::infinity();
        int best_action = std::numeric_limits<int>::max();
        Node* best = nullptr;
        for (const auto& child_ptr : node->children) {
            Node* child = child_ptr.get();
            double q = 0.0;
            if (child->visit_count > 0) {
                q = child->MeanValue();
                if (node->state.current_player != child->state.current_player) {
                    q = -q;
                }
            }
            const double u =
                exploration_weight_ * child->prior * sqrt_total /
                static_cast<double>(1 + child->visit_count);
            const double score = q + u;
            if (
                score > best_score ||
                (score == best_score && child->action_index < best_action)) {
                best_score = score;
                best_action = child->action_index;
                best = child;
            }
        }
        return best;
    }

    std::vector<Node*> SelectPath(Node* root) const {
        std::vector<Node*> path{root};
        Node* node = root;
        while (node->expanded && !node->children.empty() && !node->terminal) {
            Node* child = SelectChild(node);
            if (child == nullptr) {
                break;
            }
            node = child;
            path.push_back(node);
        }
        return path;
    }

    static void Backup(const std::vector<Node*>& path, double leaf_value) {
        if (path.empty() || !std::isfinite(leaf_value)) {
            throw std::runtime_error("backup requires a finite leaf value and non-empty path");
        }
        double value = leaf_value;
        for (std::size_t reverse = path.size(); reverse > 0; --reverse) {
            Node* node = path[reverse - 1];
            ++node->visit_count;
            node->value_sum += value;
            if (reverse > 1) {
                Node* parent = path[reverse - 2];
                if (parent->state.current_player != node->state.current_player) {
                    value = -value;
                }
            }
        }
    }

    double Expand(
        PendingEvaluation& pending,
        const float* dense_priors,
        double value) {
        Node* node = pending.node;
        node->initial_value = value;
        if (pending.legal.moves.empty()) {
            node->expanded = true;
            node->terminal = true;
            node->no_legal_terminal = !node->state.IsGameOver();
            node->initial_value =
                node->no_legal_terminal ? -1.0 : TerminalValue(node->state);
            return node->initial_value;
        }

        double prior_sum = 0.0;
        for (const int action : pending.legal.indices) {
            const double prior = static_cast<double>(dense_priors[action]);
            if (!std::isfinite(prior) || prior < 0.0) {
                ++non_finite_count_;
                throw std::runtime_error("model prior is negative, NaN, or Inf");
            }
            prior_sum += prior;
        }
        const bool uniform = !std::isfinite(prior_sum) || prior_sum <= 0.0;
        if (!std::isfinite(prior_sum)) {
            ++non_finite_count_;
            throw std::runtime_error("model prior sum is NaN or Inf");
        }

        node->children.clear();
        node->children.reserve(pending.legal.moves.size());
        for (std::size_t offset = 0; offset < pending.legal.moves.size(); ++offset) {
            const int action = pending.legal.indices[offset];
            const double prior =
                uniform
                ? 1.0 / static_cast<double>(pending.legal.moves.size())
                : static_cast<double>(dense_priors[action]) / prior_sum;
            const v0::GameState child_state =
                v0::ApplyMove(node->state, pending.legal.moves[offset], true);
            node->children.push_back(std::make_unique<Node>(
                child_state, node, prior, action));
        }
        node->expanded = true;
        return node->initial_value;
    }

    py::dict PendingBatchToPython() const {
        const py::ssize_t count = static_cast<py::ssize_t>(pending_.size());
        py::array_t<float> inputs(
            {count, static_cast<py::ssize_t>(kInputChannels),
             static_cast<py::ssize_t>(v0::kBoardSize),
             static_cast<py::ssize_t>(v0::kBoardSize)});
        py::array_t<uint8_t> masks({count, static_cast<py::ssize_t>(kActionCount)});
        py::array_t<int32_t> tree_indices(count);
        auto* input_data = static_cast<float*>(inputs.request().ptr);
        auto* mask_data = static_cast<uint8_t*>(masks.request().ptr);
        auto* tree_data = static_cast<int32_t*>(tree_indices.request().ptr);
        for (py::ssize_t row = 0; row < count; ++row) {
            const PendingEvaluation& pending =
                pending_[static_cast<std::size_t>(row)];
            std::copy(
                pending.model_input.begin(),
                pending.model_input.end(),
                input_data + row * kInputSize);
            std::copy(
                pending.legal.mask.begin(),
                pending.legal.mask.end(),
                mask_data + row * kActionCount);
            tree_data[row] = pending.tree_index;
        }
        py::dict result;
        result["model_inputs"] = std::move(inputs);
        result["legal_masks"] = std::move(masks);
        result["tree_indices"] = std::move(tree_indices);
        return result;
    }

    std::vector<Tree> trees_;
    double exploration_weight_{1.0};
    ThreadPool pool_;
    std::vector<PendingEvaluation> pending_;
    PendingKind pending_kind_{PendingKind::kNone};
    std::atomic<int> illegal_action_count_{0};
    std::atomic<int> non_finite_count_{0};
};

py::dict InspectState(const py::object& object) {
    const v0::GameState state = StateFromPython(object);
    const LegalMoves legal = BuildLegalMoves(state);
    const auto encoded = EncodeModelInput(state);
    py::array_t<float> model_input(
        {static_cast<py::ssize_t>(kInputChannels),
         static_cast<py::ssize_t>(v0::kBoardSize),
         static_cast<py::ssize_t>(v0::kBoardSize)});
    std::copy(
        encoded.begin(),
        encoded.end(),
        static_cast<float*>(model_input.request().ptr));
    py::dict result;
    result["state"] = StateToPython(state);
    result["legal_action_indices"] = legal.indices;
    result["model_input"] = std::move(model_input);
    result["game_over"] = state.IsGameOver();
    const auto winner = state.GetWinner();
    result["winner"] =
        winner.has_value() ? v0::PlayerValue(winner.value()) : 0;
    return result;
}

py::dict ApplyAction(const py::object& object, int action_index) {
    const v0::GameState state = StateFromPython(object);
    const LegalMoves legal = BuildLegalMoves(state);
    const auto found =
        std::find(legal.indices.begin(), legal.indices.end(), action_index);
    if (found == legal.indices.end()) {
        throw std::runtime_error("action is not legal in the supplied state");
    }
    const std::size_t offset =
        static_cast<std::size_t>(found - legal.indices.begin());
    return StateToPython(v0::ApplyMove(state, legal.moves[offset], true));
}

std::vector<double> DebugBackupValues(
    const std::vector<int>& players,
    double leaf_value) {
    if (players.empty() || !std::isfinite(leaf_value)) {
        throw std::runtime_error("players must be non-empty and leaf_value finite");
    }
    for (const int player : players) {
        if (player != 1 && player != -1) {
            throw std::runtime_error("players must contain only 1 or -1");
        }
    }
    std::vector<double> values(players.size(), 0.0);
    double value = leaf_value;
    for (std::size_t reverse = players.size(); reverse > 0; --reverse) {
        values[reverse - 1] = value;
        if (
            reverse > 1 &&
            players[reverse - 2] != players[reverse - 1]) {
            value = -value;
        }
    }
    return values;
}

std::vector<v0::GameState> StatesFromPython(const py::iterable& objects) {
    std::vector<v0::GameState> states;
    for (const py::handle object : objects) {
        states.push_back(
            StateFromPython(py::reinterpret_borrow<py::object>(object)));
    }
    return states;
}

}  // namespace

PYBIND11_MODULE(_liuzhou_portable_cpp, module) {
    module.doc() =
        "CPU-only threaded portable MCTS core for V1 PyTorch/MPS self-play";
    module.attr("ACTION_COUNT") = kActionCount;
    module.attr("INPUT_CHANNELS") = kInputChannels;
    module.def("inspect_state", &InspectState, py::arg("state"));
    module.def(
        "apply_action",
        &ApplyAction,
        py::arg("state"),
        py::arg("action_index"));
    module.def(
        "debug_backup_values",
        &DebugBackupValues,
        py::arg("players"),
        py::arg("leaf_value"));

    py::class_<PortableTreeBatch>(module, "PortableTreeBatch")
        .def(
            py::init([](
                const py::iterable& states,
                double exploration_weight,
                int num_threads) {
                return std::make_unique<PortableTreeBatch>(
                    StatesFromPython(states),
                    exploration_weight,
                    num_threads);
            }),
            py::arg("states"),
            py::arg("exploration_weight") = 1.0,
            py::arg("num_threads") = 1)
        .def_property_readonly("num_trees", &PortableTreeBatch::NumTrees)
        .def_property_readonly("num_threads", &PortableTreeBatch::NumThreads)
        .def_property_readonly(
            "illegal_action_count",
            &PortableTreeBatch::IllegalActionCount)
        .def_property_readonly(
            "non_finite_count",
            &PortableTreeBatch::NonFiniteCount)
        .def("prepare_roots", &PortableTreeBatch::PrepareRoots)
        .def("select_leaves", &PortableTreeBatch::SelectLeaves)
        .def(
            "complete_pending",
            &PortableTreeBatch::CompletePending,
            py::arg("priors"),
            py::arg("values"))
        .def("root_priors", &PortableTreeBatch::RootPriors)
        .def(
            "set_root_priors",
            &PortableTreeBatch::SetRootPriors,
            py::arg("priors"))
        .def("root_outputs", &PortableTreeBatch::RootOutputs)
        .def(
            "advance_roots",
            &PortableTreeBatch::AdvanceRoots,
            py::arg("actions"))
        .def(
            "deactivate",
            &PortableTreeBatch::Deactivate,
            py::arg("tree_indices"))
        .def("root_status", &PortableTreeBatch::RootStatus);
}
