#pragma once

#include <cstdint>

namespace v0 {

#ifdef __CUDACC__
#define V0_HOST_DEVICE __host__ __device__
#else
#define V0_HOST_DEVICE
#endif

constexpr int kPhasePlacement = 1;
constexpr int kPhaseMarkSelection = 2;
constexpr int kPhaseRemoval = 3;
constexpr int kPhaseMovement = 4;
constexpr int kPhaseCaptureSelection = 5;
constexpr int kPhaseForcedRemoval = 6;
constexpr int kPhaseCounterRemoval = 7;

constexpr int kPlayerBlack = 1;
constexpr int kPlayerWhite = -1;

struct Directions {
    int dr;
    int dc;
};

constexpr Directions kDirections[4] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};

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

constexpr int kMetaFields = 4;

V0_HOST_DEVICE inline int flat_index(int r, int c, int size) {
    return r * size + c;
}

#undef V0_HOST_DEVICE

}  // namespace v0
