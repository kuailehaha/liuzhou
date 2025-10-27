const apiBasePath = "/api";
const apiBaseLabel = document.getElementById("api-base");
apiBaseLabel.textContent = apiBasePath;

const newGameForm = document.getElementById("new-game-form");
const setupError = document.getElementById("setup-error");
const setupSection = document.getElementById("setup");
const gameSection = document.getElementById("game");
const startGameButton = document.getElementById("start-game");
const boardElement = document.getElementById("board");
const statusElement = document.getElementById("status");
const metaElement = document.getElementById("meta");
const hintElement = document.getElementById("human-hints");
const logElement = document.getElementById("move-log");
const processRemovalButton = document.getElementById("process-removal");

let currentGameId = null;
let currentState = null;
let currentLegalMoves = [];
let humanPlayer = null;
let selectedSource = null;
const moveLog = [];

function arraysEqual(a, b) {
  return Array.isArray(a) && Array.isArray(b) && a.length === b.length && a.every((v, i) => v === b[i]);
}

function fmtPos(position) {
  if (!position) return "-";
  return `(${position[0]}, ${position[1]})`;
}

function describeMove(move) {
  const phase = move.phase || "?";
  const action = move.action_type || "?";
  switch (action) {
    case "place":
      return `[${phase}] Place at ${fmtPos(move.position)}`;
    case "mark":
      return `[${phase}] Mark ${fmtPos(move.position)}`;
    case "capture":
      return `[${phase}] Capture ${fmtPos(move.position)}`;
    case "remove":
      return `[${phase}] Forced remove ${fmtPos(move.position)}`;
    case "counter_remove":
      return `[${phase}] Counter remove ${fmtPos(move.position)}`;
    case "no_moves_remove":
      return `[${phase}] Remove due to no moves ${fmtPos(move.position)}`;
    case "move":
      return `[${phase}] Move ${fmtPos(move.from_position)} -> ${fmtPos(move.to_position)}`;
    case "process_removal":
      return `[${phase}] Resolve removal`;
    default:
      return `[${phase}] ${action}`;
  }
}

function addLogEntry(by, move) {
  moveLog.push({ by, move });
}

function renderLog() {
  logElement.innerHTML = "";
  moveLog.slice().reverse().forEach((entry) => {
    const div = document.createElement("div");
    div.className = `log-entry ${entry.by}`;
    div.textContent = `${entry.by === "human" ? "Human" : "AI"}: ${describeMove(entry.move)}`;
    logElement.appendChild(div);
  });
}

function buildRequestBody(form) {
  const data = new FormData(form);
  const payload = {
    human_player: data.get("human_player") || "BLACK",
  };
  const modelPath = (data.get("model_path") || "").trim();
  if (modelPath) {
    payload.model_path = modelPath;
  }
  const sims = (data.get("mcts_simulations") || "").trim();
  if (sims) {
    payload.mcts_simulations = Number(sims);
  }
  const temp = (data.get("temperature") || "").trim();
  if (temp) {
    payload.temperature = Number(temp);
  }
  return payload;
}

async function postJSON(url, body) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    const detail = await response.json().catch(() => ({}));
    const message = detail?.detail || response.statusText || "Request failed";
    throw new Error(message);
  }
  return response.json();
}

function isHumanTurn(state) {
  return state.currentPlayer === humanPlayer && !state.isGameOver;
}

function updateStatus(state) {
  if (state.isGameOver) {
    const winner = state.winner ? `${state.winner} wins` : "Game drawn";
    statusElement.textContent = `Game over. ${winner}.`;
    return;
  }
  statusElement.textContent = `Phase: ${state.phase} · Current player: ${state.currentPlayer} · Move count: ${state.moveCount}`;
}

function updateMeta(meta) {
  const parts = [
    `Human: ${meta.humanPlayer}`,
    `AI: ${meta.aiPlayer}`,
    `Agent: ${meta.usingRandomAgent ? "Random" : "MCTS"}`,
  ];
  if (meta.note) {
    parts.push(`Note: ${meta.note}`);
  }
  metaElement.textContent = parts.join(" | ");
}

function updateHint(state) {
  if (!isHumanTurn(state)) {
    hintElement.textContent = state.isGameOver ? "Replay the game or start a new match." : "Waiting for AI…";
    return;
  }

  const phase = state.phase;
  const movementMoves = currentLegalMoves.filter((m) => m.action_type === "move");
  const noMoveRemovals = currentLegalMoves.filter((m) => m.action_type === "no_moves_remove");

  switch (phase) {
    case "PLACEMENT":
      hintElement.textContent = "Click an empty intersection to place a stone.";
      break;
    case "MARK_SELECTION":
      hintElement.textContent = "Select an opponent stone to mark.";
      break;
    case "CAPTURE_SELECTION":
      hintElement.textContent = "Select an opponent stone to capture.";
      break;
    case "REMOVAL":
      hintElement.textContent = "Click the button to resolve the removal phase.";
      break;
    case "FORCED_REMOVAL":
      hintElement.textContent = "Forced removal: select a stone to remove.";
      break;
    case "COUNTER_REMOVAL":
      hintElement.textContent = "Counter removal: select a stone to remove.";
      break;
    case "MOVEMENT":
      if (movementMoves.length > 0) {
        hintElement.textContent = selectedSource
          ? "Choose a destination for the selected stone."
          : "Select one of your stones to move.";
      } else if (noMoveRemovals.length > 0) {
        hintElement.textContent = "You have no legal moves, select an opponent stone to remove.";
      } else {
        hintElement.textContent = "Movement phase.";
      }
      break;
    default:
      hintElement.textContent = `Phase: ${phase}`;
  }
}

function markCellState(cell, state, row, col) {
  const stone = state.board[row][col];
  if (stone === 1) {
    const disc = document.createElement("div");
    disc.className = "stone black";
    cell.appendChild(disc);
  } else if (stone === -1) {
    const disc = document.createElement("div");
    disc.className = "stone white";
    cell.appendChild(disc);
  }

  const markedBlack = state.marked.BLACK.some((pos) => arraysEqual(pos, [row, col]));
  const markedWhite = state.marked.WHITE.some((pos) => arraysEqual(pos, [row, col]));
  if (markedBlack) {
    cell.classList.add("marked-black");
  }
  if (markedWhite) {
    cell.classList.add("marked-white");
  }

  if (selectedSource && arraysEqual(selectedSource, [row, col])) {
    cell.classList.add("selected");
  }
}

function renderBoard(state) {
  boardElement.innerHTML = "";
  const size = state.board.length;
  document.documentElement.style.setProperty("--board-size", size);
  const humanTurn = isHumanTurn(state);
  for (let r = 0; r < size; r += 1) {
    for (let c = 0; c < size; c += 1) {
      const cell = document.createElement("div");
      cell.className = "cell";
      cell.dataset.row = r;
      cell.dataset.col = c;
      markCellState(cell, state, r, c);

      if (!humanTurn || state.phase === "REMOVAL") {
        cell.classList.add("disabled");
      }

      cell.addEventListener("click", () => handleCellClick(r, c));
      boardElement.appendChild(cell);
    }
  }
}

function findSimpleMove(position) {
  return currentLegalMoves.find(
    (move) => move.position && arraysEqual(move.position, position),
  );
}

function findMovementFrom(position) {
  return currentLegalMoves.filter(
    (move) => move.action_type === "move" && arraysEqual(move.from_position, position),
  );
}

function findMovementTo(source, destination) {
  return currentLegalMoves.find(
    (move) =>
      move.action_type === "move" &&
      arraysEqual(move.from_position, source) &&
      arraysEqual(move.to_position, destination),
  );
}

function maybeSendProcessRemoval() {
  const removalMove = currentLegalMoves.find((move) => move.action_type === "process_removal");
  if (removalMove) {
    sendMove(removalMove);
  }
}

function handleCellClick(row, col) {
  if (!currentState || !isHumanTurn(currentState)) {
    return;
  }

  const clicked = [row, col];
  const movementMoves = currentLegalMoves.filter((move) => move.action_type === "move");
  if (movementMoves.length > 0) {
    if (!selectedSource) {
      const candidates = findMovementFrom(clicked);
      if (candidates.length > 0) {
        selectedSource = clicked;
        renderBoard(currentState);
        updateHint(currentState);
        return;
      }
    } else {
      const move = findMovementTo(selectedSource, clicked);
      if (move) {
        selectedSource = null;
        sendMove(move);
        return;
      }
      const newSourceCandidates = findMovementFrom(clicked);
      if (newSourceCandidates.length > 0) {
        selectedSource = clicked;
        renderBoard(currentState);
        updateHint(currentState);
        return;
      }
      selectedSource = null;
      renderBoard(currentState);
      updateHint(currentState);
      return;
    }
  }

  const simpleMove = findSimpleMove(clicked);
  if (simpleMove) {
    sendMove(simpleMove);
  }
}

async function sendMove(move) {
  if (!currentGameId) return;

  try {
    setupError.textContent = "";
    addLogEntry("human", move);
    const body = {
      phase: move.phase,
      actionType: move.action_type,
      position: move.position ?? null,
      fromPosition: move.from_position ?? null,
      toPosition: move.to_position ?? null,
    };
    const data = await postJSON(`${apiBasePath}/game/${currentGameId}/human-move`, body);
    if (!data?.state) {
      throw new Error("Malformed server response.");
    }
    updateGameState(data);
  } catch (error) {
    moveLog.pop(); // remove optimistic human entry
    setupError.textContent = error.message;
  }
}

function updateGameState(data) {
  currentGameId = data.gameId;
  currentState = data.state;
  currentLegalMoves = data.legalMoves || [];
  humanPlayer = data.meta?.humanPlayer || humanPlayer;
  selectedSource = null;

  renderBoard(currentState);
  updateStatus(currentState);
  updateMeta(data.meta || {});
  updateHint(currentState);

  if (data.aiMoves) {
    data.aiMoves.forEach((move) => addLogEntry("ai", move));
  }

  renderLog();
  toggleRemovalButton();
}

function toggleRemovalButton() {
  const removalMove = currentLegalMoves.find((move) => move.action_type === "process_removal");
  if (isHumanTurn(currentState) && removalMove) {
    processRemovalButton.classList.remove("hidden");
  } else {
    processRemovalButton.classList.add("hidden");
  }
}

async function handleNewGame(event) {
  if (event) {
    event.preventDefault();
  }
  try {
    setupError.textContent = "Starting game...";
    const payload = buildRequestBody(newGameForm);
    console.log('Submitting new game', payload);
    const data = await postJSON(`${apiBasePath}/new-game`, payload);
    if (!data?.state) {
      throw new Error("Malformed server response.");
    }
    moveLog.length = 0;
    if (data.aiMoves) {
      data.aiMoves.forEach((move) => addLogEntry("ai", move));
    }
    updateGameState(data);
    setupSection.classList.add("hidden");
    gameSection.classList.remove("hidden");
    setupError.textContent = "";
  } catch (error) {
    setupError.textContent = error.message;
  }
}

if (startGameButton) {
  startGameButton.addEventListener("click", handleNewGame);
}

if (newGameForm) {
  newGameForm.addEventListener("submit", (event) => {
    event.preventDefault();
    handleNewGame(event);
  });
}

if (processRemovalButton) {
  processRemovalButton.addEventListener("click", () => maybeSendProcessRemoval());
}
