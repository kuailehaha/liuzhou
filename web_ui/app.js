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
const evaluationElement = document.getElementById("evaluation");
const metaElement = document.getElementById("meta");
const hintElement = document.getElementById("human-hints");
const logElement = document.getElementById("move-log");
const processRemovalButton = document.getElementById("process-removal");

let currentGameId = null;
let currentState = null;
let currentLegalMoves = [];
let humanPlayer = null;
let selectedSource = null;
let currentEvaluation = null;
let lastOpponentHighlight = null;
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

function updateEvaluation(info) {
  if (!evaluationElement) {
    return;
  }

  currentEvaluation = info ?? null;
  if (!currentEvaluation) {
    evaluationElement.textContent = "Network evaluation: unavailable (no neural estimate).";
    return;
  }

  const { value, winProbability, perspective, range } = currentEvaluation;
  const [rangeMin, rangeMax] = Array.isArray(range) && range.length >= 2 ? range : [-1, 1];
  const sideLabel = perspective ? `${perspective} to move` : "Current player";

  if (!Number.isFinite(value)) {
    evaluationElement.textContent = `Network evaluation (${sideLabel}, range ${rangeMin}..${rangeMax}): unavailable.`;
    return;
  }

  const clampedValue = Math.max(rangeMin ?? -1, Math.min(rangeMax ?? 1, value));
  const rawProb = Number.isFinite(winProbability) ? winProbability : (clampedValue + 1) / 2;
  const clampedProb = Math.max(0, Math.min(1, rawProb));
  const percentText = `${(clampedProb * 100).toFixed(1)}%`;
  evaluationElement.textContent = `Network evaluation (${sideLabel}, range ${rangeMin}..${rangeMax}): value ${clampedValue.toFixed(3)}, win approx ${percentText}.`;
}

function getMoveLandingPosition(move) {
  if (!move) {
    return null;
  }
  const action = move.action_type;
  if (action === "place" && Array.isArray(move.position)) {
    return move.position.slice(0, 2);
  }
  if (action === "move" && Array.isArray(move.to_position)) {
    return move.to_position.slice(0, 2);
  }
  return null;
}

function refreshOpponentHighlight(moves) {
  if (!Array.isArray(moves) || moves.length === 0) {
    return;
  }
  const board = currentState?.board;
  for (let idx = moves.length - 1; idx >= 0; idx -= 1) {
    const landing = getMoveLandingPosition(moves[idx]);
    if (!landing) {
      continue;
    }
    const [row, col] = landing;
    if (!board || !Array.isArray(board[row]) || typeof board[row][col] === "undefined") {
      lastOpponentHighlight = landing.slice(0, 2);
      return;
    }
    const occupant = board[row][col];
    const occupantColor = occupant === 1 ? "BLACK" : occupant === -1 ? "WHITE" : null;
    if (!occupantColor) {
      continue;
    }
    if (humanPlayer && occupantColor === humanPlayer) {
      continue;
    }
    lastOpponentHighlight = landing.slice(0, 2);
    return;
  }
  lastOpponentHighlight = null;
}

function updateHint(state) {
  if (!isHumanTurn(state)) {
    hintElement.textContent = state.isGameOver ? "Replay the game or start a new match." : "Waiting for AI...";
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

  const highlightMatch = lastOpponentHighlight && arraysEqual(lastOpponentHighlight, [row, col]);
  if (highlightMatch) {
    const occupantColor = stone === 1 ? "BLACK" : stone === -1 ? "WHITE" : null;
    if (occupantColor && (!humanPlayer || occupantColor !== humanPlayer)) {
      const indicator = document.createElement("div");
      indicator.className = `last-opponent-indicator ${occupantColor.toLowerCase()}`;
      cell.appendChild(indicator);
    }
  }

  if (selectedSource && arraysEqual(selectedSource, [row, col])) {
    cell.classList.add("selected");
  }
}

// === 新增：读取 CSS 变量，得到像素数 ===
function readCssPx(varName, from = document.documentElement) {
  const v = getComputedStyle(from).getPropertyValue(varName).trim();
  return Number(v.replace('px', ''));
}

// === 新增：用 SVG 画 6x6（或任意 NxN）网格线 ===
// 规则：N 条竖线、N 条横线；每条线从第一格中心到最后一格中心。
// 因为 .board 有半格 padding，所以边/角交点不会“延伸出棋盘”。
function drawGridSvg(size) {
  const svg = document.getElementById("grid-svg");
  if (!svg) return;

  while (svg.firstChild) {
    svg.removeChild(svg.firstChild);
  }

  const board = document.getElementById("board");
  if (!board) return;

  const cellSize = readCssPx("--cell-size") || 0;
  const lineWidth = readCssPx("--line-width") || 2;
  const boardBorder = readCssPx("--board-border") || 0;
  const lineColor =
    getComputedStyle(document.documentElement).getPropertyValue("--line-color").trim() ||
    "rgba(65,45,22,0.8)";

  const targetExtent = size * cellSize + cellSize + 2 * boardBorder;
  let width = board.clientWidth || targetExtent;
  let height = board.clientHeight || targetExtent;

  if ((!board.clientWidth || !board.clientHeight) && board.offsetParent === null) {
    requestAnimationFrame(() => drawGridSvg(size));
  }

  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.setAttribute("width", width);
  svg.setAttribute("height", height);

  const boardStyle = getComputedStyle(board);
  const paddingLeft = parseFloat(boardStyle.paddingLeft);
  const effectivePadding =
    Number.isFinite(paddingLeft) && paddingLeft > 0 ? paddingLeft : cellSize / 2;
  const pad = cellSize / 2 + effectivePadding;
  const last = pad + (size - 1) * cellSize;

  for (let i = 0; i < size; i += 1) {
    const x = pad + i * cellSize;
    const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
    line.setAttribute("x1", x);
    line.setAttribute("y1", pad);
    line.setAttribute("x2", x);
    line.setAttribute("y2", last);
    line.setAttribute("stroke", lineColor);
    line.setAttribute("stroke-width", lineWidth);
    line.setAttribute("shape-rendering", "crispEdges");
    svg.appendChild(line);
  }

  for (let i = 0; i < size; i += 1) {
    const y = pad + i * cellSize;
    const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
    line.setAttribute("x1", pad);
    line.setAttribute("y1", y);
    line.setAttribute("x2", last);
    line.setAttribute("y2", y);
    line.setAttribute("stroke", lineColor);
    line.setAttribute("stroke-width", lineWidth);
    line.setAttribute("shape-rendering", "crispEdges");
    svg.appendChild(line);
  }
}



function renderBoard(state) {
  boardElement.innerHTML = "";
  const size = state.board.length;
  document.documentElement.style.setProperty("--board-size", size);

  // 1) 先把 SVG 放回去（boardElement 清空会把 svg 也清掉）
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.id = 'grid-svg';
  svg.classList.add('grid-svg');
  boardElement.appendChild(svg);

  // 2) 再铺格子（保持你原来的点击/选中逻辑）
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

  // 3) 画坐标网格线
  drawGridSvg(size);
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

  refreshOpponentHighlight(data.aiMoves || []);
  renderBoard(currentState);
  updateStatus(currentState);
  updateMeta(data.meta || {});
  updateHint(currentState);
  updateEvaluation(data.evaluation || null);

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
    if (setupSection) {
      setupSection.classList.add("hidden");
    }
    if (gameSection) {
      gameSection.classList.remove("hidden");
    }
    if (data.aiMoves) {
      data.aiMoves.forEach((move) => addLogEntry("ai", move));
    }
    lastOpponentHighlight = null;
    updateGameState(data);
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

updateEvaluation(null);
