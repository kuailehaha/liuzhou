@echo off
setlocal

set "ROOT=%~dp0.."
set "PYTHONPATH=%ROOT%\build\v0\src;%ROOT%"

echo [validate_v1_gpu] ROOT=%ROOT%
echo [validate_v1_gpu] PYTHONPATH=%PYTHONPATH%

conda run -n torchenv python "%ROOT%\tools\validate_v1_claims.py" ^
  --device cuda:0 ^
  --v0-workers 1,2,4 ^
  --v1-threads 1,2,4 ^
  --total-games 8 ^
  --v0-mcts-simulations 12 ^
  --v1-mcts-simulations 12 ^
  --v0-batch-leaves 32 ^
  --v0-inference-backend py ^
  --with-inference-baseline ^
  --inference-baseline-batch 4096 ^
  --inference-baseline-iters 120 ^
  --output-json "%ROOT%\results\v1_validation_latest.json" ^
  %*

endlocal

