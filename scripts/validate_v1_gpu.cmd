@echo off
setlocal

set "ROOT=%~dp0.."
set "PYTHONPATH=%ROOT%\build\v0\src;%ROOT%"

echo [validate_v1_gpu] ROOT=%ROOT%
echo [validate_v1_gpu] PYTHONPATH=%PYTHONPATH%

conda run -n torchenv python "%ROOT%\tools\validate_v1_claims.py" ^
  --device cuda:0 ^
  --seed 12345 ^
  --rounds 1 ^
  --v0-workers 1,2,4 ^
  --v1-threads 1,2,4 ^
  --v1-concurrent-games 8 ^
  --total-games 8 ^
  --v0-mcts-simulations 24 ^
  --v1-mcts-simulations 24 ^
  --v0-batch-leaves 512 ^
  --v0-inference-backend graph ^
  --v0-inference-batch-size 512 ^
  --v0-inference-warmup-iters 5 ^
  --v0-opening-random-moves 2 ^
  --v0-resign-threshold -0.8 ^
  --v0-resign-min-moves 36 ^
  --v0-resign-consecutive 3 ^
  --with-inference-baseline ^
  --inference-baseline-batch 4096 ^
  --inference-baseline-iters 120 ^
  --output-json "%ROOT%\results\v1_validation_latest.json" ^
  %*

endlocal
