# 작업 스크립트가 셸 상태에 의존하지 않도록 필요한 환경을 여기서 고정한다.
# `source scripts/env.sh` 로 쓴다.
#
# 왜 필요한가: 폭 스윕 30회가 CSV 를 하나도 남기지 못하고 24분을 태운 적이 있다.
#   원인은 OMNI_KIT_ACCEPT_EULA 누락이었다. Isaac Sim 은 EULA 를 대화형으로 묻는데
#   nohup/백그라운드에는 stdin 이 없어 즉시 죽는다:
#     "Unable to bootstrap inner kit kernel: EOF when reading a line"
#   기존 세션 셸에는 이 변수가 있었고 새 셸에는 없어서, 같은 스크립트가
#   셸에 따라 되기도 안 되기도 했다. 스크립트가 스스로 갖고 있어야 한다.
#
# ⚠ $HOME/.isaac_cache_env 는 setup_cache.sh 가 만들지만, 먼저 만들어진 파일에는
#   EULA 줄이 없다. 그래서 source 한 뒤 다시 한 번 명시적으로 export 한다.

[ -f "$HOME/.isaac_cache_env" ] && . "$HOME/.isaac_cache_env"

# 헤드리스 필수 — setup.sh / setup_cache.sh 와 같은 값이다.
export OMNI_KIT_ACCEPT_EULA=YES

# `python` 이 PATH 에 없는 셸에서도 돌아야 한다.
PY=${PY:-$HOME/env_isaaclab/bin/python}
[ -x "$PY" ] || PY=$(command -v python || command -v python3)
export PY
