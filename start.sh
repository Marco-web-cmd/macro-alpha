#!/bin/zsh
# Lance macro_alpha proprement avec le bon venv

VENV_PYTHON="$HOME/alpha_trading/venv/bin/python"
APP_DIR="$HOME/macro_alpha"
PORT=5001

# Tuer tout process sur le port cible
PID=$(lsof -ti:$PORT 2>/dev/null)
if [ -n "$PID" ]; then
  echo "Port $PORT occupé par PID $PID — arrêt..."
  kill -9 $PID 2>/dev/null
  sleep 1
fi

# Vérif venv
if [ ! -f "$VENV_PYTHON" ]; then
  echo "ERREUR: venv introuvable à $VENV_PYTHON"
  exit 1
fi

cd "$APP_DIR"
echo "Démarrage avec $VENV_PYTHON..."
exec "$VENV_PYTHON" app.py
