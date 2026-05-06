#!/bin/bash
# ═══════════════════════════════════════════════════════════════
#  deploy_gcp.sh — Déploiement macro_alpha sur Google Cloud
#  VM e2-micro Always Free (gratuit à vie)
#  Usage : bash deploy_gcp.sh
# ═══════════════════════════════════════════════════════════════
set -e

PROJECT_ID="${GCP_PROJECT:-$(gcloud config get-value project 2>/dev/null)}"
INSTANCE="macro-alpha-bot"
ZONE="us-central1-a"        # zone Always Free obligatoire
MACHINE="e2-micro"          # Always Free
IMAGE="ubuntu-2204-jammy-v20240223"
IMAGE_PROJECT="ubuntu-os-cloud"
APP_DIR="/opt/macro_alpha"
PORT=5001

# ── Vérifie gcloud ────────────────────────────────────────────
if ! command -v gcloud &>/dev/null; then
  echo "❌  gcloud CLI non installé."
  echo "    Installe-le : brew install --cask google-cloud-sdk"
  exit 1
fi

if [ -z "$PROJECT_ID" ]; then
  echo "❌  Aucun projet GCP configuré."
  echo "    Lance : gcloud auth login && gcloud projects list"
  echo "    Puis  : gcloud config set project TON_PROJECT_ID"
  exit 1
fi

echo "🚀  Projet : $PROJECT_ID"
echo "🖥️   Instance : $INSTANCE ($MACHINE) — $ZONE"

# ── 1. Créer la VM si elle n'existe pas ───────────────────────
if gcloud compute instances describe "$INSTANCE" --zone="$ZONE" --project="$PROJECT_ID" &>/dev/null; then
  echo "✅  VM déjà existante — déploiement du code uniquement"
else
  echo "🔨  Création de la VM…"
  gcloud compute instances create "$INSTANCE" \
    --project="$PROJECT_ID" \
    --zone="$ZONE" \
    --machine-type="$MACHINE" \
    --image="$IMAGE" \
    --image-project="$IMAGE_PROJECT" \
    --boot-disk-size=30GB \
    --boot-disk-type=pd-standard \
    --tags=macro-alpha \
    --metadata=startup-script='#!/bin/bash
apt-get update -qq
apt-get install -y -qq python3.11 python3.11-venv python3-pip gcc g++ rsync
' 2>&1
  echo "⏳  Attente démarrage VM (30s)…"
  sleep 30
fi

# ── 2. Règle firewall ─────────────────────────────────────────
if ! gcloud compute firewall-rules describe "allow-macro-alpha" --project="$PROJECT_ID" &>/dev/null; then
  echo "🔓  Création règle firewall port $PORT…"
  gcloud compute firewall-rules create "allow-macro-alpha" \
    --project="$PROJECT_ID" \
    --allow="tcp:$PORT" \
    --target-tags="macro-alpha" \
    --description="macro_alpha dashboard"
fi

# ── 3. IP publique ────────────────────────────────────────────
IP=$(gcloud compute instances describe "$INSTANCE" \
  --zone="$ZONE" \
  --project="$PROJECT_ID" \
  --format='get(networkInterfaces[0].accessConfigs[0].natIP)')
echo "🌐  IP publique : $IP"

# ── 4. Copie via tar + SSH ────────────────────────────────────
echo "📦  Envoi des fichiers…"
SSH="gcloud compute ssh $INSTANCE --zone=$ZONE --project=$PROJECT_ID --command"

# Créer le dossier
$SSH "sudo mkdir -p $APP_DIR && sudo chown -R \$USER:\$USER $APP_DIR"

# Archiver localement et envoyer en une fois
tar czf /tmp/macro_alpha_deploy.tar.gz \
  --exclude='venv' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='data' \
  --exclude='.DS_Store' \
  --exclude='*.swp' \
  --exclude='.env' \
  -C .. macro_alpha/

gcloud compute scp --zone="$ZONE" --project="$PROJECT_ID" \
  /tmp/macro_alpha_deploy.tar.gz "$INSTANCE:/tmp/deploy.tar.gz"

gcloud compute scp --zone="$ZONE" --project="$PROJECT_ID" \
  .env "$INSTANCE:/tmp/.env"

$SSH "tar xzf /tmp/deploy.tar.gz -C /tmp && \
  cp -r /tmp/macro_alpha/* $APP_DIR/ && \
  cp /tmp/.env $APP_DIR/.env && \
  mkdir -p $APP_DIR/data/cache $APP_DIR/data/logs && \
  rm -rf /tmp/macro_alpha /tmp/deploy.tar.gz /tmp/.env"

# ── 5. Install + service systemd ──────────────────────────────
echo "⚙️   Configuration…"
gcloud compute ssh "$INSTANCE" \
  --zone="$ZONE" \
  --project="$PROJECT_ID" \
  --command='
set -e
sudo apt-get update -qq
sudo apt-get install -y -qq python3.11 python3.11-venv python3-pip gcc g++

sudo mkdir -p /opt/macro_alpha
sudo chown -R $USER:$USER /opt/macro_alpha
cd /opt/macro_alpha

python3.11 -m venv venv
venv/bin/pip install --upgrade pip -q
grep -v "pandas-ta" requirements.txt > /tmp/req_server.txt
venv/bin/pip install -r /tmp/req_server.txt -q
venv/bin/pip install pandas-ta --no-deps -q 2>/dev/null || true
mkdir -p data/cache data/logs

# Service systemd
sudo tee /etc/systemd/system/macro_alpha.service > /dev/null << EOF
[Unit]
Description=macro_alpha trading bot
After=network.target

[Service]
Type=simple
User='$USER'
WorkingDirectory=/opt/macro_alpha
ExecStart=/opt/macro_alpha/venv/bin/python -m uvicorn app:app --host 0.0.0.0 --port 5001 --workers 1
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable macro_alpha
sudo systemctl restart macro_alpha
sleep 3
sudo systemctl status macro_alpha --no-pager
' 2>&1 | grep -v "^debconf:"

echo ""
echo "═══════════════════════════════════════════════"
echo "  ✅  Déploiement terminé !"
echo "  🌐  Dashboard : http://$IP:$PORT"
echo "  📋  Logs      : gcloud compute ssh $INSTANCE --zone=$ZONE --command='journalctl -u macro_alpha -f'"
echo "  🔄  Redémarrer: gcloud compute ssh $INSTANCE --zone=$ZONE --command='sudo systemctl restart macro_alpha'"
echo "  📤  Redéployer: bash deploy_gcp.sh"
echo "═══════════════════════════════════════════════"
