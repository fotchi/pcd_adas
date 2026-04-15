#!/bin/bash
# Démarre : CARLA + carla_adas_world.py + détection de voies en temps réel

WORKSPACE="/home/fedi/catkin_ws"
CARLA_PATH="${CARLA_PATH:-/home/fedi/carla}"

echo "=== ADAS Lane Detection avec carla_adas_world.py ==="

source "$WORKSPACE/devel/setup.bash"

# Cleanup au Ctrl-C
cleanup() {
    echo -e "\n Arrêt..."
    kill $ROSCORE_PID $CARLA_PID $WORLD_PID 2>/dev/null
    pkill -f CarlaUE4 2>/dev/null
    pkill -f carla_adas_world 2>/dev/null
    exit 0
}
trap cleanup EXIT SIGINT SIGTERM

# 0. Tuer les anciens processus CARLA/ROS qui bloquent le port 2000
echo "[0/4] Nettoyage des anciens processus..."
pkill -f CarlaUE4      2>/dev/null; sleep 1
pkill -f carla_adas    2>/dev/null
pkill -f rosmaster     2>/dev/null; sleep 1
fuser -k 2000/tcp      2>/dev/null; sleep 1
echo "OK"

# 1. roscore
echo "[1/4] Démarrage roscore..."
roscore &
ROSCORE_PID=$!
sleep 2

# 2. CARLA
echo "[2/4] Démarrage CARLA..."
cd "$CARLA_PATH"
./CarlaUE4.sh -world-port=2000 -quality-level=Low -fps=15 &
CARLA_PID=$!
echo "CARLA lancé (PID: $CARLA_PID) — attente 12s..."
sleep 12

# 3. carla_adas_world.py
echo "[3/4] Démarrage carla_adas_world.py..."
cd /home/fedi
python3 carla_adas_world.py &
WORLD_PID=$!
sleep 5

# 4. Lane detection + viewer
echo "[4/4] Démarrage lane detection..."
cd "$WORKSPACE"
roslaunch adas_pkg lane_only.launch
