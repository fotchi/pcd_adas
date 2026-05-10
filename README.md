# ADAS — Autonomous Driving System with CARLA & ROS

> Full-stack Advanced Driver Assistance System (ADAS) built with **ROS Noetic** and the **CARLA simulator**.  
> The vehicle combines **lane detection**, **object detection**, and **radar-based perception** to make real-time autonomous driving decisions through a priority-based state machine.

---

# Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Key Features](#key-features)
- [Detected Classes](#detected-classes)
- [Driving Scenarios](#driving-scenarios)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [ROS Topics](#ros-topics)
- [Model Training](#model-training)
- [Project Structure](#project-structure)
- [License](#license)

---

# Overview

This project implements a modular ADAS pipeline running inside the CARLA simulator using the official `ros-bridge`.

The system combines:

- **Classical computer vision** for lane detection
- **Deep learning** with YOLOv8 for object recognition
- **Radar-based obstacle detection**
- **Priority-based decision making**
- **Vehicle control using a PD controller**

The architecture is fully modular and designed around ROS nodes for easy testing, debugging, and scalability.

---

# System Architecture

```text
┌─────────────────────────────────────────────────────────────────────┐
│                         CARLA Simulator                             │
└────────────┬──────────────────────────────────┬────────────────────┘
             │                                  │
     /carla/camera/rgb                /carla/ego_vehicle/radar_front
             │                                  │
    ┌────────▼──────────┐           ┌────────────▼────────────┐
    │  lane_viewer_node │           │    radar_bridge_node     │
    └────────┬──────────┘           └────────────┬────────────┘
             │                                   │
    /lane/deviation                   /carla/radar/front
    /lane/status                                 │
             │                                   │
    ┌────────▼──────────┐                        │
    │ object_detection  │                        │
    │       node        │                        │
    └────────┬──────────┘                        │
             │                                   │
       /adas/detection                           │
             │                                   │
    ┌────────▼───────────────────────────────────▼────────────┐
    │                    decision_node                         │
    │     P0: Traffic Signals → P1: Radar → P2: Lane Keep     │
    └────────────────────────┬────────────────────────────────┘
                             │
                      /adas/control
                             │
    ┌────────────────────────▼────────────────────────────────┐
    │                  carla_adas_world                        │
    │              PD Controller + Actuators                   │
    └─────────────────────────────────────────────────────────┘
```

---

# Key Features

| Module | Technique | Published Topics |
|---|---|---|
| **Lane Detection** | HLS thresholding, sliding window tracking, polynomial fitting, EMA smoothing | `/lane/deviation`, `/lane/status`, `/lane/image` |
| **Object Detection** | Dual YOLOv8 models (`best-11.pt` + `yolov8n.pt`) with temporal voting | `/adas/detection`, `/adas/detection_image` |
| **Radar Processing** | PointCloud2 → minimum obstacle distance | `/carla/radar/front` |
| **Decision Engine** | Priority-based finite state machine | `/adas/control`, `/adas/decision` |
| **Vehicle Control** | PD steering controller + adaptive throttle | CARLA actuators |

---

# Detected Classes

## Custom YOLOv8 Model — `best-11.pt`

Custom-trained model for traffic signs and traffic lights.

| Label | Description |
|---|---|
| `red_light` | Red traffic light |
| `green_light` | Green traffic light |
| `stop_sign` | Stop sign |
| `speed_limit_10` | Speed limit 10 km/h |
| `speed_limit_30` | Speed limit 30 km/h |
| `turn_right` | Turn-right sign |

### Detection Pipeline

- Input size: **896×896**
- HSV-based traffic light verification
- 8-frame temporal voting
- 75% consensus validation

---

## COCO YOLOv8 Nano — `yolov8n.pt`

Pretrained model used for dynamic obstacle detection.

| Label | Description |
|---|---|
| `pieton` | Pedestrian |
| `voiture` | Vehicle |

### Configuration

- Input size: **640×640**
- Confidence threshold: **0.40**

---

# Driving Scenarios

The ADAS pipeline supports six autonomous driving scenarios:

```text
Init → Free Drive → S1 → S2 → S3 → S4 → S5 → S6
```

| Scenario | Trigger | Vehicle Behavior |
|---|---|---|
| **S1 — Stop Sign** | `stop_sign` | Full stop with hold timer |
| **S2 — Traffic Light** | `red_light` / `green_light` | Emergency brake on red, resume on green |
| **S3 — Speed Limit 10** | `speed_limit_10` | Cruise control ≤ 10.5 km/h |
| **S4 — Pedestrian Crossing** | Radar + pedestrian detection | Slowdown and braking |
| **S5 — Stationary Vehicle** | Radar obstacle detection | Adaptive braking |
| **S6 — Right Turn** | `turn_right` + `NO_LANE` | Multi-phase turning maneuver |

---

# Prerequisites

## System Requirements

- Ubuntu 20.04
- ROS Noetic
- CARLA 0.9.12+
- Python 3.8+

---

## Python Dependencies

```bash
pip install ultralytics opencv-python numpy
```

---

## ROS Dependencies

```bash
sudo apt install \
ros-noetic-cv-bridge \
ros-noetic-sensor-msgs \
ros-noetic-std-msgs
```

---

# Installation

## 1. Clone the Repository

```bash
git clone --recurse-submodules https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

---

## 2. Build the Workspace

```bash
catkin_make
source devel/setup.bash
```

---

## 3. Add Model Weights

Place the custom YOLO weights inside:

```bash
src/adas_pkg/src/
```

```bash
cp best-11.pt src/adas_pkg/src/
```

`yolov8n.pt` will be automatically downloaded by Ultralytics during the first execution.

> `best-11.pt` is intentionally excluded from git tracking.

---

# Usage

## 1. Launch CARLA

```bash
./CarlaUE4.sh
```

---

## 2. Start the ADAS Stack

```bash
roslaunch adas_pkg adas_complete.launch
```

This launches:

- Lane detection node
- Radar bridge node
- Object detection node
- Decision engine node

All nodes support automatic respawn.

---

## 3. Start Vehicle Control Interface

```bash
rosrun adas_pkg carla_adas_world.py
```

---

# ROS Topics

## Published Topics

| Topic | Type | Description |
|---|---|---|
| `/carla/camera/rgb` | `sensor_msgs/Image` | Front RGB camera stream |
| `/carla/speed` | `std_msgs/Float32` | Vehicle speed |
| `/carla/radar/front` | `std_msgs/Float32` | Minimum radar obstacle distance |
| `/lane/deviation` | `std_msgs/Float32` | Lane center deviation |
| `/lane/status` | `std_msgs/String` | Lane state |
| `/lane/image` | `sensor_msgs/Image` | Annotated lane frame |
| `/adas/detection` | `std_msgs/String` | Detection results |
| `/adas/detection_image` | `sensor_msgs/Image` | Annotated detection frame |
| `/adas/control` | `std_msgs/String` | Control commands |
| `/adas/decision` | `std_msgs/String` | Decision explanations |

---

## Control Commands

```text
GO
SLOW
BRAKE
LIMIT_10
LIMIT_20
STEER_LEFT
STEER_RIGHT
TURN_RIGHT
```

---

## Detection Message Format

```text
<class>:<detail>:<confidence>:<x1>:<y1>:<x2>:<y2>
```

Example:

```text
red_light:red:0.97:280:60:400:200
```

---

# Model Training

The custom traffic-sign model was trained on **Kaggle** using YOLOv8.

| Parameter | Value |
|---|---|
| Architecture | YOLOv8n |
| Input Size | 416×416 |
| Training Platform | Kaggle GPU T4*2 |
| Number of Classes | 8 |
| Traffic Light Threshold | 0.62 |
| Sign Threshold | 0.04 |
| Post-processing | HSV filtering + temporal voting |

---

# Project Structure

```text
catkin_ws/
├── src/
│   ├── adas_pkg/
│   │   ├── src/
│   │   │   ├── carla_adas_world.py
│   │   │   ├── decision_node.py
│   │   │   ├── lane_viewer_node.py
│   │   │   ├── object_detection_node.py
│   │   │   ├── radar_bridge_node.py
│   │   │   ├── best-11.pt
│   │   │   ├── yolov8n.pt
│   │   │   └── adas_signs/
│   │   ├── launch/
│   │   │   └── adas_complete.launch
│   │   ├── docs/
│   │   └── package.xml
│   └── ros-bridge/
└── README.md
```

---

# License

This project is released under the **MIT License**.

See the `LICENSE` file for more details.

---

<p align="center">
  ROS Noetic • CARLA 0.9.12 • YOLOv8 • Python 3.8
</p>
