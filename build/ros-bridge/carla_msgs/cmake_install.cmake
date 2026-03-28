# Install script for directory: /home/fedi/catkin_ws/src/ros-bridge/carla_msgs

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/fedi/catkin_ws/install")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/carla_msgs/srv" TYPE FILE FILES
    "/home/fedi/catkin_ws/src/ros-bridge/carla_msgs/srv/SpawnObject.srv"
    "/home/fedi/catkin_ws/src/ros-bridge/carla_msgs/srv/DestroyObject.srv"
    "/home/fedi/catkin_ws/src/ros-bridge/carla_msgs/srv/GetBlueprints.srv"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/carla_msgs/msg" TYPE FILE FILES
    "/home/fedi/catkin_ws/src/ros-bridge/carla_msgs/msg/CarlaBoundingBox.msg"
    "/home/fedi/catkin_ws/src/ros-bridge/carla_msgs/msg/CarlaEgoVehicleControl.msg"
    "/home/fedi/catkin_ws/src/ros-bridge/carla_msgs/msg/CarlaEgoVehicleStatus.msg"
    "/home/fedi/catkin_ws/src/ros-bridge/carla_msgs/msg/CarlaEgoVehicleInfoWheel.msg"
    "/home/fedi/catkin_ws/src/ros-bridge/carla_msgs/msg/CarlaEgoVehicleInfo.msg"
    "/home/fedi/catkin_ws/src/ros-bridge/carla_msgs/msg/CarlaCollisionEvent.msg"
    "/home/fedi/catkin_ws/src/ros-bridge/carla_msgs/msg/CarlaLaneInvasionEvent.msg"
    "/home/fedi/catkin_ws/src/ros-bridge/carla_msgs/msg/CarlaWorldInfo.msg"
    "/home/fedi/catkin_ws/src/ros-bridge/carla_msgs/msg/CarlaActorInfo.msg"
    "/home/fedi/catkin_ws/src/ros-bridge/carla_msgs/msg/CarlaActorList.msg"
    "/home/fedi/catkin_ws/src/ros-bridge/carla_msgs/msg/CarlaControl.msg"
    "/home/fedi/catkin_ws/src/ros-bridge/carla_msgs/msg/CarlaStatus.msg"
    "/home/fedi/catkin_ws/src/ros-bridge/carla_msgs/msg/CarlaTrafficLightInfo.msg"
    "/home/fedi/catkin_ws/src/ros-bridge/carla_msgs/msg/CarlaTrafficLightInfoList.msg"
    "/home/fedi/catkin_ws/src/ros-bridge/carla_msgs/msg/CarlaTrafficLightStatus.msg"
    "/home/fedi/catkin_ws/src/ros-bridge/carla_msgs/msg/CarlaTrafficLightStatusList.msg"
    "/home/fedi/catkin_ws/src/ros-bridge/carla_msgs/msg/CarlaWalkerControl.msg"
    "/home/fedi/catkin_ws/src/ros-bridge/carla_msgs/msg/CarlaWeatherParameters.msg"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/carla_msgs/cmake" TYPE FILE FILES "/home/fedi/catkin_ws/build/ros-bridge/carla_msgs/catkin_generated/installspace/carla_msgs-msg-paths.cmake")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/home/fedi/catkin_ws/devel/include/carla_msgs")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/roseus/ros" TYPE DIRECTORY FILES "/home/fedi/catkin_ws/devel/share/roseus/ros/carla_msgs")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/common-lisp/ros" TYPE DIRECTORY FILES "/home/fedi/catkin_ws/devel/share/common-lisp/ros/carla_msgs")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/gennodejs/ros" TYPE DIRECTORY FILES "/home/fedi/catkin_ws/devel/share/gennodejs/ros/carla_msgs")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  execute_process(COMMAND "/usr/bin/python3" -m compileall "/home/fedi/catkin_ws/devel/lib/python3/dist-packages/carla_msgs")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/python3/dist-packages" TYPE DIRECTORY FILES "/home/fedi/catkin_ws/devel/lib/python3/dist-packages/carla_msgs")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/pkgconfig" TYPE FILE FILES "/home/fedi/catkin_ws/build/ros-bridge/carla_msgs/catkin_generated/installspace/carla_msgs.pc")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/carla_msgs/cmake" TYPE FILE FILES "/home/fedi/catkin_ws/build/ros-bridge/carla_msgs/catkin_generated/installspace/carla_msgs-msg-extras.cmake")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/carla_msgs/cmake" TYPE FILE FILES
    "/home/fedi/catkin_ws/build/ros-bridge/carla_msgs/catkin_generated/installspace/carla_msgsConfig.cmake"
    "/home/fedi/catkin_ws/build/ros-bridge/carla_msgs/catkin_generated/installspace/carla_msgsConfig-version.cmake"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/carla_msgs" TYPE FILE FILES "/home/fedi/catkin_ws/src/ros-bridge/carla_msgs/package.xml")
endif()

