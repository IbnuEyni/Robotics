cmake_minimum_required(VERSION  3.5)
project(robotic_arm)

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)

# Declare the messages to be generated
ament_export_dependencies(
  builtin_interfaces
  geometry_msgs
  sensor_msgs
)

# Add the message files to the build
add_message_files(
  FILES
    msg/DetectedObject.msg
    msg/DetectedObjectsStamped.msg
    # ... include all your .msg files here ...
)

# Generate the messages
generate_messages(
  DEPENDENCIES
    builtin_interfaces
    geometry_msgs
    sensor_msgs
)

# Declare the package
ament_package()
