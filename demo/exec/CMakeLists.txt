IF(MNN_BUILD_DEMO)
  add_executable(pictureRecognition.out ${CMAKE_CURRENT_LIST_DIR}/pictureRecognition.cpp)
  target_link_libraries(pictureRecognition.out MNN)
  add_executable(pictureRotate.out ${CMAKE_CURRENT_LIST_DIR}/pictureRotate.cpp)
  target_link_libraries(pictureRotate.out MNN)

  add_executable(multiPose.out ${CMAKE_CURRENT_LIST_DIR}/multiPose.cpp)
  target_link_libraries(multiPose.out MNN)

  add_executable(segment.out ${CMAKE_CURRENT_LIST_DIR}/segment.cpp)
  target_link_libraries(segment.out MNN)

  add_executable(expressDemo.out ${CMAKE_CURRENT_LIST_DIR}/expressDemo.cpp)
  target_link_libraries(expressDemo.out MNN)

ENDIF()
