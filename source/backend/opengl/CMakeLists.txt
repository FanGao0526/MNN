if(MNN_OPENGL)
  FILE(GLOB MNN_OpenGL_SRC ${CMAKE_CURRENT_LIST_DIR}/*.cpp)
  add_library(MNNOpenGL OBJECT ${MNN_OpenGL_SRC} "${CMAKE_CURRENT_LIST_DIR}/AllShader.cpp")
  list(APPEND MNN_OBJECTS_TO_LINK $<TARGET_OBJECTS:MNNOpenGL>)
  target_include_directories(MNNOpenGL PRIVATE ${CMAKE_CURRENT_LIST_DIR}/)
  option(MNN_OPENGL_REGEN "Regenerate OpenGL Shaders." OFF)
  IF(MNN_OPENGL_REGEN)
    add_custom_command(OUTPUT "${CMAKE_CURRENT_LIST_DIR}/AllShader.cpp"
      COMMAND ${PYTHON_EXECUTABLE}
      "${CMAKE_CURRENT_LIST_DIR}/makeshader.py"
      "${CMAKE_CURRENT_LIST_DIR}/glsl/"
      "${CMAKE_SOURCE_DIR}/include/MNN/backend/opengl/shaders/AllShader.h"
      "${CMAKE_CURRENT_LIST_DIR}/AllShader.cpp"
      COMMENT "OpenGL Code Generation"
    )
    add_custom_target (MNNOpenGLCodeGen DEPENDS "${CMAKE_CURRENT_LIST_DIR}/AllShader.cpp")
    add_dependencies(MNNOpenGL MNNOpenGLCodeGen)
  ENDIF()
endif()
