
CMAKE_MINIMUM_REQUIRED(VERSION 2.6 FATAL_ERROR)
CMAKE_POLICY(VERSION 2.6)
FIND_PACKAGE(Torch REQUIRED)
FIND_PACKAGE(OpenMP)

IF (OPENMP_FOUND)
  MESSAGE (STATUS "OpenMP Found with compiler flag : ${OpenMP_C_FLAGS}")
  SET (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
ENDIF (OPENMP_FOUND)

SET(src init.c)
FILE(GLOB luasrc *.lua)
SET(luasrc ${luasrc} test/test.lua)
ADD_TORCH_PACKAGE(nn2 "${src}" "${luasrc}")
ADD_TORCH_DOK(dok nn2 "Machine Learning" "Neural Networks" 3.1)
TARGET_LINK_LIBRARIES(nn2 luaT TH)
