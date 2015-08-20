################################################################
################################################################
CMAKE_MINIMUM_REQUIRED (VERSION 2.8)



###############################################
SET (target "image_morphy")
PROJECT (${target} CXX)

# define source files
set(src Morphology_1.cpp)


#required packages
FIND_PACKAGE (OpenCV 2.4.0 REQUIRED)

INCLUDE_DIRECTORIES (${OPENCV_INCLUDE_DIR})
ADD_EXECUTABLE (${target} ${src})
TARGET_LINK_LIBRARIES (${target} ${OpenCV_LIBS} )

# set linker language
SET_TARGET_PROPERTIES(
	${target} 
	PROPERTIES 
	LINKER_LANGUAGE CXX)


