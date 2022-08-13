cmake_minimum_required(VERSION 3.5)

project(MLP VERSION 0.1 LANGUAGES CXX)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_PREFIX_PATH "~/Qt/6.3.1/gcc_64")

find_package(QT NAMES Qt6 Qt5 REQUIRED)
find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS Widgets)
find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS Core)
find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS Concurrent)
find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS PrintSupport)


#find_package(OpenGL REQUIRED)
#find_package(GLUT REQUIRED)
include_directories( ${OPENGL_INCLUDE_DIRS}  ${GLUT_INCLUDE_DIRS} )

set(PROJECT_SOURCES
    model/matrix/s21_matrix.cpp
    model/tests.cpp
    model/CNN.cpp
)


# if(${QT_VERSION_MAJOR} GREATER_EQUAL 6)
#     qt_add_executable(MLP MANUAL_FINALIZATION ${PROJECT_SOURCES} ${QRC_SOURCES})
# else()
#     add_executable(MLP ${PROJECT_SOURCES} ${QRC_SOURCES})
# endif()

# target_link_libraries(MLP PRIVATE Qt${QT_VERSION_MAJOR}::Widgets ${OPENGL_LIBRARIES} ${GLUT_LIBRARY} )
# target_link_libraries(MLP PRIVATE Qt${QT_VERSION_MAJOR}::Core)
# target_link_libraries(MLP PRIVATE Qt${QT_VERSION_MAJOR}::Concurrent)
# target_link_libraries(MLP PRIVATE Qt${QT_VERSION_MAJOR}::PrintSupport)

set_target_properties(MLP PROPERTIES
        MACOSX_BUNDLE_GUI_IDENTIFIER my.example.com
        MACOSX_BUNDLE_BUNDLE_VERSION ${PROJECT_VERSION}
        MACOSX_BUNDLE_SHORT_VERSION_STRING ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}
        MACOSX_BUNDLE TRUE
        WIN32_EXECUTABLE TRUE
        )

if(QT_VERSION_MAJOR EQUAL 6)
    qt_finalize_executable(MLP)
endif()