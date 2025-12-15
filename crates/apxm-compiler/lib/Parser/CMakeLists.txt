# Parser library - handles lexing, parsing, and AST generation

# Collect parser source files
file(GLOB_RECURSE APXM_PARSER_SOURCES CONFIGURE_DEPENDS
  ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
)

# Set output variable for parent CMakeLists
set(APXM_PARSER_SOURCES ${APXM_PARSER_SOURCES} PARENT_SCOPE)
