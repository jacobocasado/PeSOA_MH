# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.19

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Disable VCS-based implicit rules.
% : %,v


# Disable VCS-based implicit rules.
% : RCS/%


# Disable VCS-based implicit rules.
% : RCS/%,v


# Disable VCS-based implicit rules.
% : SCCS/s.%


# Disable VCS-based implicit rules.
% : s.%


.SUFFIXES: .hpux_make_needs_suffix_list


# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /opt/clion/bin/cmake/linux/bin/cmake

# The command to remove a file.
RM = /opt/clion/bin/cmake/linux/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/jcasado/Documentos/PeSOA_jacobo/code

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/jcasado/Documentos/PeSOA_jacobo/code

# Include any dependencies generated for this target.
include CMakeFiles/testjacobocambios.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/testjacobocambios.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/testjacobocambios.dir/flags.make

CMakeFiles/testjacobocambios.dir/testjacobocambios.cc.o: CMakeFiles/testjacobocambios.dir/flags.make
CMakeFiles/testjacobocambios.dir/testjacobocambios.cc.o: testjacobocambios.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jcasado/Documentos/PeSOA_jacobo/code/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/testjacobocambios.dir/testjacobocambios.cc.o"
	/sbin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/testjacobocambios.dir/testjacobocambios.cc.o -c /home/jcasado/Documentos/PeSOA_jacobo/code/testjacobocambios.cc

CMakeFiles/testjacobocambios.dir/testjacobocambios.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/testjacobocambios.dir/testjacobocambios.cc.i"
	/sbin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jcasado/Documentos/PeSOA_jacobo/code/testjacobocambios.cc > CMakeFiles/testjacobocambios.dir/testjacobocambios.cc.i

CMakeFiles/testjacobocambios.dir/testjacobocambios.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/testjacobocambios.dir/testjacobocambios.cc.s"
	/sbin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jcasado/Documentos/PeSOA_jacobo/code/testjacobocambios.cc -o CMakeFiles/testjacobocambios.dir/testjacobocambios.cc.s

# Object files for target testjacobocambios
testjacobocambios_OBJECTS = \
"CMakeFiles/testjacobocambios.dir/testjacobocambios.cc.o"

# External object files for target testjacobocambios
testjacobocambios_EXTERNAL_OBJECTS =

testjacobocambios: CMakeFiles/testjacobocambios.dir/testjacobocambios.cc.o
testjacobocambios: CMakeFiles/testjacobocambios.dir/build.make
testjacobocambios: libcec17_test_func.so
testjacobocambios: CMakeFiles/testjacobocambios.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/jcasado/Documentos/PeSOA_jacobo/code/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable testjacobocambios"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/testjacobocambios.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/testjacobocambios.dir/build: testjacobocambios

.PHONY : CMakeFiles/testjacobocambios.dir/build

CMakeFiles/testjacobocambios.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/testjacobocambios.dir/cmake_clean.cmake
.PHONY : CMakeFiles/testjacobocambios.dir/clean

CMakeFiles/testjacobocambios.dir/depend:
	cd /home/jcasado/Documentos/PeSOA_jacobo/code && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/jcasado/Documentos/PeSOA_jacobo/code /home/jcasado/Documentos/PeSOA_jacobo/code /home/jcasado/Documentos/PeSOA_jacobo/code /home/jcasado/Documentos/PeSOA_jacobo/code /home/jcasado/Documentos/PeSOA_jacobo/code/CMakeFiles/testjacobocambios.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/testjacobocambios.dir/depend

