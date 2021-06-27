# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.19

# Default target executed when no arguments are given to make.
default_target: all

.PHONY : default_target

# Allow only one "make -f Makefile2" at a time, but pass parallelism.
.NOTPARALLEL:


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

#=============================================================================
# Targets provided globally by CMake.

# Special rule for the target edit_cache
edit_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running CMake cache editor..."
	/usr/bin/ccmake -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : edit_cache

# Special rule for the target edit_cache
edit_cache/fast: edit_cache

.PHONY : edit_cache/fast

# Special rule for the target rebuild_cache
rebuild_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running CMake to regenerate build system..."
	/opt/clion/bin/cmake/linux/bin/cmake --regenerate-during-build -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : rebuild_cache

# Special rule for the target rebuild_cache
rebuild_cache/fast: rebuild_cache

.PHONY : rebuild_cache/fast

# The main all target
all: cmake_check_build_system
	$(CMAKE_COMMAND) -E cmake_progress_start /home/jcasado/Documentos/PeSOA_jacobo/code/CMakeFiles /home/jcasado/Documentos/PeSOA_jacobo/code//CMakeFiles/progress.marks
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 all
	$(CMAKE_COMMAND) -E cmake_progress_start /home/jcasado/Documentos/PeSOA_jacobo/code/CMakeFiles 0
.PHONY : all

# The main clean target
clean:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 clean
.PHONY : clean

# The main clean target
clean/fast: clean

.PHONY : clean/fast

# Prepare targets for installation.
preinstall: all
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall

# Prepare targets for installation.
preinstall/fast:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall/fast

# clear depends
depend:
	$(CMAKE_COMMAND) -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 1
.PHONY : depend

#=============================================================================
# Target rules for targets named test

# Build rule for target.
test: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 test
.PHONY : test

# fast build rule for target.
test/fast:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/test.dir/build.make CMakeFiles/test.dir/build
.PHONY : test/fast

#=============================================================================
# Target rules for targets named cec17_test_func

# Build rule for target.
cec17_test_func: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 cec17_test_func
.PHONY : cec17_test_func

# fast build rule for target.
cec17_test_func/fast:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/cec17_test_func.dir/build.make CMakeFiles/cec17_test_func.dir/build
.PHONY : cec17_test_func/fast

#=============================================================================
# Target rules for targets named PeSOA

# Build rule for target.
PeSOA: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 PeSOA
.PHONY : PeSOA

# fast build rule for target.
PeSOA/fast:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/PeSOA.dir/build.make CMakeFiles/PeSOA.dir/build
.PHONY : PeSOA/fast

#=============================================================================
# Target rules for targets named testrandom

# Build rule for target.
testrandom: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 testrandom
.PHONY : testrandom

# fast build rule for target.
testrandom/fast:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/testrandom.dir/build.make CMakeFiles/testrandom.dir/build
.PHONY : testrandom/fast

#=============================================================================
# Target rules for targets named testsolis

# Build rule for target.
testsolis: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 testsolis
.PHONY : testsolis

# fast build rule for target.
testsolis/fast:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/testsolis.dir/build.make CMakeFiles/testsolis.dir/build
.PHONY : testsolis/fast

PeSOA.o: PeSOA.cc.o

.PHONY : PeSOA.o

# target to build an object file
PeSOA.cc.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/PeSOA.dir/build.make CMakeFiles/PeSOA.dir/PeSOA.cc.o
.PHONY : PeSOA.cc.o

PeSOA.i: PeSOA.cc.i

.PHONY : PeSOA.i

# target to preprocess a source file
PeSOA.cc.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/PeSOA.dir/build.make CMakeFiles/PeSOA.dir/PeSOA.cc.i
.PHONY : PeSOA.cc.i

PeSOA.s: PeSOA.cc.s

.PHONY : PeSOA.s

# target to generate assembly for a file
PeSOA.cc.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/PeSOA.dir/build.make CMakeFiles/PeSOA.dir/PeSOA.cc.s
.PHONY : PeSOA.cc.s

cec17.o: cec17.c.o

.PHONY : cec17.o

# target to build an object file
cec17.c.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/cec17_test_func.dir/build.make CMakeFiles/cec17_test_func.dir/cec17.c.o
.PHONY : cec17.c.o

cec17.i: cec17.c.i

.PHONY : cec17.i

# target to preprocess a source file
cec17.c.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/cec17_test_func.dir/build.make CMakeFiles/cec17_test_func.dir/cec17.c.i
.PHONY : cec17.c.i

cec17.s: cec17.c.s

.PHONY : cec17.s

# target to generate assembly for a file
cec17.c.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/cec17_test_func.dir/build.make CMakeFiles/cec17_test_func.dir/cec17.c.s
.PHONY : cec17.c.s

cec17_test_func.o: cec17_test_func.c.o

.PHONY : cec17_test_func.o

# target to build an object file
cec17_test_func.c.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/cec17_test_func.dir/build.make CMakeFiles/cec17_test_func.dir/cec17_test_func.c.o
.PHONY : cec17_test_func.c.o

cec17_test_func.i: cec17_test_func.c.i

.PHONY : cec17_test_func.i

# target to preprocess a source file
cec17_test_func.c.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/cec17_test_func.dir/build.make CMakeFiles/cec17_test_func.dir/cec17_test_func.c.i
.PHONY : cec17_test_func.c.i

cec17_test_func.s: cec17_test_func.c.s

.PHONY : cec17_test_func.s

# target to generate assembly for a file
cec17_test_func.c.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/cec17_test_func.dir/build.make CMakeFiles/cec17_test_func.dir/cec17_test_func.c.s
.PHONY : cec17_test_func.c.s

test.o: test.cc.o

.PHONY : test.o

# target to build an object file
test.cc.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/test.dir/build.make CMakeFiles/test.dir/test.cc.o
.PHONY : test.cc.o

test.i: test.cc.i

.PHONY : test.i

# target to preprocess a source file
test.cc.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/test.dir/build.make CMakeFiles/test.dir/test.cc.i
.PHONY : test.cc.i

test.s: test.cc.s

.PHONY : test.s

# target to generate assembly for a file
test.cc.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/test.dir/build.make CMakeFiles/test.dir/test.cc.s
.PHONY : test.cc.s

testrandom.o: testrandom.cc.o

.PHONY : testrandom.o

# target to build an object file
testrandom.cc.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/testrandom.dir/build.make CMakeFiles/testrandom.dir/testrandom.cc.o
.PHONY : testrandom.cc.o

testrandom.i: testrandom.cc.i

.PHONY : testrandom.i

# target to preprocess a source file
testrandom.cc.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/testrandom.dir/build.make CMakeFiles/testrandom.dir/testrandom.cc.i
.PHONY : testrandom.cc.i

testrandom.s: testrandom.cc.s

.PHONY : testrandom.s

# target to generate assembly for a file
testrandom.cc.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/testrandom.dir/build.make CMakeFiles/testrandom.dir/testrandom.cc.s
.PHONY : testrandom.cc.s

testsolis.o: testsolis.cc.o

.PHONY : testsolis.o

# target to build an object file
testsolis.cc.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/testsolis.dir/build.make CMakeFiles/testsolis.dir/testsolis.cc.o
.PHONY : testsolis.cc.o

testsolis.i: testsolis.cc.i

.PHONY : testsolis.i

# target to preprocess a source file
testsolis.cc.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/testsolis.dir/build.make CMakeFiles/testsolis.dir/testsolis.cc.i
.PHONY : testsolis.cc.i

testsolis.s: testsolis.cc.s

.PHONY : testsolis.s

# target to generate assembly for a file
testsolis.cc.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/testsolis.dir/build.make CMakeFiles/testsolis.dir/testsolis.cc.s
.PHONY : testsolis.cc.s

# Help Target
help:
	@echo "The following are some of the valid targets for this Makefile:"
	@echo "... all (the default if no target is provided)"
	@echo "... clean"
	@echo "... depend"
	@echo "... edit_cache"
	@echo "... rebuild_cache"
	@echo "... PeSOA"
	@echo "... cec17_test_func"
	@echo "... test"
	@echo "... testrandom"
	@echo "... testsolis"
	@echo "... PeSOA.o"
	@echo "... PeSOA.i"
	@echo "... PeSOA.s"
	@echo "... cec17.o"
	@echo "... cec17.i"
	@echo "... cec17.s"
	@echo "... cec17_test_func.o"
	@echo "... cec17_test_func.i"
	@echo "... cec17_test_func.s"
	@echo "... test.o"
	@echo "... test.i"
	@echo "... test.s"
	@echo "... testrandom.o"
	@echo "... testrandom.i"
	@echo "... testrandom.s"
	@echo "... testsolis.o"
	@echo "... testsolis.i"
	@echo "... testsolis.s"
.PHONY : help



#=============================================================================
# Special targets to cleanup operation of make.

# Special rule to run CMake to check the build system integrity.
# No rule that depends on this can have commands that come from listfiles
# because they might be regenerated.
cmake_check_build_system:
	$(CMAKE_COMMAND) -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 0
.PHONY : cmake_check_build_system

