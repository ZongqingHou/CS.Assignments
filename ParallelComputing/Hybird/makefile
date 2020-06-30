# Comp 4510 - Sample Hyrbid MPI-CUDA Makefile

###############################################
# Change these files names to match your own: #
###############################################

# name of the executable to be generated
PROG = hybrid
INCLUDE_DIR = include
SRC_DIR = src
# space-delimited list of CUDA header files
# (headers for source files in which you call CUDA API functions)
CUDA_HDRS = kernel.h sum_vec.h
# space-delimited list of CUDA source files
# (files in which you call CUDA API functions)
CUDA_SRCS = kernel.cu sum_vec.cu
# space-delimited list of MPI header files
MPI_HDRS =
# space-delimited list of MPI source files
MPI_SRCS = main.c

CU_HDRS = $(patsubst %.h, $(INCLUDE_DIR)/%.h, $(CUDA_HDRS))
CU_SRCS = $(patsubst %.cu, $(SRC_DIR)/%.cu, $(CUDA_SRCS))
C_HDRS = $(patsubst %.h, $(INCLUDE_DIR)/%.h, $(MPI_HDRS))
C_SRCS = $(patsubst %.c, $(SRC_DIR)/%.c, $(MPI_SRCS))

#######################
# Don't change these: #
#######################

# directory to store object files
CUDA_OBJDIR = ./object/cuda
CUDA_OBJS = $(patsubst %.cu, $(CUDA_OBJDIR)/%.o, $(CU_SRCS))
CUDA_CC = nvcc
CUDA_CFLAGS = -arch=sm_13 -Wno-deprecated-gpu-targets -L. -I$(INCLUDE_DIR)
CUDA_LIBS = -lm

MPI_OBJDIR = ./object/mpi
MPI_OBJS = $(patsubst %.c, $(MPI_OBJDIR)/%.o, $(C_SRCS))
MPI_CC = mpic++
MPI_CFLAGS = -Wall -L/opt/cuda-6.5/lib64 -I$(INCLUDE_DIR)
MPI_LIBS = -lm -lcudart


####################
# Compiling rules: #
####################
# WARNING: *must* have a tab before each definition

# invoked when "make" is run
all : $(CUDA_OBJDIR) $(MPI_OBJDIR) $(PROG)

# links object files into executable
$(PROG) : $(CUDA_OBJS) $(MPI_OBJS)
	$(MPI_CC) $(MPI_CFLAGS) $(subst /src/,/,$^) -o $(PROG) $(MPI_LIBS)

# compiles source files into object files
$(CUDA_OBJDIR)/%.o : %.cu $(CU_HDRS)
	$(CUDA_CC) -c $(CUDA_CFLAGS) $< -o $(CUDA_OBJDIR)/$(notdir $@) $(CUDA_LIBS)

$(MPI_OBJDIR)/%.o : %.c $(C_HDRS)
	$(MPI_CC) -c $(MPI_CFLAGS) $< -o $(MPI_OBJDIR)/$(notdir $@) $(MPI_LIBS)

# creates directory to store object files
$(CUDA_OBJDIR) :
	mkdir -p $@/

$(MPI_OBJDIR) :
	mkdir -p $@/

# cleans up object files and executable
# type "make clean" to use
# Note: you can add your own commands to remove other things (e.g. output files)
clean:
	rm -rf object/
	rm -f $(PROG)
