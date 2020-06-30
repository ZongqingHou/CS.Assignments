#ifndef _UTILS_H
#define _UTILS_H

/* Prototypes */

int parse_args(int argc, char *argv[]);
void init_vec(float *vec, int len);
void print_vec(const char *label, float *vec, int len);
void check_error(cudaError_t status, const char *msg);
int get_max_block_threads();

#endif
