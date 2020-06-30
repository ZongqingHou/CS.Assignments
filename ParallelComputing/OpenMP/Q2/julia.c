#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <omp.h>
#include "qdbmp.h"
#include "julia.h"

// Takes an integer colour value and splits it into its RGB component parts.
// val (a 32-bit unsigned integer type) is expected to contain a 24-bit unsigned integer.
void toRGB(unsigned int val,
           unsigned char *r, unsigned char *g, unsigned char *b)
{
    // intentionally mixed up the order here to make the colours a little nicer...
    *r = (val & 0xff);
    val >>= 8;
    *b = (val & 0xff);
    val >>= 8;
    *g = (val & 0xff);
}

// Returns the sum of the elements in the given array.
unsigned int sum_array(unsigned int *array, int len)
{
    unsigned int total = 0;

    #pragma omp parallel for reduction(+:total)
    for (int i = 0; i < len; i++)
    {
        total += array[i];
    }

    return total;
}

// Perform "histogram colour equalization" on the data array, using the
// information in the histogram array.
// This just ensures that the colours get nicely distributed to different
// values in the data array (i.e. makes sure that if the data array only contains values
// in a narrow range (between 100 and 200), the colours won't all be the same.
void hist_eq(unsigned int *data, unsigned int *hist)
{
    unsigned int total = sum_array(hist, MAX_ITER);
    unsigned int val;

    // Create a cache to speed up the loops below,
    // since they'll require the use of the same values many times
    float cache[MAX_ITER];
    float hue = 0.0;

    int x;

    #pragma omp parallel for ordered reduction(+:hue)
    for (unsigned int i = 0; i < MAX_ITER; i++)
    {
        #pragma omp ordered
            cache[i] = hue;
        hue += (float) hist[i] / total;
    }

    // Go through each pixel in the output image and tweak its colour value
    // (such that when we're done, the colour values in the data array have a uniform distribution)
    
    #pragma omp parallel for private(x, hue, val)
    for (int y = 0; y < HEIGHT; y++)
    {
        for (x = 0; x < WIDTH; x++)
        {
            val = data[y * WIDTH + x];

            // if the number's cached, use it
            if (val < MAX_ITER)
            {
                hue = cache[val];
            }
            //otherwise, calculate it
            else
            {
                hue = cache[MAX_ITER - 1];
                for (unsigned int i = MAX_ITER; i < val; i++)
                {
                    hue += (float) hist[i] / total;
                }
            }

            // expand the value's range from [0, 1] to [0, MAX_COLOUR]
            data[y * WIDTH + x] = (unsigned int) (hue * MAX_COLOUR);
        }
    }
}

// Writes the given data to a bitmap (.bmp) file with the given name.
// To do this, it interprets each value in the data array as an RGB colour
// (by calling toRGB()).
void write_bmp(unsigned int *data, char *fname)
{
    BMP *bmp = BMP_Create((UINT) WIDTH, (UINT) HEIGHT, (USHORT) DEPTH);
    unsigned char r, g, b;

    for (int y = 0; y < HEIGHT; y++)
    {
        for (int x = 0; x < WIDTH; x++)
        {
            toRGB(data[y * WIDTH + x], &r, &g, &b);
            BMP_SetPixelRGB(bmp, (UINT) x, (UINT) y,
                            (UCHAR) r, (UCHAR) g, (UCHAR) b);
        }
    }

    BMP_WriteFile(bmp, FNAME);
    BMP_Free(bmp);
}

// Generates terms of the Julia fractal sequence (starting with the given complex number)
// until either the imaginary part exceeds LIMIT or we hit MAX_ITER iterations.
unsigned int julia_iters(float complex z)
{
    unsigned int iter = 0;
    while (fabsf(cimag(z)) < LIMIT && iter < MAX_ITER)
    {
        z = C * csin(z);
        iter++;
    }

    //this value will be used to colour a pixel on the screen
    return iter;
}

// Computes the colour data for one row of pixels in the output image.
// Results are stored in the data array. Also populates the histogram array
// with the counts of the distinct values in this row.
void compute_row(int row, unsigned int *data, unsigned int *hist)
{
    float complex z;
    float series_row;
    float series_col;
    unsigned int iters;

    for (int col = 0; col < WIDTH; col++)
    {
        series_row = row - HEIGHT / 2;
        series_col = col - WIDTH / 2;
        z = series_col / RES_FACTOR + (I / RES_FACTOR) * series_row;
        z *= SCALE;
        iters = julia_iters(z);
        data[row * WIDTH + col] = iters;
        #pragma omp critical
        hist[iters]++;
    }
}

int main(int argc, char *argv[])
{
    printf("Beginning julia set computation...\n");

    // The data array below stores the pixel data as a 1D array.
    // Each element represents colour of 1 pixel in the output image.
    // The hist (histogram) array the frequencies of the values in the data array.
    // E.g. If hist[2] == 30, that means the number appears 30 times in the data array.
    int row;
    unsigned int data[HEIGHT * WIDTH];
    unsigned int hist[MAX_ITER] = {0};

    #pragma omp parallel for private(row)
    // Build the output image one row at a time.
    for (row = 0; row < HEIGHT; row++)
    {
        compute_row(row, data, hist);
    }

    // Perform "histogram equalization" (assign colours to each of distinct values in the data array),
    // and write out the results to a .bmp image file.
    hist_eq(data, hist);
    write_bmp(data, FNAME);

    printf("Done.\n");

    return EXIT_SUCCESS;
}