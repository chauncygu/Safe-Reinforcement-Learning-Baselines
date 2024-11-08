/* To this day I have no idea how to write a makefile.
 Run as
    test: 
        echo "\n\n" && gcc -Wall preshield.c -o preshield.o && ./preshield.o

    library:
        gcc -c -fPIC preshield.c -o preshield.o && gcc -shared -o libccpreshield.so preshield.o

*/

#include<stdio.h>
#include<stdbool.h>
#include "shield_dump.c"

const int ANY_ACTION  = 1 + 2 + 4;
const int OUT_OF_BOUNDS = -1;

int get_index(int indices[])
{
    int index = 0;
    int multiplier = 1;
    int dim;
    for (dim = 0; dim < dimensions; dim++)
    {
        index += multiplier*indices[dim];
        multiplier *= size[dim];
    }
    return grid[index] - char_offset;
}

int box(double  value, int dim)
{
    return (int) ((value - lower_bounds[dim])/G);
}

int get_value_from_vector(double s[])
{
    int indices[dimensions];
    int dim;
    for (dim = 0; dim < dimensions; dim++)
    {
        if (s[dim] < lower_bounds[dim] || s[dim] >= upper_bounds[dim])
        {
            return OUT_OF_BOUNDS;
        }
        indices[dim] = box(s[dim], dim);
    }

    return get_index(indices);
}

int get_value(double v_ego, double v_front, double distance)
{
    return get_value_from_vector((double[]){v_ego, v_front, distance});
}

/*  TODO: clean up dirty glasses
    _____
   /.---.\ 
   |`````|
   \     /
    `-.-'           ____
      |    /\     .'   /\
    __|__  |K----;    |  |
   `-----` \/     '.___*/

int main()
{
    printf("These test cases are written for the grid saved in shield_dump(sample).c\n");
    printf("Note that failed cases are not highlighted.\n");
    printf("G: %f\n", G);
    printf("get_index({1, 1, 1}): %i\t\t(expected: 0)\n", get_index((int[]){1, 1, 1}));
    printf("get_index({11, 21, 51}): %i\t\t(expected: 7)\n", get_index((int[]){11, 21, 51}));
    printf("get_index({14, 21, 51}): %i\t\t(expected: 7)\n", get_index((int[]){14, 21, 51}));
    printf("box({3, 10, 20}): %i, %i, %i\t\t(expected: 26, 36, 42)\n", box(3., 0), box(10., 1), box(20., 2));
    printf("box({0, 10, 20}): %i, %i, %i\t\t(expected: 20, 36, 42)\n", box(0., 0), box(10., 1), box(20., 2));
    printf("box({-3, 5, 50}): %i, %i, %i\t\t(expected: 14, 26, 102)\n", box(-3., 0), box(5., 1), box(50., 2));
    printf("get_value({3, 10, 20}): %i\t\t(expected: 7)\n", get_value_from_vector((double[]){3, 10, 20}));
    printf("get_value({-3, 0, 8}): %i\t\t(expected: 3)\n", get_value_from_vector((double[]){-3, 0, 8}));
    printf("get_value({-3, 5, 8}): %i\t\t(expected: 7)\n", get_value_from_vector((double[]){-3, 5, 8}));
    printf("get_value({-4, -8, 17.7}): %i\t\t(expected: 3)\n", get_value_from_vector((double[]){-4, -8, 17.7}));
    printf("get_value({5, -8, 10}): %i\t\t(expected: 0)\n", get_value_from_vector((double[]){5, -8, 10}));
    printf("get_value({-8, -9, 6}): %i\t\t(expected: 3)\n", get_value_from_vector((double[]){-8, -9, 6}));
    return 0;
}