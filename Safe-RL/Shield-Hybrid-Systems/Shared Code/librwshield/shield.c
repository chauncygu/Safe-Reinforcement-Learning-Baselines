// Run as
//    test: 
//    echo "\n\n" && gcc -Wall shield.c -o shield.o && ./shield.o
//    library:
//    gcc -c -fPIC shield.c -o shield.o && gcc -shared -o librwshield.so shield.o

#include<stdio.h>
#include<stdbool.h>
#include "shield_dump.c"


const int x_count = (x_max - x_min)/G;
const int y_count = (y_max - y_min)/G;


char get_index(const char grid[], int ix, int iy)
{
    return grid[ix + iy*x_count];
}

int box_x(double  x)
{
    return (int) ((x - x_min)/G);
}

int box_y(double  y)
{
    return (int) ((y - y_min)/G);
}

char get_value(double x, double y)
{
    if (x < x_min || x >= x_max || y < y_min || y >= y_max)
    {
        return '?';
    }
    int ix = box_x(x);
    int iy = box_y(y);
    return get_index(grid, ix, iy);
}

// True if the shield requires going fast, false otherwise.
bool must_go_fast(double x, double t)
{
    char color = get_value(x, t);
    if (color == 'b' || color == 'r')
    {
        return true;
    }
    else
    {
        return false;
    }
}

int main()
{
    // Print imported shield
    int ix, it;
    for (it=y_count; it>=0; it--)
    {
        for (ix=0; ix<x_count; ix++)
        {
            printf("%c", get_index(grid, ix, it));
        }
        printf("\n");
    }
    printf("x_count: %i,   y_count: %i,   total: %i\n", x_count, y_count, x_count*y_count);
    printf("box(0.1, 0.9): %i, %i\n", box_x(0.1), box_y(0.9));
    printf("get_value(0.1, 0.9): %c\n", get_value(0.1, 0.9));
    printf("Must go fast (0.1, 0.9): %i\n", must_go_fast(0.1, 0.9));
    printf("Must go fast (1.0, 0.0): %i\n", must_go_fast(1.0, 0.0));
    return 0;
}