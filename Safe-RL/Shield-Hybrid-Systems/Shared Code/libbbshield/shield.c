/** To this day I have no idea how to write a makefile.
 Run as
    test: 
        echo "\n\n" && gcc -Wall shield.c -o shield.o
        ./shield.o

    library:
        gcc -c -fPIC shield.c -o shield.o
        gcc -shared -o libbbshield.so shield.o

*/

#include<stdio.h>
#include<stdbool.h>
#include "shield_dump.c"


const int x_count = (x_max - x_min)/G;
const int y_count = (y_max - y_min)/G;


char get_index(const char grid[], int iv, int ip)
{
    return grid[iv + ip*x_count];
}

int box_v(double  v)
{
    return (int) ((v - x_min)/G);
}

int box_p(double  p)
{
    return (int) ((p - y_min)/G);
}

char get_value(double v, double p)
{
    if (v < x_min || v >= x_max || p < y_min || p >= y_max)
    {
        return '?';
    }
    int iv = box_v(v);
    int ip = box_p(p);
    return get_index(grid, iv, ip);
}

// True if the shield requires going fast, false otherwise.
bool must_hit(double v, double p)
{
    char color = get_value(v, p);
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
    if (x_count < 256)
    {
        int iv, it;
        for (it=y_count; it>=0; it--)
        {
            for (iv=0; iv<x_count; iv++)
            {
                printf("%c", get_index(grid, iv, it));
            }
            printf("\n");
        }
    }
    printf("x_count: %i\n", x_count);
    printf("y_count: %i\n", y_count);
    printf("box(0.1, 0.9): %i, %i\n", box_v(0.1), box_p(0.9));
    printf("box(0.0, 10.0): %i, %i\n", box_v(0.0), box_p(10.0));
    printf("box(12.0, 0.0): %i, %i\n", box_v(12.0), box_p(0.0));
    printf("box(-99.0, 0.0): %i, %i\n", box_v(-99.0), box_p(0.0));
    printf("get_value(0.1, 0.9): %c\n", get_value(0.1, 0.9));
    printf("get_value(0.0, 10.0): %c\n", get_value(0.0, 10.0));
    printf("get_value(12.0, 0.0): %c\n", get_value(12.0, 0.0));
    printf("get_value(-99.0, 0.0): %c\n", get_value(-99.0, 0.0));
    printf("Must hit (0.1, 0.9): %i\n", must_hit(0.1, 0.9));
    printf("Must hit (0.0, 10.0): %i\n", must_hit(0.0, 10.0));
    printf("Must hit (12.0, 0.0): %i\n", must_hit(12.0, 0.0));
    return 0;
}