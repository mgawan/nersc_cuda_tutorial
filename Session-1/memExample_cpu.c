#include <stdio.h>
#include <assert.h>

void swap_cpu(int *a, int *b)
{
 int tmp = *a;
 *a = *b;
 *b = tmp;
}

int main()
{
 int h_a, h_b;
 h_a = 3;
 h_b = 9;

 swap_cpu(&h_a, &h_b);
 assert(h_a == 9);
 assert(h_b == 3);

 return 0;
}
