To run the code of an earlier task (e.g. 2.1, 2.2), change the #define for TASK.
To run the code for the final task (2.4), keep #define TASK 4.
The task the code will operate on is 2.TASK.


To compile the program, use the following command:

gcc -fopenmp -o [FILENAME] -std=c99 main.c -lm

To run:
./[FILENAME]

To plot the results:
#define TASK 1: gnuplot plot_2.1
#define TASK 2,3,4: gnuplot plot_2.2

To plot the vertically averaged distribution (#define TASK 4 only)
gnuplot plot_2.4
