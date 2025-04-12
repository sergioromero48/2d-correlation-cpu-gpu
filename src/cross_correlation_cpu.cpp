#include <iostream>
#include <cstdlib>      // for rand()
#include <ctime>        // for time()
#include <iomanip>      // for output formatting

using namespace std;

//matrix sizes
const int INPUT_SIZE = 256;
const int KERNEL_SIZE = 8;
const int OUTPUT_SIZE = INPUT_SIZE - KERNEL_SIZE + 1;

int main() {
    srand(static_cast<unsigned int>(time(0)));  // Seed random number generator

    //declare matrices
    float input[INPUT_SIZE][INPUT_SIZE];
    float kernel[KERNEL_SIZE][KERNEL_SIZE];
    float output[OUTPUT_SIZE][OUTPUT_SIZE] = {0};  // Zero-initialize

    //random input & kernel values in range (-1, 1)
    for (int i = 0; i < INPUT_SIZE; ++i)
        for (int j = 0; j < INPUT_SIZE; ++j)
            input[i][j] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;

    for (int i = 0; i < KERNEL_SIZE; ++i)
        for (int j = 0; j < KERNEL_SIZE; ++j)
            kernel[i][j] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;

    //2D cross-correlation (no padding)
    for (int i = 0; i < OUTPUT_SIZE; ++i) {
        for (int j = 0; j < OUTPUT_SIZE; ++j) {
            float sum = 0.0f;
            for (int m = 0; m < KERNEL_SIZE; ++m) {
                for (int n = 0; n < KERNEL_SIZE; ++n) {
                    sum += input[i + m][j + n] * kernel[m][n];
                }
            }
            output[i][j] = sum;
        }
    }

    //5x5 sample from the output matrix
    cout << fixed << setprecision(4);
    cout << "\nSample output (5x5):\n";
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 5; ++j) {
            cout << setw(9) << output[i][j] << " ";
        }
        cout << endl;
    }

    return 0;
}
