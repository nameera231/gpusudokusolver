#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "util.h"

// The width and height of a sudoku board
#define BOARD_DIM 9

// The width and heigh of a square group in a sudoku board
#define GROUP_DIM 3

// The number of boards to pass to the solver at one time
#define BATCH_SIZE 6000

// The number of threads per block, each thread is a single cell
#define THREADS_PER_BLOCK 81

/**
 * A board is an array of 81 cells. Each cell is encoded as a 16-bit integer.
 * Read about this encoding in the documentation for the digit_to_cell and
 * cell_to_digit functions' documentation.
 *
 * Boards are stored as a one-dimensional array. 
 */

// A structure that is a single sudoku board with 81 cells
typedef struct board {
  uint16_t cells[BOARD_DIM * BOARD_DIM];
} board_t;


void print_board(board_t* board);
__host__ __device__ uint16_t digit_to_cell(int digit);
__host__ __device__ int cell_to_digit(uint16_t cell);


// A kernel function that will run for each cell of the sudoku board
__global__ void kernel(board_t* boards) {

  // Variables to keep track of the current block (sudoku board)
  // and the current thread (cell)
  int board_id = blockIdx.x ;
  int thread_id = threadIdx.x;
  
  // An array for that holds the order for each region
  int regions[] =   {0,1,2,9,10,11,18,19,20,
                     3,4,5,12,13,14,21,22,23,
                     6,7,8,15,16,17,24,25,26,
                     27,28,29,36,37,38,45,46,47,
                     30,31,32,39,40,41,48,49,50,
                     33,34,35,42,43,44,51,52,53,
                     54,55,56,63,64,65,72,73,74,
                     57,58,59,66,67,68,75,76,77,
                     60,61,62,69,70,71,78,79,80};
    
  // Pointer to the current cell
  uint16_t* current_cell = &boards[board_id].cells[thread_id];
    
  // The current cell starts with all possible values
  // Possibilities are then removed
  

  int row_start = thread_id - thread_id % BOARD_DIM; // Finding the cell at the  start of a row
  int column_start = thread_id % BOARD_DIM; // Find the cell at the start of a column
  int changed_board = 1; // Variable to check if the board has been changed



  // While the board continues to change, do the following:
  while ( __syncthreads_count(changed_board) != 0)
    
    {
      // Record the value in the current cell to keep track of the state of the board before the rest of the while loop executes.
      int old_board  = cell_to_digit(*current_cell); 
      
      // Reduce the possibilities based on the column
      for(int i = 0; i < 9; i++)
        {
          // Don't compare the values of the current cell against itself
          if (thread_id != column_start + 9*i)
            {
              // Find out if the cell we are at has a final element in it 
              int propogate = cell_to_digit(boards[board_id].cells[column_start + 9*i]);
              // if it does - remove that possibility for current cell
              if (propogate != 0)
                {
                  *current_cell = *current_cell & ~digit_to_cell(propogate);
                }
            }
        }
    
      // Reduce the possibilities based on the row
      for(int i =0 ; i < 9; i++)
        {
          // Don't compare the values of the current cell against itself
          if (thread_id != row_start + i)
            {
              // Find out if the cell we are at has a final element in it 
              int row_tracker =  cell_to_digit(boards[board_id].cells[row_start + i]);
              //if it does - remove that possibility for current cell
              if (row_tracker!= 0)
                {
                  *current_cell = *current_cell & ~digit_to_cell(row_tracker);
                }
            } 
        }
    
      // Reduce the possibilities based on the region

      // Locate the region
      int current_region;
      for (int i = 0; i < 81; i++)
        {
          if (thread_id == regions[i])
            {
              current_region =(i -  i % 9) ;
              break;
            }
        }
          
      for (int k = 0; k < 9; k++)
        {
          // Don't compare the values of the current cell against itself
          if (thread_id != regions[current_region + k])
            {
              // Find out if the cell we are at has a final element in it
              int region_tracker =  cell_to_digit(boards[board_id].cells[regions[current_region+k]]);
              // if it does - remove that possibility for current cell
              if (region_tracker !=0)
                {
                  *current_cell = *current_cell & ~digit_to_cell(region_tracker);
                }
            }
        }
                                                        
      int new_board = cell_to_digit(*current_cell); // Record the current value in current cell

      // If the new cell value is same as old cell value the board was not changed by this cell
      if (new_board == old_board)
        {
          changed_board = 0;
        }
      // Otherwise, changes are still being made - continue the while loop
      else
        {
          changed_board = 1;
        }    
    }
}// Kernel end


/**
 * Take an array of boards and solve them all. The number of boards will be no
 * more than BATCH_SIZE, but may be less if the total number of input boards
 * is not evenly-divisible by BATCH_SIZE.
 *
 * 
 * \param boards      An array of boards that should be solved.
 * \param num_boards  The number of boards in the boards array
 */
void solve_boards(board_t* boards, size_t num_boards) {

  // Each board is a single block with 81 threads (one thread per cell)

  // copy boards array to gpu and use the gpu boards array to call the kernel 
  
  board_t* gpu_boards;

  // Allocate space for the boards array on the GPU
  if(cudaMalloc(&gpu_boards, sizeof(board_t) * num_boards) != cudaSuccess) {
    fprintf(stderr, "Failed to allocate X array on GPU\n");
    exit(2);
  }

  // Copy the cpu's boards array to the gpu with cudaMemcpy
  if(cudaMemcpy(gpu_boards, boards, sizeof(board_t) * num_boards, cudaMemcpyHostToDevice) != cudaSuccess) {
    fprintf(stderr, "Failed to copy boards to the gpu_boards\n");
  }
  
  // Run the kernel -- one block is a board so no. of blocks = number of boards,
  // Threads per block = 81 for the number of cells in a board
  kernel<<<num_boards, THREADS_PER_BLOCK>>>(gpu_boards);
     
  // Wait for the kernel to finish
  if(cudaDeviceSynchronize() != cudaSuccess) {
    fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
  }

  // Copy the boards array back from the gpu to the cpu
  if(cudaMemcpy(boards, gpu_boards, sizeof(board_t) * num_boards, cudaMemcpyDeviceToHost) != cudaSuccess) {
    fprintf(stderr, "Failed to copy Y from the GPU\n");
  }
}

/**
 * Take as input an integer value 0-9 (inclusive) and convert it to the encoded
 * cell form used for solving the sudoku. This encoding uses bits 1-9 to
 * indicate which values may appear in this cell.
 *
 * For example, if bit 3 is set to 1, then the cell may hold a three. Cells that
 * have multiple possible values will have multiple bits set.
 *
 * The input digit 0 is treated specially. This value indicates a blank cell,
 * where any value from one to nine is possible.
 *
 * \param digit   An integer value 0-9 inclusive
 * \returns       The encoded form of digit using bits to indicate which values
 *                may appear in this cell.
 */
__host__ __device__ uint16_t digit_to_cell(int digit) {
  if (digit == 0) {
    // A zero indicates a blank cell. Numbers 1-9 are possible, so set bits 1-9.
    return 0x3FE;
  } else {
    // Otherwise we have a fixed value. Set the corresponding bit in the board.
    return 1 << digit;
  }
}

/*
 * Convert an encoded cell back to its digit form. A cell with two or more
 * possible values will be encoded as a zero. Cells with one possible value
 * will be converted to that value.
 *
 * For example, if the provided cell has only bit three set, this function will
 * return the value 3.
 *
 * \param cell  An encoded cell that uses bits to indicate which values could
 *              appear at this point in the board.
 * \returns     The value that must appear in the cell if there is only one
 *              possibility, or zero otherwise.
 */
__host__ __device__ int cell_to_digit(uint16_t cell) {
  // Get the index of the least-significant bit in this cell's value
#if defined(__CUDA_ARCH__)
  int msb = __clz(cell);
  int lsb = sizeof(unsigned int) * 8 - msb - 1;
#else
  int lsb = __builtin_ctz(cell);
#endif

  // Is there only one possible value for this cell? If so, return it.
  // Otherwise return zero.
  if (cell == 1 << lsb)
    return lsb;
  else
    return 0;
}

/**
 * Read in a sudoku board from a string. Boards are represented as an array of
 * 81 16-bit integers. Each integer corresponds to a cell in the board. Bits
 * 1-9 of the integer indicate whether the values 1, 2, ..., 8, or 9 could
 * appear in the given cell. A zero in the input indicates a blank cell, where
 * any value could appear.
 *
 * \param output  The location where the board will be written
 * \param str     The input string that encodes the board
 * \returns       true if parsing succeeds, false otherwise
 */
bool read_board(board_t* output, const char* str) {
  for (int index = 0; index < BOARD_DIM * BOARD_DIM; index++) {
    if (str[index] < '0' || str[index] > '9') return false;

    // Convert the character value to an equivalent integer
    int value = str[index] - '0';

    // Set the value in the board
    output->cells[index] = digit_to_cell(value);
  }

  return true;
}

/**
 * Print a sudoku board. Any cell with a single possible value is printed. All
 * cells with two or more possible values are printed as blanks.
 *
 * \param board   The sudoku board to print
 */
void print_board(board_t* board) {
  for (int row = 0; row < BOARD_DIM; row++) {
    // Print horizontal dividers
    if (row != 0 && row % GROUP_DIM == 0) {
      for (int col = 0; col < BOARD_DIM * 2 + BOARD_DIM / GROUP_DIM; col++) {
        printf("-");
      }
      printf("\n");
    }

    for (int col = 0; col < BOARD_DIM; col++) {
      // Print vertical dividers
      if (col != 0 && col % GROUP_DIM == 0) printf("| ");

      // Compute the index of this cell in the board array
      int index = col + row * BOARD_DIM;

      // Get the index of the least-significant bit in this cell's value
      int digit = cell_to_digit(board->cells[index]);

      // Print the digit if it's not a zero. Otherwise print a blank.
      if (digit != 0)
        printf("%d ", digit);
      else
        printf("  ");
    }
    printf("\n");
  }
  printf("\n");
}

/**
 * Check through a batch of boards to see how many were solved correctly.
 *
 * \param boards        An array of (hopefully) solved boards
 * \param solutions     An array of solution boards
 * \param num_boards    The number of boards and solutions
 * \param solved_count  Output: A pointer to the count of solved boards.
 * \param error:count   Output: A pointer to the count of incorrect boards.
 */
void check_solutions(board_t* boards,
                     board_t* solutions,
                     size_t num_boards,
                     size_t* solved_count,
                     size_t* error_count) {
  // Loop over all the boards in this batch
  for (int i = 0; i < num_boards; i++) {
    // Does the board match the solution?
    if (memcmp(&boards[i], &solutions[i], sizeof(board_t)) == 0) {
      // Yes. Record a solved board
      (*solved_count)++;
    } else {
      // No. Make sure the board doesn't have any constraints that rule out
      // values that are supposed to appear in the solution.
      bool valid = true;
      for (int j = 0; j < BOARD_DIM * BOARD_DIM; j++) {
        if ((boards[i].cells[j] & solutions[i].cells[j]) == 0) {
          valid = false;
        }
      }
      
      //print_board(&solutions[i]);
      // If the board contains an incorrect constraint, record an error
      if (!valid){
       
        (*error_count)++;
      }
    }
  }
}

/**
 * Entry point for the program
 */
int main(int argc, char** argv) {
  // Check arguments
  if (argc != 2) {
    fprintf(stderr, "Usage: %s <input file name>\n", argv[0]);
    exit(1);
  }

  // Try to open the input file
  FILE* input = fopen(argv[1], "r");
  if (input == NULL) {
    fprintf(stderr, "Failed to open input file %s.\n", argv[1]);
    perror(NULL);
    exit(2);
  }

  // Keep track of total boards, boards solved, and incorrect outputs
  size_t board_count = 0;
  size_t solved_count = 0;
  size_t error_count = 0;

  // Keep track of time spent solving
  size_t solving_time = 0;

  // Reserve space for a batch of boards and solutions
  board_t boards[BATCH_SIZE];
  board_t solutions[BATCH_SIZE];

  // Keep track of how many boards we've read in this batch
  size_t batch_count = 0;

  // Read the input file line-by-line
  char* line = NULL;
  size_t line_capacity = 0;
  while (getline(&line, &line_capacity, input) > 0) {
    // Read in the starting board
    if (!read_board(&boards[batch_count], line)) {
      fprintf(stderr, "Skipping invalid board...\n");
      continue;
    }

    // Read in the solution board
    if (!read_board(&solutions[batch_count], line + BOARD_DIM * BOARD_DIM + 1)) {
      fprintf(stderr, "Skipping invalid board...\n");
      continue;
    }

    // Move to the next index in the batch
    batch_count++;

    // Also increment the total count of boards
    board_count++;

    // If we finished a batch, run the solver
    if (batch_count == BATCH_SIZE) {
      size_t start_time = time_ms();
      solve_boards(boards, batch_count);
      solving_time += time_ms() - start_time;

      check_solutions(boards, solutions, batch_count, &solved_count, &error_count);

      // Reset the batch count
      batch_count = 0;
    }
  }

  // Check if there's an incomplete batch to solve
  if (batch_count > 0) {
    size_t start_time = time_ms();
    solve_boards(boards, batch_count);
    solving_time += time_ms() - start_time;

    check_solutions(boards, solutions, batch_count, &solved_count, &error_count);
  }

  // Print stats
  double seconds = (double)solving_time / 1000;
  double solving_rate = (double)solved_count / seconds;

  // Don't print nan when solver is not implemented
  if (seconds < 0.01) solving_rate = 0;

  printf("Boards: %lu\n", board_count);
  printf("Boards Solved: %lu\n", solved_count);
  printf("Errors: %lu\n", error_count);
  printf("Total Solving Time: %lums\n", solving_time);
  printf("Solving Rate: %.2f sudoku/second\n", solving_rate);

  return 0;
}
