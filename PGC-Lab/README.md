# Matrix Multiplication — Code and Explanations (Parts A–D)

All four programs compute C = A × B for 4000×4000 matrices where every element of A and B is 1.0, so every element of C should equal 4000.00. This section shows the actual code used and explains what each part is doing.

---

## Part A — Sequential (`matrix_sequential.c`)
Sequential (Part A): One CPU core does the entire job, one operation after another. No parallelism at all. This just measures how long the raw computation takes and sets the baseline everything else is judged against.
```c
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 4000

int main()
{
    int i, j, k;
    double *A, *B, *C;
    clock_t start, end;

    A = (double *)malloc(N * N * sizeof(double));
    B = (double *)malloc(N * N * sizeof(double));
    C = (double *)malloc(N * N * sizeof(double));

    if (A == NULL || B == NULL || C == NULL)
    {
        printf("Memory allocation failed\n");
        return 1;
    }

    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++)
        {
            A[i * N + j] = 1.0;
            B[i * N + j] = 1.0;
            C[i * N + j] = 0.0;
        }

    start = clock();

    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++)
            for (k = 0; k < N; k++)
                C[i * N + j] += A[i * N + k] * B[k * N + j];

    end = clock();

    printf("Execution Time = %f seconds\n", (double)(end - start) / CLOCKS_PER_SEC);
    printf("Verification C[0][0] = %.2f\n", C[0]);

    free(A); free(B); free(C);
    return 0;
}
```

**What it does:**
- Matrices are stored as flat 1D arrays (`A[i*N+j]` instead of `A[i][j]`) so a single `malloc` can allocate the whole 4000×4000 block — a 2D VLA of that size would blow the stack.
- The triple loop is the textbook definition of matrix multiplication: `C[i][j] = Σ A[i][k] * B[k][j]` over `k`.
- `clock()` measures CPU time on a single core. No parallelism at all — this is the number every other part is compared against.
- Time complexity is O(N³) = 4000³ ≈ 64 billion multiply-add operations, all done serially.

---

## Part B — OpenMP (`matrix_openmp.c`)
Still one machine, one process, but now multiple threads inside that process each grab a different chunk of rows and compute them at the same time. This works because threads share the same memory — thread 2 can read matrix A and B directly without anyone copying data for it. The only real overhead is threads occasionally syncing up. This is why OpenMP tends to give close to linear speedup (8 threads ≈ ~8x faster).
```c
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N 4000

int main()
{
    int i, j, k;
    double *A, *B, *C;
    double start, end;

    A = (double *)malloc(N * N * sizeof(double));
    B = (double *)malloc(N * N * sizeof(double));
    C = (double *)malloc(N * N * sizeof(double));

    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++)
        {
            A[i * N + j] = 1.0;
            B[i * N + j] = 1.0;
            C[i * N + j] = 0.0;
        }

    start = omp_get_wtime();

    #pragma omp parallel for private(j, k)
    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++)
            for (k = 0; k < N; k++)
                C[i * N + j] += A[i * N + k] * B[k * N + j];

    end = omp_get_wtime();

    printf("Number of Threads Used = %d\n", omp_get_max_threads());
    printf("Execution Time = %f seconds\n", end - start);
    printf("Verification C[0][0] = %.2f\n", C[0]);

    free(A); free(B); free(C);
    return 0;
}
```

**What it does:**
- Identical algorithm to Part A, but `#pragma omp parallel for` splits the **outer `i` loop** across threads — each thread gets a contiguous block of rows to compute.
- `private(j, k)` gives each thread its own copies of the loop counters `j` and `k`, so threads don't clobber each other's state. `A`, `B`, `C` stay shared since threads only ever write to their own rows of `C`.
- Thread count is controlled by the `OMP_NUM_THREADS` environment variable (set to 8 in the reference run), not by the code itself — this is what lets you re-run at 4 vs 8 threads without recompiling.
- `omp_get_wtime()` measures wall-clock time (not CPU time), which is the right choice once multiple threads are running concurrently.
- All threads share one address space, so there's no data copying — just synchronization at the end of the parallel region.

---

## Part C — MPI (`matrix_mpi.c`)
Instead of threads inside one process, you now have 4 completely separate processes, potentially on 4 different machines. Separate processes means separate memory — process 2 has no way to "just read" what process 0 has in RAM. So everything has to be explicitly sent over the network: rank 0 splits up rows of A and mails a chunk to each rank, broadcasts a full copy of B to everyone (since every rank needs all of it), each rank computes its own piece independently, then all the results get mailed back to rank 0 to reassemble. The compute itself is just as parallel as OpenMP, but all that sending/receiving is pure overhead OpenMP never pays — which is why MPI often ends up slower than OpenMP even with the same worker count, unless the problem is big enough that computation dominates communication.

```c
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <unistd.h>

#define N 4000

int main(int argc, char *argv[])
{
    int rank, size, i, j, k, rows_per_process;
    char hostname[256];
    double *A = NULL, *B = NULL, *C = NULL;
    double *local_A, *local_C;
    double start, end;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    gethostname(hostname, sizeof(hostname));

    rows_per_process = N / size;

    local_A = (double *)malloc(rows_per_process * N * sizeof(double));
    local_C = (double *)malloc(rows_per_process * N * sizeof(double));
    B = (double *)malloc(N * N * sizeof(double));

    if (rank == 0)
    {
        A = (double *)malloc(N * N * sizeof(double));
        C = (double *)malloc(N * N * sizeof(double));
        for (i = 0; i < N; i++)
            for (j = 0; j < N; j++)
            {
                A[i * N + j] = 1.0;
                B[i * N + j] = 1.0;
                C[i * N + j] = 0.0;
            }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();

    MPI_Scatter(A, rows_per_process * N, MPI_DOUBLE,
                local_A, rows_per_process * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Bcast(B, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    printf("Rank %d on %s computing %d rows\n", rank, hostname, rows_per_process);

    for (i = 0; i < rows_per_process; i++)
        for (j = 0; j < N; j++)
        {
            local_C[i * N + j] = 0.0;
            for (k = 0; k < N; k++)
                local_C[i * N + j] += local_A[i * N + k] * B[k * N + j];
        }

    MPI_Gather(local_C, rows_per_process * N, MPI_DOUBLE,
               C, rows_per_process * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    end = MPI_Wtime();

    if (rank == 0)
    {
        printf("Execution Time = %f seconds\n", end - start);
        printf("Verification C[0][0] = %.2f\n", C[0]);
        free(A); free(C);
    }

    free(B); free(local_A); free(local_C);
    MPI_Finalize();
    return 0;
}
```

**What it does:**
- Unlike OpenMP, each of the 4 ranks is a **separate process** (often on a separate VM), with its own memory space — nothing is automatically shared, so every piece of data each rank needs has to be explicitly sent over MPI.
- Only rank 0 allocates and initializes the full `A` and `C` — the other ranks never see the full matrices, only their own slice.
- `MPI_Scatter` splits `A`'s 4000 rows into 4 chunks of 1000 rows and sends one chunk to each rank (rank 0 keeps a chunk too).
- `MPI_Bcast` sends the *entire* matrix `B` to every rank, because every rank needs all of `B` to compute its output rows (`C[i][j]` depends on the whole `k`-th column of `B`).
- Each rank computes its own `local_C` independently, with no communication during the compute loop itself — this is the "embarrassingly parallel" part.
- `MPI_Gather` collects each rank's `local_C` back into the full `C` on rank 0, in rank order.
- `MPI_Barrier` + `MPI_Wtime()` around the whole thing ensures every rank starts timing at the same point and rank 0 only stops the clock once every rank has finished and reported back — so the recorded time reflects the slowest rank, not just rank 0.
- This is why MPI is slower than OpenMP here despite using the same number of workers (4): scattering `A`, broadcasting a 128 MB copy of `B` to every rank, and gathering `C` all cost real network/memory-copy time that OpenMP's shared-memory threads never pay.

---

