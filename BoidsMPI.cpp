#include <mpi.h>
#include "BoidsFunctions.cpp"

// These must be changed before compilation.
// Output decides whether the program will write data to a text file for animation.
#define N_FRAMES 3000
#define OUTPUT false
#define FILENAME "BoidsOutput.txt"

// Only n_boids argument
int main(int argc, char* argv[])
{
// Adding a second argument will make the program output in a way that is easier to collect data from.
    
    int argNBoids = atoi(argv[1]);

    std::vector<std::array<std::vector<position>, 2>> progressedStates;

    int rank, size, nBoids;

    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int remainder = argNBoids % size;
    if (remainder == 0)
    {
        nBoids = argNBoids;
    }
    else
    {
        nBoids = argNBoids + (size - remainder);
        if (rank == 0 && argc == 2)
            printf("Number of boids must be a multiple of the number of processes. \nThe number of boids has been rounded up from %d to %d \n\n", argNBoids, nBoids);
    }

    int boidsPerProcess = nBoids / size;

    // Make MPI datatype for __m128d
    MPI_Datatype AVXDouble2Vector;
    MPI_Type_contiguous(sizeof(position), MPI_BYTE, &AVXDouble2Vector);
    MPI_Type_commit(&AVXDouble2Vector);

    // Set the boids each process deals with
    std::vector<int> processIndices(boidsPerProcess);
    for (int i = 0; i < boidsPerProcess; ++i)
    {
        processIndices[i] = (rank * boidsPerProcess) + i;
    }

    // Initialise position and direction buffers
    std::vector<position> currentPositions(nBoids);
    std::vector<direction> currentDirections(nBoids);
    std::vector<position> nextPositions(boidsPerProcess);
    std::vector<direction> nextDirections(boidsPerProcess);

    // Initialise timer
    Timer timer;

    // Set initial state
    if (rank == 0)
    {
        progressedStates.push_back(initialState(nBoids));
        currentPositions = progressedStates[0][0];
        currentDirections = progressedStates[0][1];

        timer.start();
    }

    for (int i = 0; i < N_FRAMES; i++)
    {
        MPI_Bcast(currentPositions.data(), nBoids, AVXDouble2Vector, 0, MPI_COMM_WORLD);
        MPI_Bcast(currentDirections.data(), nBoids, AVXDouble2Vector, 0, MPI_COMM_WORLD);

        for (int j = 0; j < boidsPerProcess; ++j)
        {
            auto nextBoid = iterateBoid(currentPositions, currentDirections, processIndices[j], nBoids);
            nextPositions[j] = nextBoid[0];
            nextDirections[j] = nextBoid[1];
        }

        MPI_Allgather(nextPositions.data(), boidsPerProcess, AVXDouble2Vector, currentPositions.data(), boidsPerProcess, AVXDouble2Vector, MPI_COMM_WORLD);
        MPI_Allgather(nextDirections.data(), boidsPerProcess, AVXDouble2Vector, currentDirections.data(), boidsPerProcess, AVXDouble2Vector, MPI_COMM_WORLD);

        if (rank == 0 && OUTPUT)
        {
            progressedStates.push_back({currentPositions, currentDirections});
        }
    }

    if (rank == 0)
    {
        timer.stop();
        if (argc == 2) std::cout << "Time taken was " << timer.elapsedSeconds() << "seconds \n";
        else std::cout << size << " " << timer.elapsedSeconds() << "\n";
    }

    if (rank == 0 && OUTPUT)
    {
        writeToFile(progressedStates, N_FRAMES, nBoids);
    }

    MPI_Finalize();
}
