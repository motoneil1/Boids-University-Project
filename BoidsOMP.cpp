#include <omp.h>
#include "BoidsFunctions.cpp"

// These must be changed before compilation.
// Output decides whether the program will write data to a text file for animation.
#define N_FRAMES 3000
#define OUTPUT false


std::array<std::vector<position>, 2> iterateBoidsOMP(std::vector<position> positions, std::vector<direction> directions, int nBoids, int threads)
{
    std::array<std::vector<position>, 2> nextIteration;
    std::vector<position> nextPositions(nBoids);
    std::vector<direction> nextDirections(nBoids);
#pragma omp parallel num_threads(threads)
#pragma omp for
    for (int i = 0; i < nBoids; i++)
    {
        auto nextBoid = iterateBoid(positions, directions, i, nBoids);
        nextPositions[i] = nextBoid[0];
        nextDirections[i] = nextBoid[1];
    }

    nextIteration = {nextPositions, nextDirections};
    return nextIteration;
}
// Args - threads - nBoids
int main(int argc, char* argv[])
{
// Adding a third argument will make the program output in a way that is easier to collect data from.

    int threads = atoi(argv[1]);

    int nBoids = atoi(argv[2]);

    std::vector<std::array<std::vector<position>, 2>> progressedStates;

    progressedStates.push_back(initialState(nBoids));

    Timer timer;
    timer.start();

    for (int i = 0; i < N_FRAMES; i++)
    {
        progressedStates.push_back(iterateBoidsOMP(progressedStates[i][0], progressedStates[i][1], nBoids, threads));
    }

    timer.stop();
    if (argc == 3) 
    {
        std::cout << "Time taken was " << timer.elapsedSeconds() << " seconds \n";
    }
    else std::cout << threads << " " << timer.elapsedSeconds() << "\n";

    if (OUTPUT)
    {
        writeToFile(progressedStates, N_FRAMES, nBoids);
    }
}