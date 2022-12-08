#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <iostream>
#include <tgmath.h>
#include <array>
#include <cmath>
#include <vector>
#include <time.h>
#include <math.h>
#include <chrono>
#include <immintrin.h>
#include <fstream>
#include "Timer.cpp"

#define PI 3.141592653589793238462643383279502884197169399375105820974944

// Coefficients for influence of different factors
#define LOCAL_RANGE 10
#define BOID_VIEW_ANGLE 2.4
#define SEPERATION_COEFFICIENT 1
#define ALIGNMENT_COEFFICIENT 0.1
#define COHESION_COEFFICIENT 0.1
#define BOID_SPEED 20
#define STEER_COEFFICIENT 0.8

// Bounds for the Boids simulation
#define X_SIZE 300
#define Y_SIZE 300
#define TIME_STEP 0.03333333333

// Datatypes
using position = __m128d;
using direction = __m128d;

// Returns a vector of the position of all the local boids to a chosen boid
std::vector<int> localBoids(int boidIndex, std::vector<position> currentPositions, direction currentDirection, int nBoids)
{
    // Storage for positions of local boids
    std::vector<int> localList;

    // Iterates for all boids
    for (int i = 0; i < nBoids; i++)
    {
        // No checking of the same boid
        if (i == boidIndex)
        {
            continue;
        }

        // Finds distance to boid
        auto diffVector = _mm_sub_pd(currentPositions[boidIndex], currentPositions[i]);
        auto squareDiffVector = _mm_mul_pd(diffVector, diffVector);
        double rDiffSquare = squareDiffVector[0] + squareDiffVector[1];

        // Checks for locality
        if (rDiffSquare < pow(LOCAL_RANGE, 2))
        {
            auto dot = currentDirection[0] * diffVector[0] + currentDirection[1] * diffVector[1];
            auto det = currentDirection[0] * diffVector[1] - currentDirection[1] * diffVector[0];
            auto angle = atan2(det, dot);
            if (abs(angle) < BOID_VIEW_ANGLE)
            {
                // Adds to vector
                localList.push_back(i);
            }
        }
    }

    return localList;
}

// Seperation stops the boids from colliding with each other
direction seperation(std::vector<position> &localBoids, position boidPosition)
{
    direction seperation = {0, 0};
    int length = localBoids.size();

    for (int i = 0; i < length; i++)
    {

        auto diffVector = _mm_sub_pd(boidPosition, localBoids[i]);
        auto squareDiffVector = _mm_mul_pd(diffVector, diffVector);
        double rSquareDiff = squareDiffVector[0] + squareDiffVector[1];
        __m128d rFourDiff_vector = _mm_set1_pd(rSquareDiff*rSquareDiff);
        auto seperationIntermediate = _mm_div_pd(diffVector, rFourDiff_vector);
        seperation = _mm_add_pd(seperation, seperationIntermediate);
    }

    return seperation;
}

// Alignment means the boids try to match velocities with surrounding boids
direction alignment(std::vector<direction> &localDirections, direction currentDirection)
{
    direction alignment = {0, 0};
    int length = localDirections.size();

    for (int i = 0; i < length; i++)
    {
        alignment = _mm_add_pd(alignment, localDirections[i]);
    }

    auto lengthVector = _mm_set1_pd(length);
    alignment = _mm_div_pd(alignment, lengthVector);

    direction alignmentSteer = _mm_sub_pd(alignment, currentDirection);

    return alignmentSteer;
}

// Cohesions means the boids fly towards the centre of mass of local boids
direction cohesion(std::vector<position> &localBoids, position boidPosition)
{
    position massCentre = {0, 0};
    int length = localBoids.size();

    for (int i = 0; i < length; i++)
    {
        massCentre = _mm_add_pd(massCentre, localBoids[i]);
    }
    
    auto lengthVector = _mm_set1_pd(length);
    massCentre = _mm_div_pd(massCentre, lengthVector);

    direction cohesion = _mm_sub_pd(massCentre, boidPosition);

    auto cohesionSquare = _mm_mul_pd(cohesion, cohesion);
    double cohesionScalar = sqrt(cohesionSquare[0] + cohesionSquare[1]);

    auto cohesionScalarVector = _mm_set1_pd(cohesionScalar);
    cohesion = _mm_div_pd(cohesion, cohesionScalarVector);

    return cohesion;
}

// Given current list of positions and directions, calculates the next step
std::array<position, 2> iterateBoid(std::vector<position> positions, std::vector<direction> directions, int index, int nBoids)
{
    std::array<position, 2> nextIteration;

    std::vector<int> localIndices = localBoids(index, positions, directions[index], nBoids);
    int localLength = localIndices.size();

    direction boidVector;

    if (localLength > 0)
    {
        std::vector<position> localPositions;
        std::vector<direction> localDirections;

        for (int j = 0; j < localLength; j++)
        {
            localPositions.push_back(positions[localIndices[j]]);
            localDirections.push_back(directions[localIndices[j]]);
        }

        direction seperationVector = seperation(localPositions, positions[index]);
        direction alignmentVector = alignment(localDirections, directions[index]);
        direction cohesionVector = cohesion(localPositions, positions[index]);

        direction seperationCoeffVector = _mm_set1_pd(SEPERATION_COEFFICIENT);
        direction alignmentCoeffVector = _mm_set1_pd(ALIGNMENT_COEFFICIENT);
        direction cohesionCoeffVector = _mm_set1_pd(COHESION_COEFFICIENT);
        direction steerVector;

        seperationVector = _mm_mul_pd(seperationCoeffVector, seperationVector);
        alignmentVector = _mm_mul_pd(alignmentCoeffVector, alignmentVector);
        cohesionVector = _mm_mul_pd(cohesionCoeffVector, cohesionVector);

        steerVector = _mm_add_pd(seperationVector,_mm_add_pd(alignmentVector,cohesionVector));

        boidVector = _mm_add_pd(steerVector, directions[index]);

        double boidScalar = sqrt(pow(boidVector[0], 2) + pow(boidVector[1], 2));

        if (boidScalar < (BOID_SPEED - BOID_SPEED / 8))
        {
            boidVector[0] += 0.1 * boidVector[0] / boidScalar;
            boidVector[1] += 0.1 * boidVector[1] / boidScalar;
        }

        if (boidScalar > (BOID_SPEED + BOID_SPEED / 8))
        {
            boidVector[0] -= 0.1 * boidVector[0] / boidScalar;
            boidVector[1] -= 0.1 * boidVector[1] / boidScalar;
        }
    }
    else
    {
        boidVector = directions[index];
    }

    nextIteration[1][0] = boidVector[0];
    nextIteration[1][1] = boidVector[1];
    nextIteration[0][0] = std::fmod(std::fmod(positions[index][0] + TIME_STEP * boidVector[0], X_SIZE) + X_SIZE, X_SIZE);
    nextIteration[0][1] = std::fmod(std::fmod(positions[index][1] + TIME_STEP * boidVector[1], Y_SIZE) + Y_SIZE, Y_SIZE);

    // std::cout << "woo \n";
    return nextIteration;
}

std::array<std::vector<position>, 2> initialState(int nBoids)
{
    srand(time(NULL));

    std::array<std::vector<position>, 2> state;

    std::vector<position> t0Positions(nBoids);

    std::vector<position> t0Directions(nBoids);

    // Boids are set in a grid on integer positions

    for (int i = 0; i < nBoids; i++)
    {

        double x, y, directionRadians;

        x = static_cast<double>(rand()) / (static_cast<double>(RAND_MAX / (X_SIZE - 2)));
        y = static_cast<double>(rand()) / (static_cast<double>(RAND_MAX / (Y_SIZE - 2)));

        x += 1;
        y += 1;

        (t0Positions)[i][0] = x;
        (t0Positions)[i][1] = y;

        // Choosing random directions

        directionRadians = static_cast<double>(rand()) / (static_cast<double>(RAND_MAX / (2 * PI)));

        (t0Directions)[i][0] = BOID_SPEED * cos(directionRadians);
        (t0Directions)[i][1] = BOID_SPEED * sin(directionRadians);
    }

    // std::array<std::array<position, nBoids>, 2> state = {t0Positions, t0Directions};
    state = {t0Positions, t0Directions};
    return state;
}

void writeToFile(std::vector<std::array<std::vector<position>, 2>> progressedStates, int nFrames, int nBoids)
{
    std::ofstream outputFile("boidsOutput.txt");

    if (outputFile.is_open())
    {
        outputFile << nBoids << "\n"
                   << nFrames << "\n"
                   << X_SIZE << "\n"
                   << Y_SIZE << "\n\n";

        for (int i = 0; i < nFrames; i++)
        {
            for (int j = 0; j < nBoids; j++)
            {
                outputFile << (progressedStates)[i][0][j][0] << "\n"
                           << (progressedStates)[i][0][j][1] << "\n"
                           << (progressedStates)[i][1][j][0] << "\n"
                           << (progressedStates)[i][1][j][1] << "\n";
            }
        }
        outputFile.close();
    }
}