# University Course Timetabling Problem

The aim of this application is to create a valid solution to the University Course Timetabling Problem 
instance given as input. The implementation is based on a differential evolution algorithm, which attemps 
to lower the penalization of the solution.

## Input
The program expects a problem instance given as input in format that can be found on site: 
http://sferics.idsia.ch/Files/ttcomp2002/IC_Problem/node7.html.

## Output
The best solution found is outputted in format described on site: 
http://sferics.idsia.ch/Files/ttcomp2002/IC_Problem/Output_format.htm.
Error messages are written to standard error output.

## Usage
$ xvales02.py [-h] -i INPUT [-p POPULATION_SIZE] [-g GENERATIONS_COUNT]
    [-c CROSSOVER_RATE] [-m MUTATION_RATE]

Argument meaning:
  -h, --help                
                        show help message and exit
  -i INPUT, --input INPUT
                        input file containing problem description
  -p POPULATION_SIZE, --population-size POPULATION_SIZE
                        size of initial population
  -g GENERATIONS_COUNT, --generations-count GENERATIONS_COUNT
                        number of generations
  -c CROSSOVER_RATE, --crossover-rate CROSSOVER_RATE
                        crossover rate
  -m MUTATION_RATE, --mutation-rate MUTATION_RATE
                        mutation rate

## Testing
Validity of results given by this application has been tested using an official 
solution checking tool available on site: 
http://sferics.idsia.ch/Files/ttcomp2002/IC_Problem/Checking_solutions.htm.

The tool checks if the solution meets all hard constraints and it gives 
a penalization for each soft constraint, which both corresponded to 
results of the implemented application.

## Examples
11 example test files are located in the subfolder 'Examples'. 5 of them are 
small, other 5 are of medium size and 1 problem definition is large. 
