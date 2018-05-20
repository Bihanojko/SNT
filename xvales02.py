#!/usr/bin/env python

# University Course Timetabling Problem
# Simulation Tools and Techniques
# Nikola Valesova, xvales02

from __future__ import print_function
import argparse
import sys
import numpy
import random
import time
from copy import deepcopy


# number of available timeslots -> 9 courses per day * 5 days per week
TIMESLOT_COUNT = 45
# number of iterations for neighbourhood moves N1
N1_ITERATION_COUNT = 100
# number of iterations for neighbourhood moves N2
N2_ITERATION_COUNT = 100


class CTTProblem(object):
    """class storing CTTProblem description"""

    def __init__(self, inputData):
        """process input file content and store information"""
        self._population = []               # list of chromosomes, of all solutions of CTTProblem
        self._solution = []                 # single solution
        self._chromosome = []               # one solution of CTTProblem, one member of population
        self._enrolledStudents = []         # list of courses and students enrolled in course
        self._enrolledSubjects = []         # list of studens and courses they are enrolled in
        self._availableRooms = []           # list of courses and rooms available for the given course
        self._collidingCourses = []         # list of colliding courses for every course
        self.solutionFitness = []           # list of fitness of solutions in population

        # parse input file
        try:
            firstLine = inputData[0].split(' ')
            self._courseCount = int(firstLine[0])       # number of courses
            self._roomCount = int(firstLine[1])         # number of rooms
            self._featureCount = int(firstLine[2])      # number of features
            self._studentCount = int(firstLine[3])      # number of students

            # list of room sizes, room size on index idx belongs to room number idx 
            self._roomSizes = []
            for idx in range(self._roomCount):
                self._roomSizes.append(int(inputData[idx + 1]))

            offset = self._roomCount + 1
            # for every student, a list of boolean values representing whether the student addends the course or not 
            self._studentAttends = []
            for idx in range(self._studentCount):
                self._studentAttends.append(inputData[idx * self._courseCount + offset : (idx + 1) * self._courseCount + offset])

            offset += self._courseCount * self._studentCount
            # for every room, a list of boolean values representing whether the room offers the feature or not 
            self._roomFeatures = []
            for idx in range(self._roomCount):
                self._roomFeatures.append(inputData[idx * self._featureCount + offset : (idx + 1) * self._featureCount + offset])

            offset += self._featureCount * self._roomCount
            # for every course, a list of boolean values representing whether the course requires the feature or not 
            self._courseFeatures = []
            for idx in range(self._courseCount):
                self._courseFeatures.append(inputData[idx * self._featureCount + offset : (idx + 1) * self._featureCount + offset])
        except:
            sys.stderr.write('ERROR: Invalid problem specification!\n')
            sys.exit(1)

        self._computeAttendingStudents()
        self._getEnrolledSubjectIDs()
        self._getAvailableRooms()
        self._getCollidingCourses()
        self._getOrderedCourses()        

    def _computeAttendingStudents(self):
        """create a list of numbers of students enrolled in a course"""
        for idx in range(self._courseCount):
            self._enrolledStudents.append(sum([int(x[idx]) for x in self._studentAttends]))

    def _getEnrolledSubjectIDs(self):
        """for every student, create a list of courses the student is enrolled in"""
        for idx in range(self._studentCount):
            self._enrolledSubjects.append([j for j in range(self._courseCount) if self._studentAttends[idx][j] == '1'])

    def _getAvailableRooms(self):
        """for every course, create a list of rooms the course can be held in"""
        roomsMatchingFeaturesList = self._getRoomsWithFeatures()
        roomsMatchingSize = self._getRoomsWithSize()
        [self._availableRooms.append(list(set(a).intersection(set(b)))) for a, b in zip(roomsMatchingFeaturesList, roomsMatchingSize)]

    def _getRoomsWithFeatures(self):
        """for every course, get a list of rooms that offer required features"""
        sufficientRoomsList = []
        for courseID in range(self._courseCount):
            sufficientRoomsList.append([roomID for roomID in range(self._roomCount) if self._hasFeatures(courseID, roomID)])
        return sufficientRoomsList

    def _hasFeatures(self, courseID, roomID):
        """return True if room roomID has features needed by course courseID"""
        for featureID in range(self._featureCount):
            if self._courseFeatures[courseID][featureID] > self._roomFeatures[roomID][featureID]:
                return False
        return True

    def _getRoomsWithSize(self):
        """for every course, get a list of rooms that offer required size"""
        sufficientRoomsList = []
        for courseID in range(self._courseCount):
            sufficientRoomsList.append([roomID for roomID in range(self._roomCount) if self._enrolledStudents[courseID] <= self._roomSizes[roomID]])
        return sufficientRoomsList

    def _getCollidingCourses(self):
        """create a list of courses that collide with the indexed one = courses that have one or more same enrolled students"""
        for courseID in range(self._courseCount):
            self._collidingCourses.append([])
            [self._collidingCourses[courseID].append(x) for x in range(self._courseCount) if x != courseID and self._haveSameStudents([courseID, x])]

    def _getOrderedCourses(self):
        """order courses from the hardest to the easiest to schedule"""
        self._orderedCourses = numpy.argsort([len(x) for x in self._availableRooms])

    def isCorrect(self):
        """check if input file is correct and contains only valid values"""
        # only non-negative integer values on the first line
        if self._courseCount < 0 or self._roomCount < 0 or self._featureCount < 0 or self._studentCount < 0:
            return False

        # all room sizes should be non-negative integer values
        if [roomSize for roomSize in self._roomSizes if roomSize < 0] != []:
            return False

        # studentAttends, roomFeatures and courseFeatures should only contain values 0 and 1
        elements = self._distinctList(self._compoundLists(self._studentAttends + self._roomFeatures + self._courseFeatures))
        subtractedElems = [item for item in elements if item not in ['0', '1']]
        if subtractedElems != []:
            return False
        return True

    @staticmethod
    def _compoundLists(seq):
        """turn list of lists into a single list - [[t]] -> [t]"""
        compoundList = []
        for singleList in seq:
            compoundList += singleList
        return compoundList

    @staticmethod
    def _distinctList(seq):
        """return distinct values of sequence seq"""
        return {}.fromkeys(seq).keys()

    def createInitialPopulation(self, populationSize):
        """create initial population of given number of chromosomes"""
        [self._createFeasibleSolution() for _ in range(populationSize)]

    def _initializeChromosome(self):
        """initialize chromosome as a list of empty sets"""
        del self._chromosome[:]
        [self._chromosome.append([]) for _ in range(self._courseCount)]

    def _createFeasibleSolution(self):
        """create a feasible solution (if possible)"""
        self._initializeChromosome()
        idx = -1

        # find a placement for every course, starting from courses that are the hardest to place
        while idx < self._courseCount - 1:
            idx += 1
            possiblePlacements = self._getPossiblePlacements(self._orderedCourses[idx])
            if possiblePlacements != []:
                self._chromosome[self._orderedCourses[idx]] = random.choice(self._getBestPlacements(possiblePlacements, idx))
            else:
                # apply neighbourhood moves to create a possible placement for current course
                for _ in range(N1_ITERATION_COUNT):
                    self._applyN1()
                    if self._getPossiblePlacements(self._orderedCourses[idx]) != []:
                        idx -= 1
                        break
                if self._getPossiblePlacements(self._orderedCourses[idx]) == []:
                    for _ in range(N2_ITERATION_COUNT):
                        self._applyN2()
                        if self._getPossiblePlacements(self._orderedCourses[idx]) != []:
                            idx -= 1
                            break
                self._chromosome[idx] = []
            
        self._population.append(deepcopy(self._chromosome))

    def _getPossiblePlacements(self, courseID):
        """get a list of tuples [timeslot, room] in which can course courseID be scheduled"""
        possiblePlacements = []
        availableRooms = self._availableRooms[courseID]
        for room in availableRooms:
            [possiblePlacements.append([timeslot, room]) for timeslot in range(TIMESLOT_COUNT) if self._roomVacant(room, timeslot) and self._doesNotCollide(courseID, timeslot)]
        return possiblePlacements

    def _roomVacant(self, room, timeslot):
        """return True if room is in timeslot vacant = no course is scheduled"""
        if room in [self._chromosome[x][1] for x in range(self._courseCount) if len(self._chromosome[x]) > 1 and self._chromosome[x][0] == timeslot]:
            return False
        return True        

    def _doesNotCollide(self, courseID, timeslot):
        """return True if in timeslot, there are scheduled no courses colliding with courseID"""
        if timeslot in [self._chromosome[x][0] for x in range(self._courseCount) if len(self._chromosome[x]) > 1 and x in self._collidingCourses[courseID]]:
            return False
        return True

    def _getBestPlacements(self, placements, idx):
        """return a list of 'best' placements = a list of tuples ['best' timeslot, 'best' room]"""
        bestTimeslots = self._getBestTimeslots(placements)
        bestPlacements = [x for x in placements if x[0] in bestTimeslots]
        bestRooms = self._getBestRooms(bestPlacements, idx)
        bestPlacements = [x for x in bestPlacements if x[1] in bestRooms]
        return bestPlacements
    
    def _getBestTimeslots(self, placements):
        """return a list of 'best' timeslots = the timeslots, in which are the most rooms already occupied"""
        timeslots = self._distinctList([x[0] for x in placements])
        emptyRooms = []
        for timeslot in timeslots:
            emptyRooms.append(len([x[1] for x in self._chromosome if len(x) > 0 and x[0] == timeslot]))
        maxEmptyRooms = max(emptyRooms)
        return [timeslots[x] for x in range(len(timeslots)) if emptyRooms[x] == maxEmptyRooms]

    def _getBestRooms(self, placements, idx):
        """return a list of 'best' rooms = the rooms that can host the minimum of unschedulled courses"""
        rooms = self._distinctList([x[1] for x in placements])
        coursesPlacable = []
        for room in rooms:
            courseCount = 0
            for course in self._orderedCourses[idx:]:
                if room in self._availableRooms[course]:
                    courseCount += 1
            coursesPlacable.append(courseCount)
        minPlacableCourses = min(coursesPlacable)
        return [rooms[x] for x in range(len(rooms)) if coursesPlacable[x] == minPlacableCourses]

    def _applyN1(self):
        """apply neighbourhood move N1"""
        courseID = self._getRandomSetCourseID()
        self._chromosome[courseID] = []
        possiblePlacements = self._getPossiblePlacements(courseID)
        if possiblePlacements != []:
            self._chromosome[courseID] = self._lowestPenaltyPlacement(possiblePlacements, courseID)
    
    def _getRandomSetCourseID(self):
        """return randomly selected course ID from already scheduled courses"""
        while True:
            courseID = random.randint(0, self._courseCount - 1)
            if len(self._chromosome[courseID]) > 0:
                return courseID

    def _lowestPenaltyPlacement(self, possiblePlacements, courseID):
        """return one placement from possiblePlacements that causes the least penalization"""
        if len(possiblePlacements) == 1:
            return possiblePlacements[0]
        
        bestScore = numpy.inf
        solution = self._chromosome
        for placement in possiblePlacements:
            solution[courseID] = placement
            penalization = self._getChromosomePenalization(solution)
            if penalization < bestScore:
                bestScore = penalization
                bestPlacement = placement
        return bestPlacement

    def _applyN2(self):
        """apply neighbourhood move N2"""
        roomID = random.randint(0, self._roomCount - 1)
        coursesInRoom = [x for x in range(self._courseCount) if len(self._chromosome[x]) > 1 and self._chromosome[x][1] == roomID]
        if coursesInRoom != []:
            courseID_1 = random.choice(coursesInRoom)
            courseID_2 = random.choice(coursesInRoom)
            if self._doesNotCollide(courseID_1, self._chromosome[courseID_2][0]) and self._doesNotCollide(courseID_2, self._chromosome[courseID_1][0]):
                self._chromosome[courseID_1][0], self._chromosome[courseID_2][0] = self._chromosome[courseID_2][0], self._chromosome[courseID_1][0]

    def _isFeasible(self, chromosome):
        """check whether current solution is feasible"""
        if self._isFullyDefined(chromosome) and self._meetsH1(chromosome) and self._meetsH2(chromosome) and self._meetsH3(chromosome) and self._meetsH4(chromosome):
            return True
        return False

    def _isFullyDefined(self, chromosome):
        """check if solution is fully defined = every course has a scheduled timeslot"""
        for courseSchedule in chromosome:
            if len(courseSchedule) == 0:
                return False
        return True

    def _meetsH1(self, chromosome):
        """check whether current solution meets hard constraint H1 =
            no student has more than one course at the same time"""
        simultaneousCoursesList = self._getSimultaneousCourses(chromosome)
        for simultaneousCourses in simultaneousCoursesList:
            if self._haveSameStudents(simultaneousCourses):
                return False
        return True

    def _getSimultaneousCourses(self, chromosome):
        """get a list of courses that are scheduled at the same timeslot"""
        simultaneousCourses = []
        for timeslot in range(TIMESLOT_COUNT):
            simultaneousCourses.append([idx for idx in range(self._courseCount) if len(chromosome[idx]) > 1 and chromosome[idx][0] == timeslot])
        return filter(lambda x: len(x) > 1, simultaneousCourses)

    def _haveSameStudents(self, simultaneousCourses):
        """check if courses in simultaneousCourses have any common students"""
        for idx in range(self._studentCount):
            attendanceCount = sum([int(self._studentAttends[idx][i]) for i in simultaneousCourses])
            if attendanceCount > 1:
                return True
        return False

    def _meetsH2(self, chromosome):
        """check whether current solution meets hard constraint H2 =
            if all rooms meet features needed by scheduled courses"""
        for idx in range(self._courseCount):
            for x in range(len(self._courseFeatures[idx])):
                if len(chromosome[idx]) < 1 or self._courseFeatures[idx][x] > self._roomFeatures[chromosome[idx][1]][x]:
                    return False
        return True

    def _meetsH3(self, chromosome):
        """check whether current solution meets hard constraint H3 =
            number of students attending a course is less than room capacity"""
        for idx in range(self._courseCount):
            if len(chromosome[idx]) < 1 or self._enrolledStudents[idx] > self._roomSizes[chromosome[idx][1]]:
                return False
        return True

    def _meetsH4(self, chromosome):
        """check whether current solution meets hard constraint H4 =
            no more than one event is scheduled in a timeslot in each room"""
        collidingEventsCount = self._distinctList([chromosome.count(x) for x in chromosome])
        if [item for item in collidingEventsCount if item not in [1]] != []:
            return False
        return True

    def removeUnfeasibleSolutions(self):
        """remove unfeasible solutions from population"""
        for idx in range(len(self._population) - 1, -1, -1):
            if not self._isFeasible(self._population[idx]):
                self._population = self._population[:idx] + self._population[idx + 1:]
        if len(self._population) == 0:
            print("NO FEASIBLE SOLUTION FOUND")
            sys.exit(0)

    def evaluatePopulation(self):
        """create a list of fitness of chromosomes in population"""
        [self.solutionFitness.append(self._getChromosomePenalization(x)) for x in self._population]

    def _getChromosomePenalization(self, chromosome):
        """get a score of solution of soft constraints violation"""
        return self._s1Score(chromosome) + self._s2Score(chromosome) + self._s3Score(chromosome)

    def _s1Score(self, chromosome):
        """check the score of current solution of violation of soft constraint S1 =
            student has an event scheduled in the last timeslot of the day"""
        eveningCourseList = self._getEveningCourses(chromosome)
        return sum([self._enrolledStudents[idx] for idx in eveningCourseList])

    def _getEveningCourses(self, chromosome):
        """get a list of courses which are scheduled at the last timeslots of the day"""
        return [idx for idx in range(self._courseCount) if len(chromosome[idx]) > 0 and chromosome[idx][0] % 9 == 8]

    def _s2Score(self, chromosome):
        """check the score of current solution of violation of soft constraint S2 =
            student has more than two consecutive events"""
        score = 0
        enrolledSubjectList = filter(lambda x: len(x) > 2, self._enrolledSubjects)
        # for every student and his/her enrolled subjects
        for enrolledSubjects in enrolledSubjectList:
            dayTimes = self._getDayTimes(enrolledSubjects, chromosome)
            for oneDay in dayTimes:
                score += self._nConsecutiveScore(oneDay)
        return score

    def _getDayTimes(self, enrolledSubjects, chromosome):
        """get a list of timeslots enrolledSubjects are scheduled at each day"""
        timeSlots = sorted([chromosome[x][0] for x in enrolledSubjects if len(chromosome[x]) > 1])
        dayTimes = []
        [dayTimes.append([]) for _ in range(5)]
        [dayTimes[x].append(y % 9) for x in range(0, 5) for y in timeSlots if y / 9 == x]
        return dayTimes

    def _nConsecutiveScore(self, oneDay):
        """get a penalty of student having more than 2 consecutive events in a day"""
        sequenceList = self._getSequences(oneDay)
        return sum([max(len(seq) - 2, 0) for seq in sequenceList])

    def _getSequences(self, oneDay):
        """get a list of sequencial courses in a day"""
        sequenceList = []
        sequence = []
        maxIdx = len(oneDay)
        for idx in range(maxIdx):
            if idx < maxIdx - 1 and oneDay[idx] + 1 == oneDay[idx + 1]:
                sequence.append(oneDay[idx])
            else:
                sequence.append(oneDay[idx])
                sequenceList.append(sequence)
                sequence = []
        return sequenceList

    def _s3Score(self, chromosome):
        """check the score of current solution of violation of soft constraint S3 =
            student has only one event scheduled on a day"""
        score = 0
        for enrolledSubjects in self._enrolledSubjects:
            courseDays = [int(chromosome[x][0] / 9) for x in enrolledSubjects if len(chromosome[x]) > 1]
            coursesPerDay = [courseDays.count(x) for x in range(0, 5)]
            score += coursesPerDay.count(1)
        return score

    def orderPopulation(self):
        """order population according to descending fitness"""
        inds = numpy.array(self.solutionFitness).argsort()
        population = (numpy.array(self._population)[inds])
        self.solutionFitness.sort()
        self._population = population.tolist()

    def executeDEAlgorithm(self, mutationRate, crossoverRate):
        """execute mutation, crossover, evaluation and selection operations
            to create a better population"""
        mutatedParents = self._mutate(mutationRate)        
        offspring = self._crossover(crossoverRate, mutatedParents)
        penalizations = [self._getChromosomePenalization(offspring[0]), self._getChromosomePenalization(offspring[1])]
        if penalizations[0] < penalizations[1] and penalizations[0] < self.solutionFitness[0]:
            self._updatePopulation(offspring[0], penalizations[0])
        elif penalizations[1] < self.solutionFitness[0]:
            self._updatePopulation(offspring[1], penalizations[1])

    def _getParents(self):
        """randomly select two parents for DE algorithm"""
        return [random.choice(self._population), random.choice(self._population)]

    def _mutate(self, mutationRate):
        """execute parent mutations"""
        parents = self._getParents()
        if random.uniform(0, 1) <= mutationRate:
            return self._mutationOperator1(parents)
        else: 
            return self._mutationOperator2(parents)

    def _mutationOperator1(self, parents):
        """execute first mutation operator"""
        for parent in parents:
            self._chromosome = parent
            self._applyN1()
            parent = self._chromosome
        return parents

    def _mutationOperator2(self, parents):
        """execute second mutation operator"""
        for parent in parents:
            self._chromosome = parent
            self._applyN2()
            parent = self._chromosome
        return parents

    def _crossover(self, crossoverRate, parents):
        """execute crossover operation on two parent timeslots"""
        timeslots = self._getRandomTimeslots()
        courseInTimeslot1 = self._getCoursesInTimeslot(timeslots[0], parents[0])
        courseInTimeslot2 = self._getCoursesInTimeslot(timeslots[1], parents[1])

        for courseInRoom in courseInTimeslot1:
            if self._roomInTimeslotEmpty(parents[1], courseInRoom[1], timeslots[0]) and \
            self._noConflictsOccur(parents[1], courseInRoom[0], timeslots[0]) and \
            random.uniform(0, 1) <= crossoverRate:
                parents[1][courseInRoom[0]] = parents[0][courseInRoom[0]]

        for courseInRoom in courseInTimeslot2:
            if self._roomInTimeslotEmpty(parents[0], courseInRoom[1], timeslots[1]) and \
            self._noConflictsOccur(parents[0], courseInRoom[0], timeslots[1]) and \
            random.uniform(0, 1) <= crossoverRate:
                parents[0][courseInRoom[0]] = parents[1][courseInRoom[0]]

        return parents

    def _getRandomTimeslots(self):
        """return random timeslot ID"""
        return [random.randint(0, TIMESLOT_COUNT - 1), random.randint(0, TIMESLOT_COUNT - 1)]

    def _getCoursesInTimeslot(self, timeslot, parent):
        """get a list of courses scheduled in given timeslot"""
        coursesInTimeslot = []
        [coursesInTimeslot.append([x, parent[x][1]]) for x in range(self._courseCount) if len(parent[x]) > 0 and parent[x][0] == timeslot]
        return coursesInTimeslot

    def _roomInTimeslotEmpty(self, parent, roomID, timeslot):
        """return False if given room is taken in given timeslot"""
        for courseSchedule in parent:
            if len(courseSchedule) > 0 and courseSchedule[0] == timeslot and courseSchedule[1] == roomID:
                return False
        return True

    def _noConflictsOccur(self, parent, courseID, timeslot):
        """return True if no collisions occur"""
        if timeslot in [parent[x][0] for x in range(self._courseCount) if len(parent[x]) > 0 and x in self._collidingCourses[courseID]]:
            return False
        return True

    def _updatePopulation(self, child, penalization):
        """replace worst solution in a population with the newly created child"""
        self._population = [child] + self._population[:-1]
        self.solutionFitness = [penalization] + self.solutionFitness[:-1]

    def printSolution(self):
        """output solution in defined format"""
        [print(str(x[0]) + ' ' + str(x[1])) for x in self._population[0] if len(x) > 0]


def parseArguments():
    """process input arguments and set default values"""
    arg_parser = argparse.ArgumentParser(description="A university course timetabling problem solver using differential evolution algorithm")
    arg_parser.add_argument('-i', '--input', dest="input", type=str, required=True,
                            help="input file containing problem description")
    arg_parser.add_argument('-p', '--population-size', dest="population_size", type=int,
                            default=50,
                            help="size of initial population")
    arg_parser.add_argument('-g', '--generations-count', dest="generations_count", type=int,
                            default=200000,
                            help="number of generations")
    arg_parser.add_argument('-c', '--crossover-rate', dest="crossover_rate", type=float,
                            default=0.8,
                            help="crossover rate")
    arg_parser.add_argument('-m', '--mutation-rate', dest="mutation_rate", type=float,
                            default=0.5,
                            help="mutation rate")
    return arg_parser.parse_args()


def checkArgs(args):
    """check validity of argument values"""
    if args.population_size < 0:
        sys.stderr.write('ERROR: Parameter -p is less than 0!\n')
        sys.exit(3)
    if args.generations_count < 0:
        sys.stderr.write('ERROR: Parameter -g is less than 0!\n')
        sys.exit(3)
    if args.crossover_rate < 0 or args.crossover_rate > 1:
        sys.stderr.write('ERROR: Parameter -c outside of interval <0,1>!\n')
        sys.exit(3)
    if args.mutation_rate < 0 or args.mutation_rate > 1:
        sys.stderr.write('ERROR: Parameter -m outside of interval <0,1>!\n')
        sys.exit(3)


def main():
    """MAIN"""
    random.seed(time.time())

    # process and check given arguments
    args = parseArguments()
    checkArgs(args)
    with open(args.input, 'r') as f:
        inputData = f.read()

    # initialize CTTP definition with data from input file
    CTTProblemDefinition = CTTProblem(filter(lambda x: x != '', inputData.splitlines()))
    if not CTTProblemDefinition.isCorrect():
        sys.stderr.write('ERROR: Invalid problem specification!\n')
        sys.exit(1)

    # create initial population of given size
    CTTProblemDefinition.createInitialPopulation(args.population_size)
    # remove solutions that are not feasible, exit if no feasible solutions are found
    CTTProblemDefinition.removeUnfeasibleSolutions()
    # get a list of fitness of chromosomes in population
    CTTProblemDefinition.evaluatePopulation()
    # order population according to their fitness function
    CTTProblemDefinition.orderPopulation()

    # while no perfect solution was found and number of iterations has not been met
    # execute DE algorithm = execute mutation, crossover, evaluation and selection 
    iterationNumber = 0
    while CTTProblemDefinition.solutionFitness[0] != 0 and iterationNumber < args.generations_count:
        iterationNumber += 1
        CTTProblemDefinition.executeDEAlgorithm(args.mutation_rate, args.crossover_rate)

    # print the best found solution
    CTTProblemDefinition.printSolution()


if __name__ == "__main__":
    main()
