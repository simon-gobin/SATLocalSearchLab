#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 17:44:57 2024

@author: dgrimes
"""
# ============for benchmark
import pandas as pd
import logging
import os

# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# ===================original import

import numpy as np
from time import perf_counter
import random
import sys
from os import listdir

myStudentNum = 292931  # TODO Replace 12345 with your student number where R0002345 = 2345
random.seed(myStudentNum)
np.random.seed(myStudentNum)

'''
Data structures
state:          the current candidate solution
clauses:        list of lists, each list contains the literal of the clause
unsat_clauses:  the index of each currently unsat clause
makecounts:     the current makecount for each variable 
                (number of currently unsat clauses involving the variable) 
breakcounts:    the current breakcount for each variable 
                (number of currently sat clauses involving the variable, 
                 where the variable is the only satisfying literal of the clause
                 i.e the clause will go unsat if this variable is flipped) 
litToClauses:   dictionary containing 2*vars entries, one for each literal associated with each variable

NB: The variables and their associated literals are numbered 1..n rather than 0..n-1, 
so to allow us to index in with variable number without having to -1 every time, 
a lot of the data structures are set up to be of size n+1, with the first element 
(index 0) ignored
'''


class GSAT_solver:

    def __init__(self, file, _h, _wp, _maxFlips, _maxRestarts, _tl):
        self.maxFlips = _maxFlips  # input: Number of flips before restarting
        self.maxRestarts = _maxRestarts  # input: Number of restarts before exiting
        self.wp = _wp  # input: walk probability
        self.h = _h  # input: heuristic to choose variable
        self.tl = _tl  # input: tabu length
        self.flips = 0  # current number of flips performed
        self.badflips = 0  # current number of bad flips performed (flips where obj fn was worse after flipping)
        self.restarts = 0  # current number of restarts performed
        self.nVars, self.nClauses, self.clauses, self.litToClauses = -1, -1, [], {}
        self.readInstance(file)
        self.state = [0 for _ in range(self.nVars + 1)]  # State stores current state (i.e. current solution)
        self.makecounts = np.zeros(self.nVars + 1, dtype=int)  # unsat that would go sat
        self.breakcounts = np.zeros(self.nVars + 1, dtype=int)  # sat that would go unsat
        self.lastFlip = np.full(self.nVars + 1, -self.tl)  # Iteration that variable was last updated in
        self.lastFlip[0] = self.maxFlips
        self.bestSol = [0 for _ in range(self.nVars)]  # Current best solution found so far
        self.bestObj = self.nClauses + 1  # Current best objective found so far (obj of bestSol)
        self.breakcounts[0] = self.nClauses + 1  # sat that would go unsat
        self.clauses_abs = [[abs(lit) for lit in clause] for clause in self.clauses]
        self.not_improving_limit = int(0.2 * self.maxFlips)
        self.badflips = 0  # current number of bad flips performed (flips where obj fn was worse after flipping)

    def readInstance(self, fName):
        file = open(fName, 'r')
        current_clause = []
        clauseInd = 0

        for line in file:
            data = line.split()

            if len(data) == 0:
                continue
            if data[0] == 'c':
                continue
            if data[0] == 'p':
                self.nVars = int(data[2])
                self.nClauses = int(data[3])

                continue
            if data[0] == '%':
                break
            if self.nVars == -1 or self.nClauses == -1:
                print("Error, unexpected data")
                sys.exit(0)

            ##now data represents a clause
            for var_i in data:
                literal = int(var_i)
                if literal == 0:
                    self.clauses.append(current_clause)
                    current_clause = []
                    clauseInd += 1
                    continue
                current_clause.append(literal)
                if literal in self.litToClauses:
                    self.litToClauses[literal].add(clauseInd)
                else:
                    self.litToClauses[literal] = set([clauseInd])

        for i in range(1, self.nVars + 1):
            if i not in self.litToClauses:
                self.litToClauses[i] = set()
            if -i not in self.litToClauses:
                self.litToClauses[-i] = set()

        if self.nClauses != len(self.clauses):
            print(self.nClauses, len(self.clauses))
            print("Unexpected number of clauses in the problem")
            sys.exit(0)
        file.close()

    def generateSolution(self):
        for i in range(1, self.nVars + 1):
            choice = [-1, 1]
            self.state[i] = (i * random.choice(choice))

    def initial_cost(self):
        # Compute objective value of initial solution, reset counters and recompute
        self.obj = self.nClauses
        self.unsat_clauses = set()
        self.makecounts = np.zeros(self.nVars + 1, dtype=int)  # unsat that would go sat
        self.breakcounts = np.zeros(self.nVars + 1, dtype=int)  # sat that would go unsat
        self.breakcounts[0] = self.nClauses + 1
        num_unsat = 0
        clsInd = 0
        for clause in self.clauses:
            satLits = 0
            breakV = 0
            cStatus = False
            for lit in clause:
                if lit in self.state:
                    cStatus = True
                    satLits += 1
                    breakV = lit
                if satLits > 1:
                    break
            if satLits == 1:
                self.breakcounts[abs(breakV)] += 1
            if not cStatus:
                num_unsat += 1
                self.unsat_clauses.add(clsInd)
                for lit in clause:
                    self.makecounts[abs(lit)] += 1
            clsInd += 1
        self.obj = num_unsat
        if self.bestObj == -1:
            self.bestObj = num_unsat
            self.bestSol = self.state[1:]

    def flip(self, variable):
        self.flips += 1
        self.state[variable] *= -1
        self.update_counts(variable)
        self.lastFlip[variable] = self.flips

    # Function to update objective value and counts of variables
    # Run after flipping
    def update_counts(self, variable):
        literal = self.state[variable]
        for clauseInd in self.litToClauses[literal]:
            satLits = 0
            if clauseInd in self.unsat_clauses:
                for lit in self.clauses[clauseInd]:
                    self.makecounts[abs(lit)] -= 1
                # Was unsat so only flipvar now satisfies it
                self.breakcounts[variable] += 1
            else:
                for lit in self.clauses[clauseInd]:
                    if lit in self.state:
                        satLits += 1
                        if lit != literal:
                            breaklit = lit
                if satLits == 2:
                    self.breakcounts[abs(breaklit)] -= 1
        self.unsat_clauses = self.unsat_clauses - self.litToClauses[literal]
        for clauseInd in self.litToClauses[literal * (-1)]:
            satLits = 0
            cStatus = False
            for lit in self.clauses[clauseInd]:
                if lit in self.state:
                    cStatus = True
                    satLits += 1
                    breaklit = lit
            if satLits == 1:
                self.breakcounts[abs(breaklit)] += 1
            if not cStatus:
                self.breakcounts[variable] -= 1  # flipvar was only 1 satisfying it
                self.unsat_clauses.add(clauseInd)
                for lit in self.clauses[clauseInd]:
                    self.makecounts[abs(lit)] += 1
        self.obj = len(self.unsat_clauses)

    def selectVar(self):
        if self.h == "gsat":
            return self.selectGSATvar()
        elif self.h == "gwsat":
            return self.selectGWSATvar()
        elif self.h == "hsat":
            return self.selectHSATvar()
        elif self.h == "walksat":
            return self.selectWalkSATvar()
        elif self.h == "hsatTabu":
            return self.selectHSATtabuvar()
        elif self.h == "grimesHsat":
            return self.selectGrimesHSATvar()
        else:
            return self.selectGrimesWSATvar()

    def selectGSATvar(self):
        gains = self.makecounts - self.breakcounts
        hvars = np.where(gains == np.amax(gains))[0]
        return np.random.choice(hvars)

    def selectRWvar(self):
        hvars = np.where(self.makecounts > 0)[0]
        return np.random.choice(hvars)

    def selectGWSATvar(self):
        if random.random() < self.wp:
            nextvar = self.selectRWvar()
        else:
            nextvar = self.selectGSATvar()
        return nextvar

    def selectHSATvar(self):
        gains = self.makecounts - self.breakcounts
        hvars = np.where(gains == np.amax(gains))[0]
        return hvars[np.where(self.lastFlip[hvars] == np.amin(self.lastFlip[hvars]))[0]][0]

    def selectWalkSATvar(self):
        nextCls = random.choice(tuple(self.unsat_clauses))
        varsCls = [abs(lit) for lit in self.clauses[nextCls]]
        gains = np.array([self.breakcounts[i] for i in varsCls])
        hvars = np.where(gains == 0)[0]
        if len(hvars) > 0:
            return varsCls[np.random.choice(hvars)]
        elif random.random() < self.wp:
            return random.choice(varsCls)
        else:
            hvars = np.where(gains == np.amin(gains))[0]
            return varsCls[np.random.choice(hvars)]

    def selectHSATtabuvar(self):
        '''
        Add tabu search to basic hsat, with aspiration criteria of
        improving on best solution found so far in this search attempt
        (i.e. not including from previous restarts).
        Advice: adapt Hsat code from selectHSATvar and add
        tabu criteria using LastFlip data structure
        '''

        gains = self.makecounts - self.breakcounts
        current_best = getattr(self, "runBestObj", self.bestObj)

        tabu_mask = (self.flips - self.lastFlip) < self.tl

        # change the for loop for vectorisation
        predicted_obj = self.obj - gains
        aspiration_mask = (tabu_mask & (predicted_obj < current_best))

        allowed_mask = np.logical_or(~tabu_mask, aspiration_mask)
        allowed_vars = np.where(allowed_mask)[0]

        if allowed_vars.size == 0:
            return self.selectHSATvar()

        best_gain = np.max(gains[allowed_vars])
        best_vars = allowed_vars[gains[allowed_vars] == best_gain]

        oldest_vars = best_vars[self.lastFlip[best_vars] == np.min(self.lastFlip[best_vars])]
        return int(np.random.choice(oldest_vars))

    def selectGrimesWSATvar(self):
        '''
        (a) If zero damage variable, zero damage variable step
            (select maximum positive gain variable from variables with positive gain > 0,
             and negative gain = 0, if such variables exists)
        (b) Random walk step with probability wp:
                choose randomly from variables involved in at least one unsatisfied clause
        (c) Otherwise randomly choose unsat clause and choose variable with maximum net gain, breaking ties randomly
        Advice: adapt WalkSAT code from selectWalkSATvar
        '''
        # Net gains for all variables
        gains = self.makecounts - self.breakcounts

        # ---------- (a) zero damage variable step ----------
        # "positive gain > 0 and negative gain = 0"  ==>
        # makecounts > 0 and breakcounts == 0
        zero_damage_mask = (self.makecounts > 0) & (self.breakcounts == 0)
        zero_damage_vars = np.where(zero_damage_mask)[0]

        if zero_damage_vars.size > 0:
            # Randomly select one zero-damage variable
            return int(np.random.choice(zero_damage_vars))

        # ---------- (b) random walk step with prob wp ----------
        elif random.random() < self.wp:
            vars_unsat = {abs(lit)
                          for cls_idx in self.unsat_clauses
                          for lit in self.clauses[cls_idx]}
            if vars_unsat:
                return random.choice(list(vars_unsat))

        # ---------- (c) otherwise: pick best gain variable in random unsat clause ----------
        else:
            # Randomly choose an unsatisfied clause
            nextCls = random.choice(list(self.unsat_clauses))
            varsCls = self.clauses_abs[nextCls]
            gains_cls = gains[varsCls]

            # Variables with maximum net gain
            best_gain = np.max(gains_cls)
            best_vars = [v for v, g in zip(varsCls, gains_cls) if g == best_gain]
            return random.choice(best_vars)

    def selectGrimesHSATvar(self):
        '''
        (a) If zero damage variable, zero damage variable step
            (select maximum positive gain variable from variables with positive gain > 0,
             and negative gain = 0, if such variables exists)
        (b) Age walk step with probability wp:
                choose variable involved in at least one unsatisfied clause that was flipped the least recently
        (c) Otherwise randomly choose unsat clause and choose variable with maximum net gain, breaking ties randomly
        Advice: adapt WalkSAT code from selectWalkSATvar
        '''
        # Net gains for all variables
        gains = self.makecounts - self.breakcounts

        # ---------- (a) zero damage variable step ----------
        # "positive gain > 0 and negative gain = 0"  ==>
        # makecounts > 0 and breakcounts == 0
        zero_damage_mask = (self.makecounts > 0) & (self.breakcounts == 0)
        zero_damage_vars = np.where(zero_damage_mask)[0]

        if zero_damage_vars.size > 0:
            # Randomly select one zero-damage variable
            return int(np.random.choice(zero_damage_vars))

        # ---------- (b) random walk step with prob wp ----------
        elif random.random() < self.wp:
            vars_unsat = {abs(lit)
                          for cls_idx in self.unsat_clauses
                          for lit in self.clauses[cls_idx]}
            if vars_unsat:
                vars_unsat = np.array(list(vars_unsat), dtype=int)
                last_flips = self.lastFlip[vars_unsat]
                oldest_value = np.min(last_flips)
                oldest_vars = vars_unsat[last_flips == oldest_value]
                return random.choice(list(oldest_vars))

        # ---------- (c) otherwise: pick best gain variable in random unsat clause ----------
        else:
            # Randomly choose an unsatisfied clause
            nextCls = random.choice(list(self.unsat_clauses))
            varsCls = self.clauses_abs[nextCls]
            gains_cls = gains[varsCls]

            # Variables with maximum net gain
            best_gain = np.max(gains_cls)
            best_vars = [v for v, g in zip(varsCls, gains_cls) if g == best_gain]
            return random.choice(best_vars)

    def solve(self):
        self.restarts = 0
        totalFlips = 0
        while self.restarts < self.maxRestarts and self.bestObj > 0:
            self.restarts += 1
            self.generateSolution()
            self.initial_cost()
            self.flips = 0
            self.badflips = 0
            self.lastFlip = np.full(self.nVars + 1, -self.tl)
            self.lastFlip[0] = self.maxFlips
            while (
                    self.flips < self.maxFlips
                    and self.bestObj > 0
                    and self.badflips < self.not_improving_limit
            ):
                prev_obj = self.obj  # objective before the flip
                nextvar = self.selectVar()
                self.flip(nextvar)  # updates self.obj
                # Bad flip: objective got worse than previous
                if self.obj > prev_obj:
                    self.badflips += 1
                else:
                    # Reset counter when we improve or stay the same
                    self.badflips = 0
                if self.obj < self.bestObj:
                    self.bestObj = self.obj
                    self.bestSol = self.state[1:]

            totalFlips += self.flips

        if self.bestObj == 0:
            solutionChecker(self.clauses, self.bestSol)
        return totalFlips, self.restarts, self.bestObj


def solutionChecker(clauses, sol):
    unsat_clause = 0
    for clause in clauses:
        cStatus = False
        for var in clause:
            if var in sol:
                cStatus = True
                break
        if not cStatus:
            unsat_clause += 1
    if unsat_clause > 0:
        print("UNSAT Clauses: ", unsat_clause)
        return False
    return True


def run_config_on_all_instances(filesDir, alg, nRuns, maxRes, maxFlips, wp, tl):
    """
    Runs one parameter configuration on all instances in filesDir
    and returns a list of result dicts (one per instance).
    """
    results = []
    last_digit = str(myStudentNum)[-1]

    for filename in os.listdir(filesDir):
        if not filename.endswith(last_digit + ".cnf"):
            continue

        satInst = os.path.join(filesDir, filename)
        avgRestarts = avgFlips = avgUnsatC = avgTime = 0.0
        unsolved = 0

        for i in range(nRuns):
            # keep same seeding logic as your main()
            random.seed(myStudentNum + i * 100)
            np.random.seed(myStudentNum + i * 100)

            gsat = GSAT_solver(satInst, alg, wp, maxFlips, maxRes, tl)
            startPython = perf_counter()
            ctrFlips, ctrRestarts, ctrObj = gsat.solve()
            stopPython = perf_counter()

            avgFlips += ctrFlips
            avgRestarts += ctrRestarts
            avgUnsatC += ctrObj
            avgTime += (stopPython - startPython)
            if ctrObj > 0:
                unsolved += 1

        results.append({
            "instance": filename,
            "algo": alg,
            "nRuns": nRuns,
            "maxRes": maxRes,
            "maxFlips": maxFlips,
            "wp": wp,
            "tl": tl,
            "solved": nRuns - unsolved,
            "avg_unsat": avgUnsatC / nRuns,
            "avg_restarts": avgRestarts / nRuns,
            "avg_flips": avgFlips / nRuns,
            "avg_time": avgTime / nRuns,
        })

    return results




logger = logging.getLogger(__name__)


def benchmark(
        filesDir="uf150-645",
        csv_path="benchmark_results.csv",
        repetitions=10  # how many times to repeat the full experiment set
):
    """
    Run the SAT benchmark experiments and log results incrementally to CSV.

    Experiments:
      1) Core comparison: gsat, gwsat, hsat, walksat, hsatTabu, grimesHsat, GrimesWSAT
      2) HSAT: vary (nRestarts, nFlips) with constant product
      3) GWSAT: vary wp
      4) HSATTABU: vary tl
    """

    # ------------------------------------------------------------------
    # 0) Default parameter settings for the 6 main algorithms (core exp)
    #    (nRuns, maxRes, maxFlips, wp, tl)
    # ------------------------------------------------------------------
    AlgoDefaults = {
        'gsat': (10, 50, 500, 0.1, 0),
        'gwsat': (10, 50, 500, 0.1, 0),
        'hsat': (10, 50, 500, 0.1, 0),
        'walksat': (10, 50, 500, 0.1, 0),
        'hsatTabu': (10, 50, 500, 0.1, 10),
        'grimesHsat': (10, 50, 500, 0.1, 10),
        'GrimesWSAT': (10, 50, 500, 0.1, 10),
    }

    # ------------------------------------------------------------------
    # Parameter sweeps required by the assignment
    # ------------------------------------------------------------------
    hsat_pairs = [
        (5, 1000),
        (10, 500),
        (20, 250),
        (50, 100),
        (100, 50),
        (250, 20),
    ]

    # GWSAT: sweep wp
    gwsat_wps = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]

    # HSATTABU: sweep tl
    hsatTabu_tls = [1, 5, 10, 20]

    # ===============================
    # 1) Load existing CSV if present
    # ===============================
    if os.path.exists(csv_path):
        logger.info(f"Resuming from existing CSV: {csv_path}")
        existing_df = pd.read_csv(csv_path)

        if not existing_df.empty:
            # We identify completed configs by this tuple
            cols_needed = ["rep", "algo", "nRuns", "maxRes", "maxFlips", "wp", "tl"]
            missing = [c for c in cols_needed if c not in existing_df.columns]
            if missing:
                logger.warning(
                    f"Existing CSV is missing columns {missing}, "
                    "resume logic may not work properly."
                )

            available_cols = [c for c in cols_needed if c in existing_df.columns]

            done_configs = set(
                tuple(row)
                for row in existing_df[available_cols]
                .drop_duplicates()
                .itertuples(index=False, name=None)
            )
        else:
            done_configs = set()
    else:
        logger.info(f"No existing CSV found, starting fresh: {csv_path}")
        done_configs = set()

    # Helper to append a batch of rows to CSV & update done_configs
    def append_rows(rows, rep, alg, nRuns, maxRes, maxFlips, wp, tl):
        # Add repetition + config info into each row dict
        for r in rows:
            r["rep"] = rep
            r["algo"] = alg
            r["nRuns"] = nRuns
            r["maxRes"] = maxRes
            r["maxFlips"] = maxFlips
            r["wp"] = wp
            r["tl"] = tl

        new_df = pd.DataFrame(rows)
        file_exists = os.path.exists(csv_path)

        new_df.to_csv(
            csv_path,
            mode="a",
            header=not file_exists,  # only write header on first write
            index=False,
        )

        config_key = (rep, alg, nRuns, maxRes, maxFlips, wp, tl)
        done_configs.add(config_key)

    # =====================================
    # 2) Iterate over repetitions + configs
    # =====================================
    for rep in range(repetitions):
        logger.info(f"=== Benchmark repetition {rep + 1}/{repetitions} ===")

        # ----------------------------------------------------------
        # A) Core comparison: 6 algorithms with default parameters
        # ----------------------------------------------------------
        logger.info("Starting core comparison experiments...")
        for alg, (nRuns, maxRes, maxFlips, wp, tl) in AlgoDefaults.items():

            logger.info(
                f"[CORE] Running rep={rep}, alg={alg}, "
                f"nRuns={nRuns}, maxRes={maxRes}, maxFlips={maxFlips}, "
                f"wp={wp}, tl={tl}"
            )

            rows = run_config_on_all_instances(
                filesDir, alg, nRuns, maxRes, maxFlips, wp, tl
            )

            # (optional) tag experiment type
            for r in rows:
                r["exp_type"] = "core"

            append_rows(rows, rep, alg, nRuns, maxRes, maxFlips, wp, tl)

        # ----------------------------------------------------------
        # B) HSAT parameter sweep: (nRestarts, nFlips)
        # ----------------------------------------------------------
        logger.info("Starting HSAT (nRestarts, nFlips) parameter sweep...")
        alg = "hsat"
        nRuns = 10
        wp = 0.1
        tl = 0

        for maxRes, maxFlips in hsat_pairs:

            logger.info(
                f"[HSAT_SWEEP] Running rep={rep}, alg={alg}, nRuns={nRuns}, "
                f"maxRes={maxRes}, maxFlips={maxFlips}, wp={wp}, tl={tl}"
            )

            rows = run_config_on_all_instances(
                filesDir, alg, nRuns, maxRes, maxFlips, wp, tl
            )

            for r in rows:
                r["exp_type"] = "hsat_nRes_nFlips"

            append_rows(rows, rep, alg, nRuns, maxRes, maxFlips, wp, tl)

        # ----------------------------------------------------------
        # C) GWSAT parameter sweep: wp
        # ----------------------------------------------------------
        logger.info("Starting GWSAT wp parameter sweep...")
        alg = "gwsat"
        nRuns = 10
        maxRes = 50
        maxFlips = 500
        tl = 0

        for wp in gwsat_wps:


            logger.info(
                f"[GWSAT_SWEEP] Running rep={rep}, alg={alg}, nRuns={nRuns}, "
                f"maxRes={maxRes}, maxFlips={maxFlips}, wp={wp}, tl={tl}"
            )

            rows = run_config_on_all_instances(
                filesDir, alg, nRuns, maxRes, maxFlips, wp, tl
            )

            for r in rows:
                r["exp_type"] = "gwsat_wp"

            append_rows(rows, rep, alg, nRuns, maxRes, maxFlips, wp, tl)

        # ----------------------------------------------------------
        # D) HSATTABU parameter sweep: tl
        # ----------------------------------------------------------
        logger.info("Starting HSATTABU tl parameter sweep...")
        alg = "hsatTabu"
        nRuns = 10
        maxRes = 50
        maxFlips = 500
        wp = 0.1

        for tl in hsatTabu_tls:

            logger.info(
                f"[HSATTABU_SWEEP] Running rep={rep}, alg={alg}, nRuns={nRuns}, "
                f"maxRes={maxRes}, maxFlips={maxFlips}, wp={wp}, tl={tl}"
            )

            rows = run_config_on_all_instances(
                filesDir, alg, nRuns, maxRes, maxFlips, wp, tl
            )

            for r in rows:
                r["exp_type"] = "hsatTabu_tl"

            append_rows(rows, rep, alg, nRuns, maxRes, maxFlips, wp, tl)

    logger.info(f"Benchmark finished. Results in: {csv_path}")


def main():
    if len(sys.argv) == 1:
        filesDir = "uf150-645"
        alg, nRuns, maxRes, maxFlips, wp, tl = "gwsat", 10, 50, 500, 0.1, 10
    elif len(sys.argv) < 8:
        print(len(sys.argv))
        print("Error - Incorrect input")
        print("Expecting python gsat.py [fileDir] [alg] [number of runs] [max restarts]",
              "[max flips] [walk prob]")
        sys.exit(0)
    else:
        _, filesDir, alg, nRuns, maxRes, maxFlips, wp, tl = sys.argv
        nRuns, maxRes, maxFlips, wp, tl = int(nRuns), int(maxRes), int(maxFlips), float(wp), int(tl)

    # Iterate through all instances in the directory that end with
    # last value of your student number
    statsList = ["Inst", "Solved:", "Obj:", "Res:", "Flips:", "Time:"]
    format_row = "{:>12}" * (len(statsList))
    print(alg, nRuns, maxRes, maxFlips, wp)
    print(format_row.format(*statsList))
    for filename in listdir(filesDir):
        if filename.endswith(str(myStudentNum)[-1] + ".cnf"):
            satInst = filesDir + "/" + filename
            avgRestarts, avgFlips, avgUnsatC, avgTime, unsolved = 0, 0, 0, 0, 0

            for i in range(nRuns):
                random.seed(myStudentNum + i * 100)
                np.random.seed(myStudentNum + i * 100)
                gsat = GSAT_solver(satInst, alg, wp, maxFlips, maxRes, tl)
                startPython = perf_counter()
                ctrFlips, ctrRestarts, ctrObj = gsat.solve()
                stopPython = perf_counter()
                avgFlips += ctrFlips
                avgRestarts += ctrRestarts
                avgUnsatC += ctrObj
                avgTime += (stopPython - startPython)
                if ctrObj > 0:
                    unsolved += 1
            resList = [filename, nRuns - unsolved, avgUnsatC / nRuns, avgRestarts / nRuns, avgFlips / nRuns,
                       round(avgTime / nRuns, 3)]
            print(format_row.format(*resList))


'''
Reading in parameters, but it is up to you to implement what needs implementing
TO DO:
1/ Update "myStudentNum" variable to your student number (line 14 in this file)
2/ Implement HsatTabu, grimesWsat, grimesHsat
3/ Implement alternative stopping condition involving badFlips
'''

if __name__ == "__main__":
    # If first argument is "bench", run the benchmark instead of the normal main()
    if len(sys.argv) > 1 and sys.argv[1] == "bench":
        # You can tune these defaults if you want
        df = benchmark(
            filesDir="uf150-645",
            csv_path="benchmark_results_4.csv",
        )
    else:
        main()

algo = ['gsat', 'gwsat', 'hsat', 'walksat', 'hsatTabu', 'grimesHsat', 'GrimesWSAT']
number_of_runs = [5, 10, 20]
max_restarts = [15, 50, 100]
max_flips = [100, 500, 1000]
walk_probs = [0.1, 0.2, 0.5]
tabu = [5, 10, 20]

Algo = {
    'gsat': "gsat 10 50 500 0.1 10 10",
    'gwsat': "gwsat 10 50 100 0.1 10 10",
    'hsat': "hsat 10 50 100 0.1 10 10",
    'walksat': "walksat 10 50 100 0.1 10 10",
    'hsatTabu': "hsatTabu 10 50 100 0.1 10 10",
    'grimesHsat': "grimesHsat 10 50 100 0.1 10 10",
    'GrimesWSAT': "GrimesWSAT 10 50 100 0.1 10 10",
}