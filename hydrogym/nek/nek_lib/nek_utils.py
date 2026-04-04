"""
Collection of NEK5000 usage
"""

import os
import shutil
import subprocess
from pathlib import Path

from ..configs import Runner as drl
from ..configs import Simulation as nek
from .writer_int_pos import write_channel

"""
A checklist of dependencies: 

        - CASE.re2
        - CASE.ma2
        - CASE.par
        - SESSION.NAME
        - int_pos (if needed)
        - nek5000 

"""

V17_PARAM_MAPPING = {
    "density": "p001",
    "viscosity": "p002",
    "numSteps": "p011",
    "dt": "p012",
    "writeInterval": "p015",
    "target_cfl": "p026",
    "writeLA2": "p070",
    "p_residualTol": "p021",
    "v_residualTol": "p022",
    "writePTS": "p051",
    "READCHKPT": "p076",
    "CHKPFNUMBER": "p067",
    "CHKPINTERVAL": "p075",
    "AVSTEP": "p087",
    "IOSTEP": "p088",
    "ndrl": "p089",
    "znmf_avg": "p090",
}


class NEK_INIT:
    def __init__(self, nek: nek, drl: drl, rank_folder):
        """
        A class for initialization of NEK Dependencies
        nek:[dataclass]Simulation config
        drl:[dataclass]DRL config
        rank_folder:[str]target folders to run drl
        """
        self.nek = nek
        self.drl = drl
        self.rank_folder = rank_folder
        self.is_done = []

    def _is_v17(self) -> bool:
        version = getattr(self.nek, "solver_version", "v19")
        return "v17" in str(version).lower()

    def get_Case_Files(self):
        """
        Get required case files for running simulation
        IF it is complusory, it will be rewritten no matter if the file exists
        IF it is optional, it will NOT be covered if it Exist.
        """
        if self._is_v17():
            checklist = {
                "must": [
                    # Solver
                    "nek5000",
                    # Mesh
                    f"{self.nek.CASENAME}.re2",
                    f"{self.nek.CASENAME}.map",
                    f"{self.nek.CASENAME}.wall",
                    f"{self.nek.CASENAME}.restart",
                    # Time Series Probs
                    "stat_pts.in",
                    # Tripping
                    "forparam.i",
                ],
                "option": [
                    "SIZE",
                    f"mask_{self.nek.CASENAME}0.f00002",
                ],
            }
        else:
            checklist = {
                "must": [
                    # Solver
                    "nek5000",
                    # Mesh
                    f"{self.nek.CASENAME}.re2",
                    f"{self.nek.CASENAME}.ma2",
                    f"{self.nek.CASENAME}.usr",
                    "int_pos",
                ],
                "option": [
                    "SIZE",
                    # Rotation
                    #'int_pos',
                ],
            }
        for fname in checklist["must"]:
            from_file = os.path.join(self.nek.compile_path, fname)
            to_file = os.path.join(self.rank_folder, fname)
            if not os.path.exists(from_file):
                raise FileNotFoundError(f"[IO] {from_file} not EXIST!")
            else:
                # IF the file exist, we clean it to ensure everything works fine.
                if os.path.exists(to_file):
                    os.remove(to_file)
                    print(f"[IO] REMOVE EXIST: {to_file}")

                shutil.copy(from_file, to_file)
                print(f"[IO] {to_file} COPIED", flush=True)

        for fname in checklist["option"]:
            from_file = os.path.join(self.nek.compile_path, fname)
            to_file = os.path.join(self.rank_folder, fname)
            if not os.path.exists(to_file):
                print(f"[IO] WARNING: {to_file} not EXIST", flush=True)
                shutil.copy(from_file, to_file)
            else:
                print(f"[IO] {to_file} EXIST", flush=True)
                pass

        return True

    def write_SESSION_NAME(self):
        """Write the session name and where the code should be executed"""

        solver_root = Path(self.rank_folder)
        fileName = os.path.join(solver_root, "SESSION.NAME")
        is_exist = os.path.exists(fileName)
        if is_exist:
            os.remove(fileName)
        command = "cd %s && touch SESSION.NAME" % solver_root
        subprocess.call(command, shell=True)
        command = "cd %s && echo %s > SESSION.NAME" % (solver_root, self.nek.CASENAME)
        subprocess.call(command, shell=True)
        command = "cd %s && echo $(pwd) >> SESSION.NAME" % solver_root
        subprocess.call(command, shell=True)
        print("[IO] SESSION NAME WRITTEN", flush=True)
        return True

    def rewrite_REA_v17(self):
        """
        Re-Write parameter files for NEK version <= 17.
        For the controllable params, please see config.
        """
        output_path = os.path.join(self.rank_folder, f"{self.nek.CASENAME}.rea")

        with open(output_path, "r") as f:
            lines = f.readlines()
            updated_lines = []
            for line in lines:
                if "p" in line:
                    parts = line.split()
                    if len(parts) > 1 and parts[1] in V17_PARAM_MAPPING.values():
                        for attr, pkey in V17_PARAM_MAPPING.items():
                            if parts[1] == pkey:
                                parts[0] = f"{getattr(self.nek, attr):.6E}"
                                line = "\t".join(parts) + "\n"
                                break
                updated_lines.append(line)

        with open(output_path, "w") as f:
            f.writelines(updated_lines)

        return True

    def rewrite_REA_v19(self):
        """
        Write parameter files for NEK version >= 19
        """
        fname = os.path.join(self.rank_folder, f"{self.nek.CASENAME}.par")
        print(f"[IO] Writing .par file:\n{fname}", flush=True)
        with open(fname, "w") as fpar:
            fpar.write("# nek parameter file\n")

            # General Setup
            fpar.write("[GENERAL]\n")
            if self.nek.stopAt is not None:
                fpar.write("stopAt = %s   \n" % self.nek.stopAt)
                fpar.write("numSteps = %d \n" % self.nek.numSteps)

            fpar.write("dt = %.3e           \n" % self.nek.dt)
            fpar.write("timeStepper = %s  \n" % self.nek.timeStepper)
            fpar.write("variableDt = %s   \n" % self.nek.variableDt)
            fpar.write("\n")
            fpar.write("writeControl = %s \n" % self.nek.writeControl)
            fpar.write("writeInterval = %d\n" % self.nek.writeInterval)
            fpar.write("\n")
            fpar.write("dealiasing = %s   \n" % self.nek.dealiasing)
            fpar.write("filtering = %s    \n" % self.nek.filtering)
            fpar.write("filterWeight = %f \n" % self.nek.filterWeight)
            fpar.write("filterCutoffRatio = %f \n" % self.nek.filterCutoffRatio)
            fpar.write("\n")

            # DRL SETUP
            userp = 1
            fpar.write("#------DRL SETUP-------\n")
            fpar.write("userParam%02d = %s \n" % (userp, self.nek.ndrl))  # Number of DRL step
            userp += 1
            fpar.write("userParam%02d = %s \n" % (userp, self.nek.znmf_avg))  # Average Z-mode for DRL
            userp += 1
            fpar.write("userParam%02d = %s \n" % (userp, self.nek.y_sensing))  # Sensing plane location for DRL
            userp += 1
            fpar.write("userParam%02d = %s \n" % (userp, self.nek.retau))  # Reynolds number for reference channel
            userp += 1
            fpar.write("userParam%02d = %s \n" % (userp, self.nek.ys_bdf))  # Sensing plane location for Body-Force
            userp += 1
            fpar.write("userParam%02d = %s \n" % (userp, self.nek.amp_bdf))  # Amplitude for Body-Force
            userp += 1
            fpar.write("userParam%02d = %s \n" % (userp, self.nek.ret_bdf))  # Scale for Body-Force
            # userp+ = 1
            fpar.write("#---------------------\n")
            fpar.write("\n")

            # Problem Type
            fpar.write("[PROBLEMTYPE]\n")
            fpar.write("stressFormulation = %s\n" % self.nek.stressFormulation)
            fpar.write("variableProperties = %s\n" % self.nek.variableProperties)
            fpar.write("\n")

            # PRESSURE
            fpar.write("[PRESSURE]\n")
            fpar.write("residualTol = %e\n" % self.nek.p_residualTol)
            fpar.write("residualProj = %s\n" % self.nek.p_residualProj)
            fpar.write("\n")

            # VELOCITY
            fpar.write("[VELOCITY]\n")
            fpar.write("residualTol = %e\n" % self.nek.v_residualTol)
            fpar.write("residualProj = %s\n" % self.nek.v_residualProj)
            fpar.write("density = %f\n" % self.nek.density)
            fpar.write("viscosity = %f\n" % self.nek.viscosity)
            fpar.write("advection = %s\n" % self.nek.advection)
            fpar.write("\n")

            # _RUNPAR
            fpar.write("[_RUNPAR]\n")
            fpar.write("PARFWRITE = %s\n" % self.nek.PARFWRITE)
            fpar.write("outparfile = %s\n" % self.nek.PARFNAME)
            fpar.write("\n")

            # MONITOR
            fpar.write("[_MONITOR]\n")
            fpar.write("LOGLEVEL = %d\n" % self.nek.LOGLEVEL)
            fpar.write("WALLTIME = %s\n" % self.nek.WALLTIME)
            fpar.write("\n")

            # _CHECKPOINT
            fpar.write("[_CHKPOINT]\n")
            fpar.write("READCHKPT = %s\n" % self.nek.READCHKPT)
            fpar.write("CHKPFNUMBER = %d\n" % self.nek.CHKPFNUMBER)
            fpar.write("CHKPINTERVAL = %d\n" % self.nek.CHKPINTERVAL)
            fpar.write("\n")

            # _STAT
            fpar.write("[_STAT]\n")
            fpar.write("AVSTEP = %d\n" % self.nek.AVSTEP)
            fpar.write("IOSTEP = %d\n" % self.nek.IOSTEP)
            fpar.write("\n")

            # _STAT
            fpar.write("[_TSRS]\n")
            fpar.write("SMPSTEP = %d\n" % self.nek.SMPSTEP)
            fpar.write("\n")
            fpar.close()
            print(f"[IO] WRITTEN .par file: {fname}", flush=True)
        return True

    def init_restart(self):
        """Copy the restart file to the target folder only if RSTART NOT EXIST"""

        restart_folder = os.path.join(self.nek.restart_folder, f"init_{self.drl.rank}")
        rs_list = os.listdir(restart_folder)
        rs_list = [f for f in rs_list if "rs" in f]

        rs_exist = os.listdir(self.rank_folder)
        rs_exist = [f for f in rs_exist if "rs" in f]

        if len(rs_exist) == 0:
            print("[INIT] IMPORTING RESTART FILES!", flush=True)
            for rsfile in rs_list:
                rsfile = os.path.join(restart_folder, rsfile)
                shutil.copy(rsfile, dst=self.rank_folder)
                print(f"[STB3] RS6 file RESET: {rsfile}", flush=True)
                # print('[STB3] RS6 file RESET',flush=True)
        else:
            file_example = rs_exist[0]
            loc = file_example.find("rs")
            rsx = int(file_example[loc + 2 : loc + 3])
            print(f"[INIT] {rsx} FILE!", flush=True)
            if len(rs_exist) < rsx // 2:
                print(
                    f"[INIT] {rsx} > {len(rs_list)}: IMPORTING RESTART FILES!",
                    flush=True,
                )
                for rsfile in rs_list:
                    rsfile = os.path.join(restart_folder, rsfile)
                    shutil.copy(rsfile, dst=self.rank_folder)
                    print(f"[STB3] RS6 file RESET: {rsfile}", flush=True)
            else:
                print("[INIT] File Exists no need to copy!", flush=True)

        return True

    def write_timeSeries(self):
        """Write the int_pos file for the case file"""
        is_done = write_channel(
            path=self.nek.compile_path,
            Ret=self.nek.retau,
            yplus=self.nek.y_sensing,
            Lx=self.nek.Lx,
            Lz=self.nek.Lz,
            Nx=self.nek.Nx,
            Nz=self.nek.Nz,
            lx1=self.nek.lx1,
        )
        return is_done

    def main(self):
        # -- Initialize Nek --
        if self._is_v17():
            self.is_done.append(self.get_Case_Files())
            self.is_done.append(self.write_SESSION_NAME())
            self.is_done.append(self.rewrite_REA_v17())
            self.is_done.append(self.init_restart())
        else:
            self.is_done.append(self.write_timeSeries())
            self.is_done.append(self.get_Case_Files())
            self.is_done.append(self.write_SESSION_NAME())
            self.is_done.append(self.rewrite_REA_v19())
            self.is_done.append(self.init_restart())

        if False not in self.is_done:
            return True
        else:
            return False


def remove_sch(current_path):
    file_list = os.listdir(current_path)
    file_list = [f for f in file_list if ".sch" in f]
    if len(file_list) > 0:
        for f in file_list:
            os.remove(os.path.join(current_path, f))
    return
