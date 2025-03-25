from tap import Tap

# Whole bunch of HydroGym-specific imports


class ArgumentParser(Tap):
    environment: str  # Choice of environment to be run
    algorithm: str  # Choice of RL algorithm to be run
    

def main():
    # Read in the command-line arguments
    args = ArgumentParser().parse_args()

    # Checks for the correctness of the inputs

    print("We did it!", args.environment)


if __name__ == "__main__":
    main()
