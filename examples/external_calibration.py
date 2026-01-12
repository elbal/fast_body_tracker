import fast_body_tracker as fbt


def main():
    final_params = fbt.external_calibration([0, 1, 2])

    #with open("../../data/cal_parameters.json", "w") as f:
    #json.dump(final_params, f, indent=4)

if __name__ == "__main__":
    main()
