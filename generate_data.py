import argparse
import random
import numpy as np


rng = np.random.default_rng(seed=123)

MIX_COEFF = 0.8


def _mix_adjacent_rock_types(t1, t2, std_dev_proportion=0.1) -> (float, float):
    # the mixing coefficient is distributed according to Gauss; the standard deviation proportion of that number is given as input.
    if std_dev_proportion != 0:
        coeff = rng.normal(loc=MIX_COEFF, scale=0.9*std_dev_proportion)
    else:
        coeff = MIX_COEFF
    if coeff>1:
        coeff = 1
    if coeff < 0.1:
        coeff = 0.1  # swapping is impossible
    t1_after_mixing = coeff*t1 + (1-coeff)*t2
    t2_after_mixing = (1-coeff)*t1 + coeff*t2
    return t1_after_mixing, t2_after_mixing


def _compute_rock_types(t1, t2, t3, std_dev_proportion) -> (float, float, float):
    rt1, t2_after_first_mixing = _mix_adjacent_rock_types(t1, t2, std_dev_proportion)
    rt2, rt3 = _mix_adjacent_rock_types(t2_after_first_mixing, t3, std_dev_proportion)
    return rt1, rt2, rt3


def _get_weights() -> (float, float, float):
    how_many_types = random.randint(1, 3)

    if how_many_types == 1:
        return 1, 0, 0

    if how_many_types == 2:
        w1 = random.randint(1,100)
        return w1/100, 1-w1/100, 0

    assert how_many_types == 3

    w1 = random.randint(1, 98)
    w2 = random.randint(1, 100-w1)
    w3 = 100-w1-w2

    return w1/100, w2/100, w3/100


def run(nof_rows:int):
    print("t1\tt2\tt3\tw1\tw2\tw3\tdiff\taverage\tperfect_average\treal_average")
    for i in range(nof_rows):
        rock_types = [1,2,3,4]
        random.shuffle(rock_types)
        w1, w2, w3 = _get_weights()
        t1, t2, t3 = rock_types[:-1]
        rt1, rt2, rt3 = _compute_rock_types(t1, t2, t3, 0.1)
        perfect_rt1, perfect_rt2, perfect_rt3 = _compute_rock_types(t1, t2, t3, 0)
        average = w1*t1 + w2*t2 + w3*t3
        real_average = w1*rt1 + w2*rt2 + w3*rt3
        perfect_average = w1*perfect_rt1 + w2*perfect_rt2 + w3*perfect_rt3
        print(f"{t1}\t{t2}\t{t3}\t{w1}\t{w2}\t{w3}\t{real_average-average}\t{average}\t{perfect_average}\t{real_average}")


def main():
    parser = argparse.ArgumentParser(description='Generate data for learning corrected weighted average.',
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('number', nargs='?', type=int, default=1000, help='the number of examples')
    args = parser.parse_args()
    if args.number <= 0:
        print(f"the number of examples must be > 0: {args.number}")
        exit()
    run(args.number)


if __name__ == '__main__':
    main()
