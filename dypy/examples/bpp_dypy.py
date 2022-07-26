#!/usr/bin/env python3

import argparse
import math

import dypy as dp

import read_bpp


def solve(n, c, weights, time_limit=None):
    model = dp.Model()
    item = model.add_object_type(n)
    unpacked = model.add_set_var(item, [i for i in range(n)])
    residual = model.add_int_var(0)
    bin_number = model.add_element_resource_var(item, 0, less_is_better=True)
    weight_table = model.add_int_table(weights)
    lb2_weight1 = model.add_int_table(
        [1 if weights[i] > c / 2 else 0 for i in range(n)]
    )
    lb2_weight2 = model.add_float_table(
        [0.5 if weights[i] == c / 2 else 0 for i in range(n)]
    )
    lb3_weight = model.add_float_table(
        [
            1.0
            if weights[i] > c * 2 / 3
            else 2 / 3 // 0.001 / 1000
            if weights[i] == c * 2 / 3
            else 0.5
            if weights[i] > c / 3
            else 1 / 3 // 0.001 / 1000
            if weights[i] == c / 3
            else 0.0
            for i in range(n)
        ]
    )
    model.add_base_case([unpacked.is_empty()])

    for i in range(n):
        t = dp.Transition(
            name="pack {}".format(i),
            cost=dp.IntExpr.state_cost(),
            effects=[
                (unpacked, unpacked.remove(i)),
                (residual, residual - weight_table[i]),
            ],
            preconditions=[
                unpacked.contains(i),
                weight_table[i] <= residual,
                bin_number <= i + 1,
            ],
        )
        model.add_transition(t)

        ft = dp.Transition(
            name="open a new bin and pack {}".format(i),
            cost=dp.IntExpr.state_cost() + 1,
            preconditions=[
                bin_number <= i,
                unpacked.contains(i),
                weight_table[i] > residual,
            ]
            + [
                ~unpacked.contains(j) | (weight_table[j] > residual)
                for j in range(n)
                if i != j
            ],
            effects=[
                (unpacked, unpacked.remove(i)),
                (residual, c - weight_table[i]),
                (bin_number, bin_number + 1),
            ],
        )
        model.add_transition(ft, forced=True)

    model.add_dual_bound(math.ceil((weight_table[unpacked] - residual) / c))
    model.add_dual_bound(
        lb2_weight1[unpacked]
        + math.ceil(lb2_weight2[unpacked])
        - (residual >= c / 2).if_then_else(1, 0)
    )
    model.add_dual_bound(
        math.ceil(lb3_weight[unpacked]) - (residual >= c / 3).if_then_else(1, 0)
    )

    solver = dp.CAASDy(time_limit=time_limit)
    solution = solver.solve(model)

    for t in solution.transitions:
        print(t.name)

    print("expanded: {}".format(solution.expanded))
    print("cost: {}".format(solution.cost))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str)
    parser.add_argument("--time-out", default=1800, type=int)
    args = parser.parse_args()

    n, c, weights = read_bpp.read(args.input)
    solve(n, c, weights, time_limit=args.time_out)
