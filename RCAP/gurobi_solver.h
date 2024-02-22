#pragma once

template <typename cost_type, typename weight_type>
cost_type solve_with_gurobi(cost_type *costs, weight_type *weights, weight_type *budgets, uint N, uint K);