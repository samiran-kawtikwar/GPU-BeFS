#pragma once

template <typename cost_type>
cost_type solve_with_gurobi(cost_type *distances, cost_type *flows, uint N);