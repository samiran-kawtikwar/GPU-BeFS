#pragma once
// Solve RCAP with Gurobi

#include <gurobi_c++.h>
#include "host_logger.h"
#include "stdio.h"

template <typename cost_type, typename weight_type>
weight_type solve_with_gurobi(cost_type *costs, weight_type *weights, weight_type *budgets, uint N, uint K);