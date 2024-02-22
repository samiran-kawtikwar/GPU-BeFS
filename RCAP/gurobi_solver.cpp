// Solve RCAP with Gurobi

#include <gurobi_c++.h>
#include "host_logger.h"
#include "stdio.h"
#include "gurobi_solver.h"
#include <sstream>

using namespace host_log;

template <typename cost_type, typename weight_type>
cost_type solve_with_gurobi(cost_type *costs, weight_type *weights, weight_type *budgets, uint N, uint K)
{
  try
  {
    GRBEnv env = GRBEnv();
    env.set(GRB_IntParam_OutputFlag, 0);
    GRBModel model = GRBModel(env);

    // Create variables
    GRBVar *x = new GRBVar[N * N];
    for (uint i = 0; i < N; i++)
    {
      for (uint j = 0; j < N; j++)
      {
        std::stringstream s;
        s << "X_" << i << "_" << j << std::endl;
        x[i * N + j] = model.addVar(0, 1, costs[i * N + j], GRB_BINARY, s.str());
      }
    }
    model.update();

    // Add assignment constraints
    for (uint i = 0; i < N; i++)
    {
      GRBLinExpr expr1 = 0, expr2 = 0;
      for (uint j = 0; j < N; j++)
      {
        expr1 += x[i * N + j];
        expr2 = expr2 + x[j * N + i];
      }
      model.addConstr(expr1 == 1);
      model.addConstr(expr2 == 1);
    }
    model.update();
    model.optimize();
    cost_type LAP_obj = (cost_type)model.getObjective().getValue();
    cost_type UB = LAP_obj;
    Log(info, "LAP solved with objective: %u", LAP_obj);

    // Make sure budget constraints are active
    do
    {
      Log(debug, "Trying");
      model.reset();
      // Add budget constraints
      for (uint k = 0; k < K; k++)
      {
        GRBLinExpr expr = 0;
        for (uint i = 0; i < N * N; i++)
        {
          expr += weights[k * N * N + i] * x[i];
        }
        model.addConstr(expr <= budgets[k]);
      }
      model.update();
      model.optimize();
      UB = (cost_type)model.getObjective().getValue();
      if (UB <= LAP_obj)
      {
        // Reduce the budgets
        Log(debug, "Reducing budgets");
        for (uint k = 0; k < K; k++)
          budgets[k] = std::max((weight_type)1, budgets[k] - 1);
      }
    } while (UB <= LAP_obj);
    model.write("scratch/model.lp");
    Log(info, "RCAP solved with objective: %u", UB);

    delete[] x;
    return UB;
  }
  catch (GRBException e)
  {
    Log(error, "Error code = %d", e.getErrorCode());
    std::cout << e.getErrorCode() << "\n\n"
              << e.getMessage() << std::endl;
    exit(-1);
  }
  catch (...)
  {
    Log(error, "Exception during optimization");
  }
  return 0;
}

// Explicit instantiations
template uint solve_with_gurobi<uint, uint>(uint *, uint *, uint *, uint, uint);
template uint solve_with_gurobi<uint, float>(uint *, float *, float *, uint, uint);
template uint solve_with_gurobi<uint, double>(uint *, double *, double *, uint, uint);
template float solve_with_gurobi<float, uint>(float *, uint *, uint *, uint, uint);
template float solve_with_gurobi<float, float>(float *, float *, float *, uint, uint);
template float solve_with_gurobi<float, double>(float *, double *, double *, uint, uint);
template double solve_with_gurobi<double, uint>(double *, uint *, uint *, uint, uint);
template double solve_with_gurobi<double, float>(double *, float *, float *, uint, uint);
template double solve_with_gurobi<double, double>(double *, double *, double *, uint, uint);