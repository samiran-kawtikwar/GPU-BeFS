// Solve QAP with Gurobi

#include <gurobi_c++.h>
#include "../utils/logger.cuh"
#include "stdio.h"
#include "gurobi_solver.h"
#include <sstream>

template <typename cost_type>
cost_type solve_with_gurobi(cost_type *distances, cost_type *flows, uint N)
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
      for (uint p = 0; p < N; p++)
      {
        std::stringstream s;
        s << "X_" << i << "_" << p << std::endl;
        x[i * N + p] = model.addVar(0, 1, 0, GRB_BINARY, s.str());
      }
    }
    model.update();
    // Add assignment constraints
    for (uint i = 0; i < N; i++)
    {
      GRBLinExpr expr1 = 0, expr2 = 0;
      for (uint p = 0; p < N; p++)
      {
        expr1 += x[i * N + p];
        expr2 += x[p * N + i];
      }
      model.addConstr(expr1 == 1);
      model.addConstr(expr2 == 1);
    }
    model.update();

    // Add QAP objective function and optimize (return objective value)
    // use i, j for facilities and p, q for locations
    GRBQuadExpr objective = 0;
    for (uint i = 0; i < N; i++)
    {
      for (uint j = 0; j < N; j++)
      {
        for (uint p = 0; p < N; p++)
        {
          for (uint q = 0; q < N; q++)
          {
            cost_type f = flows[i * N + j];
            cost_type d = distances[p * N + q];
            objective += f * d * x[i * N + p] * x[j * N + q];
          }
        }
      }
    }

    model.setObjective(objective, GRB_MINIMIZE);
    model.optimize();

    cost_type objVal = static_cast<cost_type>(model.get(GRB_DoubleAttr_ObjVal));

    delete[] x;
    return objVal;
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

// Explicit instantiation
template int solve_with_gurobi<int>(int *distances, int *flows, uint N);
template uint solve_with_gurobi<uint>(uint *distances, uint *flows, uint N);
template double solve_with_gurobi<double>(double *distances, double *flows, uint N);
template float solve_with_gurobi<float>(float *distances, float *flows, uint N);