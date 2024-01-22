#include "gurobi_solver.h"

using namespace host_log;

template <typename cost_type, typename weight_type>
weight_type solve_with_gurobi(cost_type *costs, weight_type *weights, weight_type *budgets, uint N, uint K)
{
  try
  {
    GRBEnv env = GRBEnv(true);
    Log(info, "Gurobi success!!");
    env.set(GRB_IntParam_OutputFlag, 1);
    Log(info, "Gurobi success!!");
    GRBModel model = GRBModel(env);
    Log(info, "Gurobi success!!");

    // Create variables
    GRBVar *x = model.addVars(N * N, GRB_BINARY);
  }
  catch (GRBException e)
  {
    Log(error, "Error code = %d", e.getErrorCode());
    std::cout << e.getErrorCode() << "\n\n"
              << /*e.getMessage() <<*/ std::endl;
    // std::cout << e.getErrorCode() << std::endl;
  }
  catch (...)
  {
    Log(error, "Exception during optimization");
  }
  return 0;
}

template uint solve_with_gurobi<uint, uint>(uint *, uint *, uint *, uint, uint);
