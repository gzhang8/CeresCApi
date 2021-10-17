#ifndef CERES_C_API_H
#define CERES_C_API_H

#include <ceres/ceres.h>
#include <glog/logging.h>
extern "C" {


void ceres_init();

/* Create and destroy a problem */
void* ceres_create_problem();

typedef int (*cost_function_cb)(double* user_data,
                                double** parameters,
                                double* residuals,
                                double** jacobians,
                                int user_data_size,
                                int num_residuals,
                                int num_parameter_block,
                                int* parameter_block_sizes);
// currently loss_function_data is not used, we are using wrapped ceres c++ loss
// for  loss_function
void* ceres_problem_add_residual_block(
   void* problem,
   void* cost_function,
   double* cost_function_data,
   int cost_function_data_size,
   void* loss_function,
   double* loss_function_data,
   int num_residuals,
   int num_parameter_blocks,
   int* parameter_block_sizes,
   double** parameters);

// Set the local parameterization for one of the parameter blocks.
// The local_parameterization is owned by the Problem by default. It
// is acceptable to set the same parameterization for multiple
// parameters; the destructor is careful to delete local
// parameterizations only once. The local parameterization can only
// be set once per parameter, and cannot be changed once set.
// void SetParameterization(double* values,
//                          LocalParameterization* local_parameterization);
void ceres_problem_set_parameterization(void* problem,
                                        double* values,
                                        void* local_parameterization);
// quaternion is stored by w, i, j, k
void* ceres_create_quaternion_parameterization();

void ceres_set_parameter_block_constant(void* problem, double* data);

void ceres_solve(void* c_problem, int max_iter_num, int solver_type);

}
// This cost function wraps a C-level function pointer from the user, to bridge
// between C and C++.
class CostFunctionCxx : public ceres::CostFunction {
public:
 CostFunctionCxx(void* cost_function,
                 double* user_data,
                 int user_data_size,
                 int num_residuals,
                 int num_parameter_block,
                 int* parameter_block_sizes)
     : user_data_size_(user_data_size),
       num_residuals_(num_residuals),
       num_parameter_block_(num_parameter_block),
       parameter_block_sizes_(parameter_block_sizes),
       cost_function_(reinterpret_cast<cost_function_cb>(cost_function)),
       user_data_(user_data) {
   set_num_residuals(num_residuals);
   for(int i = 0; i < num_parameter_block; i ++) {
     mutable_parameter_block_sizes()->push_back(parameter_block_sizes[i]);
   }
 }

 virtual ~CostFunctionCxx() {}

 virtual bool Evaluate(double const* const* parameters,
                       double* residuals,
                       double** jacobians) const {

   // (double* user_data,
   //  double** parameters,
   //  double* residuals,
   //  double** jacobians,
   //  int user_data_size,
   //  int num_residuals,
   //  int num_parameter_block,
   //  int* parameter_block_sizes)

   bool res = (*cost_function_)(user_data_,
                                const_cast<double**>(parameters),
                                residuals,
                                jacobians,
                                user_data_size_,
                                num_residuals_,
                                num_parameter_block_,
                                parameter_block_sizes_);
   return res;
 }

private:
  int user_data_size_;
  int num_residuals_;
  int num_parameter_block_;
  int* parameter_block_sizes_;
  cost_function_cb cost_function_;
  double* user_data_;
};


void AddResidualBlockCxx(ceres::Problem* problem,
                         CostFunctionCxx* costfunction,
                         double** params,
                         int* parameter_block_sizes,
                         int num_parameter_block){

    std::vector<double*> parameter_blocks(params, params + num_parameter_block);


    // TODO check each params size
    problem->AddResidualBlock(
        costfunction,
        NULL,
        parameter_blocks);
}

#endif
