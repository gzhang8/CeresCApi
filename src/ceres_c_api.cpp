#include "ceres_c_api.h"

extern "C" {

   void ceres_init(){
      google::InitGoogleLogging("ceres");
   }

   void* ceres_create_problem() {
     return reinterpret_cast<void*>(new ceres::Problem);
   }


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
   double** parameters) {

      ceres::Problem* ceres_problem = reinterpret_cast<ceres::Problem*>(problem);

        ceres::CostFunction* callback_cost_function =
            new CostFunctionCxx(cost_function,
                                cost_function_data,
                                cost_function_data_size,
                                num_residuals,
                                num_parameter_blocks,
                                parameter_block_sizes);

        ceres::LossFunction* callback_loss_function = (ceres::LossFunction*)(
           loss_function
        );
        //std::cout << "lf ptr: " << callback_loss_function << std::endl;
        // if (loss_function != NULL) {
        //   callback_loss_function = new CallbackLossFunction(loss_function,
        //                                                     loss_function_data);
        // }

        std::vector<double*> parameter_blocks(parameters,
                                              parameters + num_parameter_blocks);
        return reinterpret_cast<void*>(
            ceres_problem->AddResidualBlock(callback_cost_function,
                                            callback_loss_function,
                                            parameter_blocks));


   }

   void ceres_problem_set_parameterization(void* problem,
                                           double* values,
                                           void* local_parameterization){
      ceres::Problem* ceres_problem = reinterpret_cast<ceres::Problem*>(problem);
      ceres::LocalParameterization* ceres_local_parameterization = reinterpret_cast<ceres::LocalParameterization*>(local_parameterization);
      ceres_problem->SetParameterization(values, ceres_local_parameterization);

   }

   void* ceres_create_quaternion_parameterization(){
      ceres::QuaternionParameterization* quat_paramztin = new ceres::QuaternionParameterization();
      void* ret = reinterpret_cast<void*>(quat_paramztin);
      return ret;
   }

   // constant

   void ceres_set_parameter_block_variable(void* c_problem, double *values) {
      ceres::Problem* problem = reinterpret_cast<ceres::Problem*>(c_problem);
      problem->SetParameterBlockVariable(values);
   }

   void ceres_set_parameter_block_constant(void* problem, double* data){
      ceres::Problem* ceres_problem = reinterpret_cast<ceres::Problem*>(problem);
      ceres_problem->SetParameterBlockConstant(data);
   }




   void ceres_solve(void* c_problem, int max_iter_num, int solver_type) {
     ceres::Problem* problem = reinterpret_cast<ceres::Problem*>(c_problem);

     ceres::Solver::Options options;
     options.max_num_iterations = max_iter_num;
     switch (solver_type) {
        case 1:
            options.linear_solver_type = ceres::DENSE_QR;
            break;
        case 2:
            options.linear_solver_type = ceres::DENSE_SCHUR;
            break;
        case 3:
            options.linear_solver_type = ceres::SPARSE_SCHUR;
            break;
         case 4:
             options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
             break;

     }

     options.minimizer_progress_to_stdout = true;

     ceres::Solver::Summary summary;
     ceres::Solve(options, problem, &summary);
     std::cout << summary.FullReport() << "\n";
   }


   // loss functions
   // https://en.wikipedia.org/wiki/Huber_loss
   void* ceres_create_huber_loss(double delta){
      ceres::HuberLoss* hloss = new ceres::HuberLoss(delta);
      return (void*)hloss;
   }

   // bounds

   void ceres_SetParameterLowerBound(void* c_problem, double *values, int index, double lower_bound) {
       ceres::Problem* problem = reinterpret_cast<ceres::Problem*>(c_problem);
       problem->SetParameterLowerBound(values, index, lower_bound);
   }


    void ceres_SetParameterUpperBound(void* c_problem, double *values, int index, double upper_bound) {
        ceres::Problem* problem = reinterpret_cast<ceres::Problem*>(c_problem);
        problem->SetParameterUpperBound(values, index, upper_bound);
    }


}
