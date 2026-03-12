/*
 * Low-Order finite element preconditioner computed with HYPRE's AMG solver
*/

#include <math.h>
#include "gslib.h"

#define fem_amg_setup FORTRAN_UNPREFIXED(fem_amg_setup, FEM_AMG_SETUP)
#define fem_amg_solve FORTRAN_UNPREFIXED(fem_amg_solve, FEM_AMG_SOLVE)

#ifdef HYPRE

/* Headers */
#include "_hypre_utilities.h"
#include "HYPRE_parcsr_ls.h"
#include "_hypre_parcsr_ls.h"
#include "HYPRE.h"

#include "fem_amg_preco.h"

/* Global variables */
int precond_type;
int meshing_type;
int n_x, n_y, n_z, n_elem, n_dim;
int n_xyz, n_xyze;
double *x_m, *y_m, *z_m;
long long *glo_num;
double *pmask;
double *binv;
int num_loc_dofs;
long long *dof_map;
int amg_ready = 0;
long long row_start;
long long row_end;
HYPRE_IJMatrix A_bc;
HYPRE_ParCSRMatrix A_fem;
HYPRE_IJMatrix B_bc;
HYPRE_ParCSRMatrix B_fem;
HYPRE_IJVector Bd_bc;
HYPRE_ParVector Bd_fem;
HYPRE_IJVector f_bc;
HYPRE_ParVector f_fem;
HYPRE_IJVector u_bc;
HYPRE_ParVector u_fem;
HYPRE_IJVector Bf_bc;
HYPRE_ParVector Bf_fem;
HYPRE_IJVector Binv_sem_bc;
HYPRE_ParVector Binv_sem;
HYPRE_Solver amg_preconditioner;

struct comm comm;
struct gs_data *gsh;

#define NPARAM 10
double HYPREsettings[NPARAM]; 

/* Interface definition */
void fem_amg_setup(const sint *n_x_, const sint *n_y_, const sint *n_z_, 
                   const sint *n_elem_, const sint *n_dim_, 
                   double *x_m_, double *y_m_, double *z_m_, 
                   double *pmask_, double *binv_, const sint *nullspace,
                   const sint *gshf, double *param)
{
    precond_type = 1;
    meshing_type = 0;
    n_x = *n_x_;
    n_y = *n_y_;
    n_z = *n_z_;
    n_elem = *n_elem_;
    n_dim = *n_dim_;
    x_m = x_m_;
    y_m = y_m_;
    z_m = z_m_;
    pmask = pmask_;
    binv = binv_;

    n_xyz = n_x * n_y * n_z;
    n_xyze = n_x * n_y * n_z * n_elem;

    gsh = gs_hf2c(*gshf);
    comm_init(&comm, gsh->comm.c);

    if (comm.id == 0) printf("fem_amg_setup ...\n");

    if (sizeof(HYPRE_Int) != sizeof(long long)) {
        if (comm.id == 0)
          fail(1,__FILE__,__LINE__,"incompatible long long size");
        exit(EXIT_FAILURE);
    }

    double time0 = comm_time();
    matrix_distribution();
    fem_assembly();

    long long row;
    long long row_start = hypre_ParCSRMatrixFirstRowIndex(A_fem);
    long long row_end = hypre_ParCSRMatrixLastRowIndex(A_fem);

    HYPRE_IJVectorCreate(comm.c, row_start, row_end, &u_bc);
    HYPRE_IJVectorSetObjectType(u_bc, HYPRE_PARCSR);
    HYPRE_IJVectorInitialize(u_bc);
    HYPRE_IJVectorAssemble(u_bc);
    HYPRE_IJVectorGetObject(u_bc, (void**) &u_fem);

    HYPRE_IJVectorCreate(comm.c, row_start, row_end, &f_bc);
    HYPRE_IJVectorSetObjectType(f_bc, HYPRE_PARCSR);
    HYPRE_IJVectorInitialize(f_bc);
    HYPRE_IJVectorAssemble(f_bc);
    HYPRE_IJVectorGetObject(f_bc, (void**) &f_fem);

    HYPRE_IJVectorCreate(comm.c, row_start, row_end, &Bf_bc);
    HYPRE_IJVectorSetObjectType(Bf_bc, HYPRE_PARCSR);
    HYPRE_IJVectorInitialize(Bf_bc);
    HYPRE_IJVectorAssemble(Bf_bc);
    HYPRE_IJVectorGetObject(Bf_bc, (void**) &Bf_fem);

    HYPRE_IJVectorCreate(comm.c, row_start, row_end, &Binv_sem_bc);
    HYPRE_IJVectorSetObjectType(Binv_sem_bc, HYPRE_PARCSR);
    HYPRE_IJVectorInitialize(Binv_sem_bc);

    for (row = row_start; row <= row_end; row++)
        HYPRE_IJVectorSetValues(Binv_sem_bc, 1, &row, &(binv[dof_map[row - row_start]]));

    HYPRE_IJVectorAssemble(Binv_sem_bc);
    HYPRE_IJVectorGetObject(Binv_sem_bc, (void**) &Binv_sem);

    HYPRE_BoomerAMGCreate(&amg_preconditioner);

    int uparam = (int) param[0];

    int i;
    if (uparam) {
       for (i=1; i < NPARAM; i++) { 
           HYPREsettings[i-1] = param[i];
           if (comm.id == 0) 
              printf("Custom HYPREsettings[%d]: %.2f\n", i, HYPREsettings[i-1]); 
       }
    } else {
      HYPREsettings[0] = 10;    /* HMIS                       */
      HYPREsettings[1] = 6;     /* Extended+i                 */
      HYPREsettings[2] = 3;     /* SOR is default smoother    */
      HYPREsettings[3] = 3;     /* SOR smoother for crs level */
      HYPREsettings[4] = 1;
      HYPREsettings[5] = 0.25;
      HYPREsettings[6] = 0.1;
    }

    HYPRE_BoomerAMGSetCoarsenType(amg_preconditioner, HYPREsettings[0]); 
    HYPRE_BoomerAMGSetInterpType(amg_preconditioner, HYPREsettings[1]);   
    if (*nullspace == 1 || (int) param[0] != 0) {
       HYPRE_BoomerAMGSetRelaxType(amg_preconditioner, HYPREsettings[2]);
       HYPRE_BoomerAMGSetCycleRelaxType(amg_preconditioner, HYPREsettings[3], 3); /* Coarse grid solver */
       HYPRE_BoomerAMGSetCycleNumSweeps(amg_preconditioner, HYPREsettings[4], 3); /* Number of sweeps at coarse level */
    }
    HYPRE_BoomerAMGSetStrongThreshold(amg_preconditioner, HYPREsettings[5]);
    HYPRE_BoomerAMGSetNonGalerkinTol(amg_preconditioner, HYPREsettings[6]);
    if (uparam) {
    HYPRE_BoomerAMGSetLevelNonGalerkinTol(amg_preconditioner, HYPREsettings[7], 0);
    HYPRE_BoomerAMGSetLevelNonGalerkinTol(amg_preconditioner, HYPREsettings[8], 1);
    
    fflush(stdout);HYPRE_BoomerAMGSetLevelNonGalerkinTol(amg_preconditioner, HYPREsettings[9], 2);
    }

    HYPRE_BoomerAMGSetMaxRowSum(amg_preconditioner, 1); /* Don't check for maximum row sum */
    HYPRE_BoomerAMGSetPMaxElmts(amg_preconditioner, 4);
    HYPRE_BoomerAMGSetMaxCoarseSize(amg_preconditioner, 50);
    HYPRE_BoomerAMGSetMaxIter(amg_preconditioner,1);
    HYPRE_BoomerAMGSetTol(amg_preconditioner, 0); 
    HYPRE_BoomerAMGSetPrintLevel(amg_preconditioner, 1);

    HYPRE_BoomerAMGSetup(amg_preconditioner, A_fem, NULL, NULL);

    double time1 = comm_time();
    if (comm.id == 0) printf("fem_amg_setup: done %fs\n",time1-time0);
    fflush(stdout);
    amg_ready = 1;
}

void fem_amg_solve(double *z, double *w)
{
    long long row;
    int idx;

    if (amg_ready == 0)
    {
        if (comm.id == 0)
          fail(1,__FILE__,__LINE__,"AMG hasn't been setup");
        exit(EXIT_FAILURE);
    }

    /* Choose preconditioner type */
    HYPRE_IJVectorInitialize(f_bc);

    switch (precond_type)
    {
        /* No mass operator */
        case 1:
            for (row = row_start; row <= row_end; row++)
                HYPRE_IJVectorSetValues(f_bc, 1, &row, &(w[dof_map[row - row_start]]));

            break;

        /* Diagonal mass operator */
        case 2:
            for (row = row_start; row <= row_end; row++)
            {
                double Bd_fem_value;
                double Binv_sem_value;

                HYPRE_IJVectorGetValues(Bd_bc, 1, &row, &Bd_fem_value);
                HYPRE_IJVectorGetValues(Binv_sem_bc, 1, &row, &Binv_sem_value);

                double value = Bd_fem_value * Binv_sem_value * w[dof_map[row - row_start]];

                HYPRE_IJVectorSetValues(f_bc, 1, &row, &value);
            }

            break;

        /* Full mass operator */
        case 3:
            for (row = row_start; row <= row_end; row++)
            {
                double Binv_sem_value;

                HYPRE_IJVectorGetValues(Binv_sem_bc, 1, &row, &Binv_sem_value);

                double value = Binv_sem_value * w[dof_map[row - row_start]];

                HYPRE_IJVectorSetValues(Bf_bc, 1, &row, &value);
            }

            HYPRE_ParCSRMatrixMatvec(1.0, B_fem, Bf_fem, 0.0, f_fem);

            break;
    }

    HYPRE_IJVectorAssemble(f_bc);

    /* Solve preconditioned system */
    HYPRE_BoomerAMGSolve(amg_preconditioner, A_fem, f_fem, u_fem);

    /* Map data back to SEM mesh */
    for (idx = 0; idx < n_xyze; idx++)
        z[idx] = 0.0;

    for (row = row_start; row <= row_end; row++)
        HYPRE_IJVectorGetValues(u_bc, 1, &row, &(z[dof_map[row - row_start]]));

    gs(z, gs_double, gs_add, 0, gsh, 0);
}

/* FEM Assembly definition */
void matrix_distribution()
{
    /*
     * Ranks the global numbering array after removing the Dirichlet nodes
     * which is then used in the assembly of the matrices to map degrees of 
     * freedom to rows of the matrix
     */

    int idx;
    buffer my_buffer;
    long long idx_start = n_xyze;
    long long scan_out[2], scan_buf[2];
    comm_scan(scan_out, &comm, gs_long_long, gs_add, &idx_start, 1, scan_buf);
    idx_start = scan_out[0];

    glo_num = mem_alloc_1D_long(n_xyze);

    for (idx = 0; idx < n_xyze; idx++)
    {
        if (pmask[idx] > 0.0)
            glo_num[idx] = idx_start + (long long)idx;
        else
            glo_num[idx] = -1;
    }

    gs(glo_num, gs_long_long, gs_min, 0, gsh, 0);

    /* Rank ids */
    long long maximum_value_local = 0;
    long long maximum_value = 0;

    for (idx = 0; idx < n_xyze; idx++)
    {
        maximum_value_local = (glo_num[idx] > maximum_value_local) ? glo_num[idx] : maximum_value_local;
    }

    comm_allreduce(&comm, gs_long_long, gs_max, &maximum_value_local, 1, &maximum_value);
    const long long nstar = maximum_value/comm.np + 1;

    struct ranking_tuple
    {
        long long rank;
        unsigned int proc;
        unsigned int idx;
    };

    struct array ranking_transfer;
    array_init(struct ranking_tuple, &ranking_transfer, n_xyze);
    ranking_transfer.n = n_xyze;
    struct ranking_tuple *ranking_tuple_array = ranking_transfer.ptr;

    for (idx = 0; idx < ranking_transfer.n; idx++)
    {
       ranking_tuple_array[idx].rank = glo_num[idx];
       ranking_tuple_array[idx].proc = glo_num[idx] / nstar;
       ranking_tuple_array[idx].idx = idx;
    }

    struct crystal crystal_router_handle;
    crystal_init(&crystal_router_handle, &comm);
    sarray_transfer(struct ranking_tuple, &ranking_transfer, proc, 1, &crystal_router_handle);
    ranking_tuple_array = ranking_transfer.ptr;

    buffer_init(&my_buffer, 1);
    sarray_sort(struct ranking_tuple, ranking_transfer.ptr, ranking_transfer.n, rank, 1, &my_buffer);

    long long current_rank = ranking_tuple_array[0].rank;
    long long current_count = 0;
    ranking_tuple_array[0].rank = current_count;

    for (idx = 1; idx < ranking_transfer.n; idx++) {

        if (ranking_tuple_array[idx].rank > current_rank) {
            current_count++;
            current_rank = ranking_tuple_array[idx].rank;
            ranking_tuple_array[idx].rank = current_count;
        } else if (ranking_tuple_array[idx].rank == current_rank) {
            ranking_tuple_array[idx].rank = current_count;
        } else {
            break;
        }
    }

    current_count += 1;

    long long rank_start;
    comm_scan(scan_out, &comm, gs_long_long, gs_add, &current_count, 1, scan_buf);
    rank_start = scan_out[0];

    for (idx = 0; idx < ranking_transfer.n; idx++)
    {
        ranking_tuple_array[idx].rank += rank_start;
    }

    sarray_transfer(struct ranking_tuple, &ranking_transfer, proc, 1, &crystal_router_handle);
    ranking_tuple_array = ranking_transfer.ptr;

    buffer_init(&my_buffer, 1);
    sarray_sort(struct ranking_tuple, ranking_transfer.ptr, ranking_transfer.n, idx, 0, &my_buffer);

    for (idx = 0; idx < n_xyze; idx++)
    {
        glo_num[idx] = ranking_tuple_array[idx].rank;
    }

    array_free(&ranking_transfer);
    crystal_free(&crystal_router_handle);
}

void fem_assembly()
{
    /*
     * Assembles the low-order FEM matrices from the spectral element mesh
     *
     * Returns A_fem and B_fem
     */

    /* Variables */
    int i, j, k, e, d, t, q;
    int idx;
    long long row;

    /*
     * Rank and prepare data to be mapped to rows of the matrix so it can 
     * be solved with Hypre (Ranking done on the Fortran side since here it fails
     * for some reason I couldn't figure out)
     */
    long long *ranking = mem_alloc_1D_long(n_xyze);

    for (idx = 0; idx < n_xyze; idx++)
        ranking[idx] = glo_num[idx];

    row_start = 0;
    row_end = 0;

    for (idx = 0; idx < n_xyze; idx++)
        if (ranking[idx] >= 0) row_end = maximum(row_end, ranking[idx]);

    long long scan_out[2], scan_buf[2];
    comm_scan(scan_out, &comm, gs_long_long, gs_max, &row_end, 1, scan_buf);
    if (comm.id > 0) row_start = scan_out[0] + 1;

    num_loc_dofs = row_end - row_start + 1;

    dof_map = mem_alloc_1D_long(num_loc_dofs);

    for (idx = 0; idx < n_xyze; idx++)
    {
        if ((row_start <= ranking[idx]) && (ranking[idx] <= row_end))
        {
            dof_map[ranking[idx] - row_start] = idx;
        }
    }

    /* Assemble FE matrices with boundary conditions applied */
    HYPRE_IJMatrixCreate(comm.c, row_start, row_end, row_start, row_end, &A_bc);
    HYPRE_IJMatrixSetObjectType(A_bc, HYPRE_PARCSR);
    HYPRE_IJMatrixInitialize(A_bc);

    HYPRE_IJMatrixCreate(comm.c, row_start, row_end, row_start, row_end, &B_bc);
    HYPRE_IJMatrixSetObjectType(B_bc, HYPRE_PARCSR);
    HYPRE_IJMatrixInitialize(B_bc);

    HYPRE_IJVectorCreate(comm.c, row_start, row_end, &Bd_bc);
    HYPRE_IJVectorSetObjectType(Bd_bc, HYPRE_PARCSR);
    HYPRE_IJVectorInitialize(Bd_bc);

    double *Bd_sum = mem_alloc_1D_double(n_xyze);
    for (i = 0; i < n_xyze; i++) Bd_sum[i] = 0.0;

    /* Set quadrature rule */
    int n_quad = (n_dim == 2) ? 3 : 4;
    double **q_r;
    double *q_w;

    quadrature_rule(&q_r, &q_w, n_quad, n_dim);

    /* Mesh connectivity (Can be changed to fill-out or one-per-vertex) */
    int num_fem;
    int **v_coord;
    int **t_map;

    if (n_dim == 2)
        num_fem = (meshing_type == 0) ? 4 : 2;
    else
        num_fem = (meshing_type == 0) ? 8 : 6;

    mesh_connectivity(&v_coord, &t_map, num_fem, n_dim);

    /* Finite element assembly */
    double **A_loc = mem_alloc_2D_double(n_dim + 1, n_dim + 1);
    double **B_loc = mem_alloc_2D_double(n_dim + 1, n_dim + 1);
    double **J_xr = mem_alloc_2D_double(n_dim, n_dim);
    double **J_rx = mem_alloc_2D_double(n_dim, n_dim);
    double **x_t = mem_alloc_2D_double(n_dim, n_dim + 1);
    double *q_x = mem_alloc_1D_double(n_dim);
    double *dp = mem_alloc_1D_double(n_dim);

    Basis phi[n_dim + 1];
    DBasis dphi[n_dim + 1];

    if (n_dim == 2)
    {
        phi[0] = phi_2D_1; phi[1] = phi_2D_2; phi[2] = phi_2D_3;
        dphi[0] = dphi_2D_1; dphi[1] = dphi_2D_2; dphi[2] = dphi_2D_3;
    }
    else
    {
        phi[0] = phi_3D_1; phi[1] = phi_3D_2; phi[2] = phi_3D_3; phi[3] = phi_3D_4;
        dphi[0] = dphi_3D_1; dphi[1] = dphi_3D_2; dphi[2] = dphi_3D_3; dphi[3] = dphi_3D_4;
    }

    int s_x, s_y, s_z;
    int E_x = n_x - 1;
    int E_y = n_y - 1;
    int E_z = (n_dim == 2) ? 1 : n_z - 1;

    for (e = 0; e < n_elem; e++)
    {
        /* Cycle through collocated quads/hexes */
        for (s_z = 0; s_z < E_z; s_z++)
        {
            for (s_y = 0; s_y < E_y; s_y++)
            {
                for (s_x = 0; s_x < E_x; s_x++)
                {
                    /* Get indices */
                    int s[n_dim];

                    if (n_dim == 2)
                    {
                        s[0] = s_x; s[1] = s_y;
                    }
                    else
                    {
                        s[0] = s_x; s[1] = s_y; s[2] = s_z;
                    }

                    int idx[(int)(pow(2, n_dim))];

                    for (i = 0; i < pow(2, n_dim); i++)
                    {
                        idx[i] = 0;

                        for (d = 0; d < n_dim; d++)
                        {
                            idx[i] += (s[d] + v_coord[i][d]) * pow(n_x, d);
                        }
                    }

                    /* Cycle through collocated triangles/tets */
                    for (t = 0; t < num_fem; t++)
                    {
                        /* Get vertices */
                        for (i = 0; i < n_dim + 1; i++)
                        {
                            if (n_dim == 2)
                            {
                                x_t[0][i] = x_m[idx[t_map[t][i]] + e * n_xyz];
                                x_t[1][i] = y_m[idx[t_map[t][i]] + e * n_xyz];
                            }
                            else
                            {
                                x_t[0][i] = x_m[idx[t_map[t][i]] + e * n_xyz];
                                x_t[1][i] = y_m[idx[t_map[t][i]] + e * n_xyz];
                                x_t[2][i] = z_m[idx[t_map[t][i]] + e * n_xyz];
                            }
                        }

                        /* Local FEM matrices */
                        /* Reset local stiffness and mass matrices */
                        for (i = 0; i < n_dim + 1; i++)
                        {
                            for (j = 0; j < n_dim + 1; j++)
                            {
                                A_loc[i][j] = 0.0;
                                B_loc[i][j] = 0.0;
                            }
                        }

                        /* Build local stiffness matrices by applying quadrature rules */
                        for (q = 0; q < n_quad; q++)
                        {
                            /* From r to x */
                            x_map(&q_x, q_r[q], x_t, n_dim, phi);
                            J_xr_map(&J_xr, q_r[q], x_t, n_dim, dphi);
                            inverse(&J_rx, J_xr, n_dim);
                            double det_J_xr = determinant(J_xr, n_dim);

                            /* Integrand */
                            for (i = 0; i < n_dim + 1; i++)
                            {
                                for (j = 0; j < n_dim + 1; j++)
                                {
                                    int alpha, beta;
                                    double func = 0.0;

                                    for (alpha = 0; alpha < n_dim; alpha++)
                                    {
                                        double a = 0.0, b = 0.0;

                                        for (beta = 0; beta < n_dim; beta++)
                                        {
                                            dphi[i](&dp, q_r[q]);
                                            a += dp[beta] * J_rx[beta][alpha];

                                            dphi[j](&dp, q_r[q]);
                                            b += dp[beta] * J_rx[beta][alpha];
                                        }

                                        func += a * b;
                                    }

                                    A_loc[i][j] += func * det_J_xr * q_w[q];
                                    B_loc[i][j] += phi[i](q_r[q]) * phi[j](q_r[q]) * det_J_xr * q_w[q];
                                }
                            }
                        }

                        /* Add to global matrix */
                        for (i = 0; i < n_dim + 1; i++)
                        {
                            for (j = 0; j < n_dim + 1; j++)
                            {
                                if ((pmask[idx[t_map[t][i]] + e * n_xyz] > 0.0) && (pmask[idx[t_map[t][j]] + e * n_xyz] > 0.0))
                                {
                                    long long row = ranking[idx[t_map[t][i]] + e * n_xyz];
                                    long long col = ranking[idx[t_map[t][j]] + e * n_xyz];

                                    double A_val = A_loc[i][j];
                                    double B_val = B_loc[i][j];

                                    long long ncols = 1;
                                    int insert_error;

                                    if (fabs(A_val) > 1.0e-14)
                                    {
                                        insert_error = HYPRE_IJMatrixAddToValues(A_bc, 1, &ncols, &row, &col, &A_val);
                                    }

                                    if (fabs(B_val) > 1.0e-14)
                                    {
                                        insert_error = HYPRE_IJMatrixAddToValues(B_bc, 1, &ncols, &row, &col, &B_val);
                                    }

                                    if (insert_error != 0)
                                    {
                                        if (comm.id == 0)
                                            printf("There was an error with entry A(%lld, %lld) = %f or B(%lld, %lld) = %f\n", row, col, A_val, row, col, B_val);

                                        exit(EXIT_FAILURE);
                                    }
                                }

                                int row = idx[t_map[t][i]] + e * n_xyz;
                                int col = idx[t_map[t][j]] + e * n_xyz;
                                Bd_sum[row] += B_loc[i][j];
                                Bd_sum[col] += B_loc[i][j];
                            }
                        }
                    }
                }
            }
        }
    }

    for (idx = 0; idx < n_xyze; idx++)
        Bd_sum[idx] /= 2.0;

    gs(Bd_sum, gs_double, gs_add, 0, gsh, 0);

    for (row = row_start; row <= row_end; row++)
        HYPRE_IJVectorSetValues(Bd_bc, 1, &row, &(Bd_sum[dof_map[row - row_start]]));

    HYPRE_IJMatrixAssemble(A_bc);
    HYPRE_IJMatrixGetObject(A_bc, (void**) &A_fem);

    HYPRE_IJMatrixAssemble(B_bc);
    HYPRE_IJMatrixGetObject(B_bc, (void**) &B_fem);

    HYPRE_IJVectorAssemble(Bd_bc);
    HYPRE_IJVectorGetObject(Bd_bc, (void**) &Bd_fem);

    /* Free memory */
    mem_free_1D_long(&ranking, n_xyze);
    mem_free_1D_long(&glo_num, n_xyze);
    mem_free_1D_double(&Bd_sum, n_xyze);
    mem_free_2D_double(&q_r, n_quad, n_dim);
    mem_free_1D_double(&q_w, n_quad);
    mem_free_2D_int(&v_coord, pow(n_dim, 2), n_dim);
    mem_free_2D_int(&t_map, num_fem, n_dim + 1);
    mem_free_2D_double(&A_loc, n_dim + 1, n_dim + 1);
    mem_free_2D_double(&B_loc, n_dim + 1, n_dim + 1);
    mem_free_2D_double(&J_xr, n_dim, n_dim);
    mem_free_2D_double(&J_rx, n_dim, n_dim);
    mem_free_2D_double(&x_t, n_dim, n_dim + 1);
    mem_free_1D_double(&q_x, n_dim);
    mem_free_1D_double(&dp, n_dim);
}

void quadrature_rule(double ***q_r, double **q_w, int n_quad, int n_dim)
{
    (*q_r) = mem_alloc_2D_double(n_quad, n_dim);
    (*q_w) = mem_alloc_1D_double(n_quad);

    if (n_dim == 2)
    {
        if (n_quad == 3)
        {
            (*q_r)[0][0] = 1.0 / 6.0; (*q_r)[0][1] = 1.0 / 6.0;
            (*q_r)[1][0] = 2.0 / 3.0; (*q_r)[1][1] = 1.0 / 6.0;
            (*q_r)[2][0] = 1.0 / 6.0; (*q_r)[2][1] = 2.0 / 3.0;

            (*q_w)[0] = 1.0 / 6.0;
            (*q_w)[1] = 1.0 / 6.0;
            (*q_w)[2] = 1.0 / 6.0;
        }
        else if (n_quad == 4)
        {
            (*q_r)[0][0] = 1.0 / 3.0; (*q_r)[0][1] = 1.0 / 3.0;
            (*q_r)[1][0] = 1.0 / 5.0; (*q_r)[1][1] = 3.0 / 5.0;
            (*q_r)[2][0] = 1.0 / 5.0; (*q_r)[2][1] = 1.0 / 5.0;
            (*q_r)[3][0] = 3.0 / 5.0; (*q_r)[3][1] = 1.0 / 5.0;

            (*q_w)[0] = - 27.0 / 96.0;
            (*q_w)[1] = 25.0 / 96.0;
            (*q_w)[2] = 25.0 / 96.0;
            (*q_w)[3] = 25.0 / 96.0;
        }
        else
        {
            printf("No quadrature rule for %d points available\n", n_quad);
            exit(EXIT_FAILURE);
        }
    }
    else
    {
        if (n_quad == 4)
        {
            double a = (5.0 + 3.0 * sqrt(5.0)) / 20.0;
            double b = (5.0 - sqrt(5.0)) / 20.0;

            (*q_r)[0][0] = a; (*q_r)[0][1] = b; (*q_r)[0][2] = b;
            (*q_r)[1][0] = b; (*q_r)[1][1] = a; (*q_r)[1][2] = b;
            (*q_r)[2][0] = b; (*q_r)[2][1] = b; (*q_r)[2][2] = a;
            (*q_r)[3][0] = b; (*q_r)[3][1] = b; (*q_r)[3][2] = b;

            (*q_w)[0] = 1.0 / 24.0;
            (*q_w)[1] = 1.0 / 24.0;
            (*q_w)[2] = 1.0 / 24.0;
            (*q_w)[3] = 1.0 / 24.0;
        }
        else if (n_quad == 5)
        {
            (*q_r)[0][0] = 1.0 / 2.0; (*q_r)[0][1] = 1.0 / 6.0; (*q_r)[0][2] = 1.0 / 6.0;
            (*q_r)[1][0] = 1.0 / 6.0; (*q_r)[1][1] = 1.0 / 2.0; (*q_r)[1][2] = 1.0 / 6.0;
            (*q_r)[2][0] = 1.0 / 6.0; (*q_r)[2][1] = 1.0 / 6.0; (*q_r)[2][2] = 1.0 / 2.0;
            (*q_r)[3][0] = 1.0 / 6.0; (*q_r)[3][1] = 1.0 / 6.0; (*q_r)[3][2] = 1.0 / 6.0;
            (*q_r)[4][0] = 1.0 / 4.0; (*q_r)[4][1] = 1.0 / 4.0; (*q_r)[4][2] = 1.0 / 4.0;

            (*q_w)[0] = 9.0 / 20.0;
            (*q_w)[1] = 9.0 / 20.0;
            (*q_w)[2] = 9.0 / 20.0;
            (*q_w)[3] = 9.0 / 20.0;
            (*q_w)[4] = - 4.0 / 5.0;
        }
        else
        {
            printf("No quadrature rule for %d points available\n", n_quad);
            exit(EXIT_FAILURE);
        }
    }
}

void mesh_connectivity(int ***v_coord, int ***t_map, int num_fem, int n_dim)
{
    (*v_coord) = mem_alloc_2D_int(pow(n_dim, 2), n_dim);
    (*t_map) = mem_alloc_2D_int(num_fem, n_dim + 1);

    if (n_dim == 2)
    {
        (*v_coord)[0][0] = 0; (*v_coord)[0][1] = 0;
        (*v_coord)[1][0] = 1; (*v_coord)[1][1] = 0;
        (*v_coord)[2][0] = 0; (*v_coord)[2][1] = 1;
        (*v_coord)[3][0] = 1; (*v_coord)[3][1] = 1;

        if (num_fem == 2)
        {
            (*t_map)[0][0] = 0; (*t_map)[0][1] = 1; (*t_map)[0][2] = 3;
            (*t_map)[1][0] = 0; (*t_map)[1][1] = 3; (*t_map)[1][2] = 2;
        }
        else if (num_fem == 4)
        {
            (*t_map)[0][0] = 1; (*t_map)[0][1] = 2; (*t_map)[0][2] = 0;
            (*t_map)[1][0] = 3; (*t_map)[1][1] = 0; (*t_map)[1][2] = 1;
            (*t_map)[2][0] = 0; (*t_map)[2][1] = 3; (*t_map)[2][2] = 2;
            (*t_map)[3][0] = 2; (*t_map)[3][1] = 1; (*t_map)[3][2] = 3;
        }
        else
        {
            printf("Wrong number of triangles\n");
            exit(EXIT_FAILURE);
        }
    }
    else
    {
        (*v_coord)[0][0] = 0; (*v_coord)[0][1] = 0; (*v_coord)[0][2] = 0;
        (*v_coord)[1][0] = 1; (*v_coord)[1][1] = 0; (*v_coord)[1][2] = 0;
        (*v_coord)[2][0] = 0; (*v_coord)[2][1] = 1; (*v_coord)[2][2] = 0;
        (*v_coord)[3][0] = 1; (*v_coord)[3][1] = 1; (*v_coord)[3][2] = 0;
        (*v_coord)[4][0] = 0; (*v_coord)[4][1] = 0; (*v_coord)[4][2] = 1;
        (*v_coord)[5][0] = 1; (*v_coord)[5][1] = 0; (*v_coord)[5][2] = 1;
        (*v_coord)[6][0] = 0; (*v_coord)[6][1] = 1; (*v_coord)[6][2] = 1;
        (*v_coord)[7][0] = 1; (*v_coord)[7][1] = 1; (*v_coord)[7][2] = 1;

        if (num_fem == 6)
        {
            (*t_map)[0][0] = 0; (*t_map)[0][1] = 2; (*t_map)[0][2] = 1; (*t_map)[0][3] = 5;
            (*t_map)[1][0] = 1; (*t_map)[1][1] = 2; (*t_map)[1][2] = 3; (*t_map)[1][3] = 5;
            (*t_map)[2][0] = 0; (*t_map)[2][1] = 4; (*t_map)[2][2] = 2; (*t_map)[2][3] = 5;
            (*t_map)[3][0] = 5; (*t_map)[3][1] = 3; (*t_map)[3][2] = 7; (*t_map)[3][3] = 2;
            (*t_map)[4][0] = 4; (*t_map)[4][1] = 5; (*t_map)[4][2] = 6; (*t_map)[4][3] = 2;
            (*t_map)[5][0] = 5; (*t_map)[5][1] = 7; (*t_map)[5][2] = 6; (*t_map)[5][3] = 2;
        }
        else if (num_fem == 8)
        {
            (*t_map)[0][0] = 0; (*t_map)[0][1] = 2; (*t_map)[0][2] = 1; (*t_map)[0][3] = 4;
            (*t_map)[1][0] = 1; (*t_map)[1][1] = 0; (*t_map)[1][2] = 3; (*t_map)[1][3] = 5;
            (*t_map)[2][0] = 2; (*t_map)[2][1] = 6; (*t_map)[2][2] = 3; (*t_map)[2][3] = 0;
            (*t_map)[3][0] = 3; (*t_map)[3][1] = 2; (*t_map)[3][2] = 7; (*t_map)[3][3] = 1;
            (*t_map)[4][0] = 4; (*t_map)[4][1] = 5; (*t_map)[4][2] = 6; (*t_map)[4][3] = 0;
            (*t_map)[5][0] = 5; (*t_map)[5][1] = 7; (*t_map)[5][2] = 4; (*t_map)[5][3] = 1;
            (*t_map)[6][0] = 6; (*t_map)[6][1] = 7; (*t_map)[6][2] = 2; (*t_map)[6][3] = 4;
            (*t_map)[7][0] = 7; (*t_map)[7][1] = 3; (*t_map)[7][2] = 6; (*t_map)[7][3] = 5;
        }
        else
        {
            printf("Wrong number of tetrahedrals\n");
            exit(EXIT_SUCCESS);
        }
    }
}

void x_map(double **x, double *r, double **x_t, int n_dim, Basis *phi)
{
    int i, d;

    for (d = 0; d < n_dim; d++)
    {
        (*x)[d] = 0.0;

        for (i = 0; i < n_dim + 1; i++)
        {
            (*x)[d] += x_t[d][i] * phi[i](r);
        }
    }
}

void J_xr_map(double ***J_xr, double *r, double **x_t, int n_dim, DBasis *dphi)
{
    int i, j, k;
    double *deriv = mem_alloc_1D_double(n_dim);

    for (i = 0; i < n_dim; i++)
    {
        for (j = 0; j < n_dim; j++)
        {
            (*J_xr)[i][j] = 0.0;

            for (k = 0; k < n_dim + 1; k++)
            {
                dphi[k](&deriv, r);

                (*J_xr)[i][j] += x_t[i][k] * deriv[j];
            }
        }
    }

    mem_free_1D_double(&deriv, n_dim);
}

/* Basis functions and derivatives in 2D */
double phi_2D_1(double *r) { return r[0]; }
double phi_2D_2(double *r) { return r[1]; }
double phi_2D_3(double *r) { return 1.0 - r[0] - r[1]; }
void dphi_2D_1(double **dp, double *r) { (*dp)[0] = 1.0; (*dp)[1] = 0.0; }
void dphi_2D_2(double **dp, double *r) { (*dp)[0] = 0.0; (*dp)[1] = 1.0; }
void dphi_2D_3(double **dp, double *r) { (*dp)[0] = -1.0; (*dp)[1] = -1.0; }

/* Basis functions and derivatives in 3D */
double phi_3D_1(double *r) { return r[0]; }
double phi_3D_2(double *r) { return r[1]; }
double phi_3D_3(double *r) { return r[2]; }
double phi_3D_4(double *r) { return 1.0 - r[0] - r[1] - r[2]; }
void dphi_3D_1(double **dp, double *r) { (*dp)[0] = 1.0; (*dp)[1] = 0.0; (*dp)[2] = 0.0; }
void dphi_3D_2(double **dp, double *r) { (*dp)[0] = 0.0; (*dp)[1] = 1.0; (*dp)[2] = 0.0; }
void dphi_3D_3(double **dp, double *r) { (*dp)[0] = 0.0; (*dp)[1] = 0.0; (*dp)[2] = 1.0; }
void dphi_3D_4(double **dp, double *r) { (*dp)[0] = -1.0; (*dp)[1] = -1.0; (*dp)[2] = -1.0; }

/* Math functions */
long long maximum(long long a, long long b)
{
    return a > b ? a : b;
}

double determinant(double** A, int n)
{
    /*
     * Computes the determinant of a matrix
     */
    if (n == 2)
    {
        double d_1 = A[0][0] * A[1][1] - A[1][0] * A[0][1];

        return d_1;
    }
    else if (n == 3)
    {
        double d_1 = A[0][0] * (A[1][1] * A[2][2] - A[2][1] * A[1][2]);
        double d_2 = A[0][1] * (A[1][0] * A[2][2] - A[2][0] * A[1][2]);
        double d_3 = A[0][2] * (A[1][0] * A[2][1] - A[2][0] * A[1][1]);

        return d_1 - d_2 + d_3;
    }
    else if (n == 4)
    {
        double d_1 = A[0][0];
        double d_11 = A[1][1] * (A[2][2] * A[3][3] - A[3][2] * A[2][3]);
        double d_12 = A[1][2] * (A[2][1] * A[3][3] - A[3][1] * A[2][3]);
        double d_13 = A[1][3] * (A[2][1] * A[3][2] - A[3][1] * A[2][2]);

        double d_2 = A[0][1];
        double d_21 = A[1][0] * (A[2][2] * A[3][3] - A[3][2] * A[2][3]);
        double d_22 = A[1][2] * (A[2][0] * A[3][3] - A[3][0] * A[2][3]);
        double d_23 = A[1][3] * (A[2][0] * A[3][2] - A[3][0] * A[2][2]);

        double d_3 = A[0][2];
        double d_31 = A[1][0] * (A[2][1] * A[3][3] - A[3][1] * A[2][3]);
        double d_32 = A[1][1] * (A[2][0] * A[3][3] - A[3][0] * A[2][3]);
        double d_33 = A[1][3] * (A[2][0] * A[3][1] - A[3][0] * A[2][1]);

        double d_4 = A[0][3];
        double d_41 = A[1][0] * (A[2][1] * A[3][2] - A[3][1] * A[2][2]);
        double d_42 = A[1][1] * (A[2][0] * A[3][2] - A[3][0] * A[2][2]);
        double d_43 = A[1][2] * (A[2][0] * A[3][1] - A[3][0] * A[2][1]);

        return d_1 * (d_11 - d_12 + d_13) - d_2 * (d_21 - d_22 + d_23) + d_3 * (d_31 - d_32 + d_33) - d_4 * (d_41 - d_42 + d_43);
    }
    else
    {
        printf("ERROR: No determinant implementation for matrices of size > 4\n");
        exit(EXIT_FAILURE);
    }
}

void inverse(double*** inv_A, double** A, int n)
{
    /*
     * Computes the inverse of a matrix
     */
    double det_A = determinant(A, n);

    if (n == 2)
    {
        (*inv_A)[0][0] = (1.0 / det_A) * A[1][1];
        (*inv_A)[0][1] = -(1.0 / det_A) * A[0][1];
        (*inv_A)[1][0] = -(1.0 / det_A) * A[1][0];
        (*inv_A)[1][1] = (1.0 / det_A) * A[0][0];
    }
    else if (n == 3)
    {
        (*inv_A)[0][0] = (1.0 / det_A) * (A[1][1] * A[2][2] - A[2][1] * A[1][2]);
        (*inv_A)[0][1] = (1.0 / det_A) * (A[0][2] * A[2][1] - A[2][2] * A[0][1]);
        (*inv_A)[0][2] = (1.0 / det_A) * (A[0][1] * A[1][2] - A[1][1] * A[0][2]);
        (*inv_A)[1][0] = (1.0 / det_A) * (A[1][2] * A[2][0] - A[2][2] * A[1][0]);
        (*inv_A)[1][1] = (1.0 / det_A) * (A[0][0] * A[2][2] - A[2][0] * A[0][2]);
        (*inv_A)[1][2] = (1.0 / det_A) * (A[0][2] * A[1][0] - A[1][2] * A[0][0]);
        (*inv_A)[2][0] = (1.0 / det_A) * (A[1][0] * A[2][1] - A[2][0] * A[1][1]);
        (*inv_A)[2][1] = (1.0 / det_A) * (A[0][1] * A[2][0] - A[2][1] * A[0][0]);
        (*inv_A)[2][2] = (1.0 / det_A) * (A[0][0] * A[1][1] - A[1][0] * A[0][1]);
    }
    else
    {
        printf("ERROR: No inverse implementation for matrices of size > 3\n");
        exit(EXIT_FAILURE);
    }
}

/* Memory management */
int* mem_alloc_1D_int(int n)
{
    int *array = (int*) malloc(n * sizeof(int));

    return array;
}

long long* mem_alloc_1D_long(int n)
{
    long long *array = (long long*) malloc(n * sizeof(long long));

    return array;
}

double* mem_alloc_1D_double(int n)
{
    double *array = (double*) malloc(n * sizeof(double));

    return array;
}

int** mem_alloc_2D_int(int n, int m)
{
    int i;
    int **array = (int**) malloc(n * sizeof(int*));

    for (i = 0; i < n; i++)
        array[i] = (int*) malloc(m * sizeof(int));

    return array;
}

double** mem_alloc_2D_double(int n, int m)
{
    int i;
    double **array = (double**) malloc(n * sizeof(double*));

    for (i = 0; i < n; i++)
        array[i] = (double*) malloc(m * sizeof(double));

    return array;
}

void mem_free_1D_int(int **array, int n)
{
    free((*array));
}

void mem_free_1D_long(long long **array, int n)
{
    free((*array));
}

void mem_free_1D_double(double **array, int n)
{
    free((*array));
}

void mem_free_2D_int(int ***array, int n, int m)
{
    int i;

    for (i = 0; i < n; i++)
        free((*array)[i]);

    free((*array));
}

void mem_free_2D_double(double ***array, int n, int m)
{
    int i;

    for (i = 0; i < n; i++)
        free((*array)[i]);

    free((*array));
}
#else
void fem_amg_setup(const sint *n_x_, const sint *n_y_, const sint *n_z_,
                   const sint *n_elem_, const sint *n_dim_,
                   double *x_m_, double *y_m_, double *z_m_,
                   double *pmask_, double *binv_, const sint *nullspace,
                   const sint *gshf)
{
     fail(1,__FILE__,__LINE__,"please recompile with HYPRE support");
}
void fem_amg_solve(double *z, double *w)
{
     fail(1,__FILE__,__LINE__,"please recompile with HYPRE support");
}
#endif
