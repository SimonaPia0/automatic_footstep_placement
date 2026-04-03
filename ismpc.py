import numpy as np
import casadi as cs
import copy

class Ismpc:
  def __init__(self, initial, footstep_planner, params):
    self.params = params
    self.N = params['N']
    self.N_f = 3
    self.delta = params['world_time_step']
    self.h = params['h']
    self.eta = params['eta']
    self.foot_size = params['foot_size']
    self.initial = initial
    self.footstep_planner = footstep_planner
    
    # Backup plan to maintain step WIDTH
    self.original_plan = copy.deepcopy(self.footstep_planner.plan)
    
    self.sigma = lambda t, t0, t1: np.clip((t - t0) / (t1 - t0 + 1e-9), 0, 1)

    self.A_lip = np.array([[0, 1, 0], [self.eta**2, 0, -self.eta**2], [0, 0, 0]])
    self.B_lip = np.array([[0], [0], [1]])

    self.f = lambda x, u: cs.vertcat(
      self.A_lip @ x[0:3] + self.B_lip * u[0],
      self.A_lip @ x[3:6] + self.B_lip * u[1],
      self.A_lip @ x[6:9] + self.B_lip * u[2] + np.array([[0], [- params['g']], [0]]),
    )

    self.opt = cs.Opti('conic')
    p_opts = {"expand": True}
    s_opts = {"max_iter": 3000, "verbose": False, "adaptive_rho": True} 
    self.opt.solver("osqp", p_opts, s_opts)

    self.U = self.opt.variable(3, self.N)
    self.X = self.opt.variable(9, self.N + 1)
    
    self.F = self.opt.variable(2, self.N_f) 
    #relax of ZMP constraints 
    self.slack_x = self.opt.variable(self.N)
    self.slack_y = self.opt.variable(self.N)

    #current state measured
    self.x0_param = self.opt.parameter(9)
    self.zmp_z_mid_param = self.opt.parameter(self.N)
    #the last step fixed on the ground
    self.fixed_prev_pos_param = self.opt.parameter(2)
    #interpolation time coefficients
    self.sigma_param = self.opt.parameter(self.N, self.N_f)
    #nomial steps taken from the orginial plan
    self.nominal_F_param = self.opt.parameter(2, self.N_f)
    #tell us if a step is blocked
    self.lock_mask_param = self.opt.parameter(self.N_f)

    self.optimized_steps = {}
    init_x = (initial['lfoot']['pos'][3] + initial['rfoot']['pos'][3]) / 2.
    init_y = (initial['lfoot']['pos'][4] + initial['rfoot']['pos'][4]) / 2.
    self.optimized_steps[0] = np.array([init_x, init_y])

    for i in range(self.N):
      self.opt.subject_to(self.X[:, i + 1] == self.X[:, i] + self.delta * self.f(self.X[:, i], self.U[:, i]))

    #we start from the previous step
    mc_x_sym = self.fixed_prev_pos_param[0]
    mc_y_sym = self.fixed_prev_pos_param[1]
    
    for k in range(self.N_f):
        prev_x = self.fixed_prev_pos_param[0] if k == 0 else self.F[0, k-1]
        prev_y = self.fixed_prev_pos_param[1] if k == 0 else self.F[1, k-1]
        
        #if sigma=0 we are in the previous support yet, if sigma=1 we are in the new step
        mc_x_sym = mc_x_sym + self.sigma_param[:, k] * (self.F[0, k] - prev_x)
        mc_y_sym = mc_y_sym + self.sigma_param[:, k] * (self.F[1, k] - prev_y)
        
        # ANTI-SPLIT RESTRAINTS (Loose leash for wider strides)
        self.opt.subject_to(self.F[0, k] - self.nominal_F_param[0, k] <= 0.30)  # Fino a 30 cm in avanti
        self.opt.subject_to(self.F[0, k] - self.nominal_F_param[0, k] >= -0.30) # Fino a 30 cm indietro
        self.opt.subject_to(self.F[1, k] - self.nominal_F_param[1, k] <= 0.20)  # Fino a 20 cm di lato
        self.opt.subject_to(self.F[1, k] - self.nominal_F_param[1, k] >= -0.20)

    # --- cost function ---
    cost = 10 * cs.sumsqr(self.U) 
    cost += 100 * cs.sumsqr(self.X[2, 1:].T - mc_x_sym)
    cost += 100 * cs.sumsqr(self.X[5, 1:].T - mc_y_sym)
    cost += 100 * cs.sumsqr(self.X[8, 1:].T - self.zmp_z_mid_param)
    cost += 1e5 * cs.sumsqr(self.X[6, 1:].T - self.h) # Anti-squatting
    cost += 1e3 * cs.sumsqr(self.X[7, 1:].T)
    
    # Cost of avoiding extreme deviations from smooth walking
    for k in range(self.N_f):
        cost += 5000 * cs.sumsqr(self.F[:, k] - self.nominal_F_param[:, k])
        # Se il piede è a terra, spostarlo costa infinito
        cost += 1e7 * self.lock_mask_param[k] * cs.sumsqr(self.F[:, k] - self.nominal_F_param[:, k])
    
    cost += 1e6 * cs.sumsqr(self.slack_x)
    cost += 1e6 * cs.sumsqr(self.slack_y)

    self.opt.minimize(cost)

    # --- VINCOLI ZMP "SOFT" ---
    self.opt.subject_to(self.X[2, 1:].T <= mc_x_sym + self.foot_size / 2. + self.slack_x)
    self.opt.subject_to(self.X[2, 1:].T >= mc_x_sym - self.foot_size / 2. - self.slack_x)
    self.opt.subject_to(self.X[5, 1:].T <= mc_y_sym + self.foot_size / 2. + self.slack_y)
    self.opt.subject_to(self.X[5, 1:].T >= mc_y_sym - self.foot_size / 2. - self.slack_y)
    
    self.opt.subject_to(self.slack_x >= 0)
    self.opt.subject_to(self.slack_y >= 0)

    self.opt.subject_to(self.X[:, 0] == self.x0_param)

    self.opt.subject_to(self.X[1, 0] + self.eta * (self.X[0, 0] - self.X[2, 0]) == \
                        self.X[1, self.N] + self.eta * (self.X[0, self.N] - self.X[2, self.N]))
    self.opt.subject_to(self.X[4, 0] + self.eta * (self.X[3, 0] - self.X[5, 0]) == \
                        self.X[4, self.N] + self.eta * (self.X[3, self.N] - self.X[5, self.N]))
    self.opt.subject_to(self.X[7, 0] + self.eta * (self.X[6, 0] - self.X[8, 0]) == \
                        self.X[7, self.N] + self.eta * (self.X[6, self.N] - self.X[8, self.N]))

    self.x = np.zeros(9)
    self.lip_state = {'com': {'pos': np.zeros(3), 'vel': np.zeros(3), 'acc': np.zeros(3)},
                      'zmp': {'pos': np.zeros(3), 'vel': np.zeros(3)}}
    
    self.last_valid_x = np.zeros(9)
    self.last_valid_u = np.zeros(3)
    self.last_valid_F = np.zeros((2, self.N_f))

  def solve(self, current, t):
    try:
        c_p = np.nan_to_num(current['com']['pos'], nan=0.0, posinf=0.0, neginf=0.0)
        c_v = np.nan_to_num(current['com']['vel'], nan=0.0, posinf=0.0, neginf=0.0)
        z_p = np.nan_to_num(current['zmp']['pos'], nan=0.0, posinf=0.0, neginf=0.0)
        
        c_p = np.clip(c_p, -10.0, 10.0)
        c_v = np.clip(c_v, -10.0, 10.0)
        z_p = np.clip(z_p, -10.0, 10.0)
        
        self.x = np.array([c_p[0], c_v[0], z_p[0],
                           c_p[1], c_v[1], z_p[1],
                           c_p[2], c_v[2], z_p[2]])
    except Exception:
        self.x = self.last_valid_x

    fixed_prev, sigma_matrix, nominal_F, lock_mask, mc_z = self.generate_moving_constraint(t)

    self.opt.set_value(self.x0_param, self.x)
    self.opt.set_value(self.fixed_prev_pos_param, fixed_prev)
    self.opt.set_value(self.sigma_param, sigma_matrix)
    self.opt.set_value(self.nominal_F_param, nominal_F)
    self.opt.set_value(self.lock_mask_param, lock_mask)
    self.opt.set_value(self.zmp_z_mid_param, mc_z)

    try:
        #extraction of the solution
        sol = self.opt.solve()
        self.x = sol.value(self.X[:,1])
        self.u = sol.value(self.U[:,0])
        f_val = sol.value(self.F)
        
        #As a starting point you give him the solution you found now
        self.opt.set_initial(self.U, sol.value(self.U))
        self.opt.set_initial(self.X, sol.value(self.X))
        self.opt.set_initial(self.F, f_val)
        self.opt.set_initial(self.slack_x, sol.value(self.slack_x))
        self.opt.set_initial(self.slack_y, sol.value(self.slack_y))
        
        #save last valid solution
        self.last_valid_x = self.x
        self.last_valid_u = self.u
        self.last_valid_F = f_val

        #update footsep planner. We write the optimized future steps into the footstep_planner.plan
        idx = self.footstep_planner.get_step_index_at_time(t)
        
        for k in range(self.N_f):
            target_idx = idx + k + 1
            if target_idx < len(self.footstep_planner.plan):
                # If the foot is in flight and not locked, we update the plan
                if lock_mask[k] == 0.0:
                    self.footstep_planner.plan[target_idx]['pos'][0] = f_val[0, k]
                    self.footstep_planner.plan[target_idx]['pos'][1] = f_val[1, k]
                    self.optimized_steps[target_idx] = f_val[:, k]
    
    #solver failure management
    except Exception as e:
        self.x = self.last_valid_x
        self.u = self.last_valid_u
        self.opt.set_initial(self.X, np.tile(self.last_valid_x, (self.N + 1, 1)).T)
        self.opt.set_initial(self.U, np.tile(self.last_valid_u, (self.N, 1)).T)
        self.opt.set_initial(self.F, self.last_valid_F)
        self.opt.set_initial(self.slack_x, np.zeros(self.N))
        self.opt.set_initial(self.slack_y, np.zeros(self.N))
    
    #Convert the self.x vector to a more readable structure
    self.lip_state['com']['pos'] = np.array([self.x[0], self.x[3], self.x[6]])
    self.lip_state['com']['vel'] = np.array([self.x[1], self.x[4], self.x[7]])
    self.lip_state['zmp']['pos'] = np.array([self.x[2], self.x[5], self.x[8]])
    self.lip_state['zmp']['vel'] = self.u
    self.lip_state['com']['acc'] = self.eta**2 * (self.lip_state['com']['pos'] - self.lip_state['zmp']['pos']) + np.hstack([0, 0, - self.params['g']])
    
    contact = self.footstep_planner.get_phase_at_time(t)
    if contact == 'ss':
      contact = self.footstep_planner.plan[self.footstep_planner.get_step_index_at_time(t)]['foot_id']
    return self.lip_state, contact
  
  def generate_moving_constraint(self, t):
    idx = self.footstep_planner.get_step_index_at_time(t)
    phase = self.footstep_planner.get_phase_at_time(t)
    
    #determination of the previous fixed foot
    if idx in self.optimized_steps:
        fixed_prev_pos = self.optimized_steps[idx]
    else:
        fixed_prev_pos = self.footstep_planner.plan[idx]['pos'][:2]
        self.optimized_steps[idx] = fixed_prev_pos
    
    #for every instant of the horizon and for every future step, it says how much that step "contributes"
    sigma_matrix = np.zeros((self.N, self.N_f))
    #contains the future nominal steps
    nominal_F = np.zeros((2, self.N_f))
    #tells if a step is blocked
    lock_mask = np.zeros(self.N_f)
    #future time vector you work on
    time_array = np.arange(t, t + self.N)
    
    # We build the future path by adding the RELATIVE distances to the supporting foot
    current_nominal_base = fixed_prev_pos.copy()
    
    for k in range(self.N_f):
        target_idx = idx + k + 1
        if target_idx < len(self.original_plan):
            fs_start_time = self.footstep_planner.get_start_time(target_idx - 1)
            ds_start_time = fs_start_time + self.footstep_planner.plan[target_idx - 1]['ss_duration']
            fs_end_time = ds_start_time + self.footstep_planner.plan[target_idx - 1]['ds_duration']
            
            sigma_matrix[:, k] = self.sigma(time_array, ds_start_time, fs_end_time)
            
            # Calculating the "Relative Distance" from the original plane
            rel_step = self.original_plan[target_idx]['pos'][:2] - self.original_plan[target_idx-1]['pos'][:2]
            current_nominal_base = current_nominal_base + rel_step
            
            nominal_F[0, k] = current_nominal_base[0]
            nominal_F[1, k] = current_nominal_base[1]
            
            # LOCK: If it is the step in progress and we are landing on the ground, it is locked
            if k == 0 and phase == 'ds':
                lock_mask[k] = 1.0
                # We force the nominal to match exactly with the actual landed position
                nominal_F[0, k] = self.footstep_planner.plan[target_idx]['pos'][0]
                nominal_F[1, k] = self.footstep_planner.plan[target_idx]['pos'][1]

        #edge management to avoid incomplete arrays in case the plan runs out of steps
        else:
            if k > 0:
                nominal_F[:, k] = nominal_F[:, k-1]
            else:
                nominal_F[:, k] = fixed_prev_pos

    mc_z = np.zeros(self.N)
    
    return fixed_prev_pos, sigma_matrix, nominal_F, lock_mask, mc_z