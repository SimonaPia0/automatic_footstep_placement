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

        # backup original plan
        self.original_plan = copy.deepcopy(self.footstep_planner.plan)

        self.sigma = lambda t, t0, t1: np.clip((t - t0) / (t1 - t0 + 1e-9), 0, 1)

        self.A_lip = np.array([
            [0, 1, 0],
            [self.eta ** 2, 0, -self.eta ** 2],
            [0, 0, 0]
        ])

        self.B_lip = np.array([[0], [0], [1]])

        self.f = lambda x, u: cs.vertcat(
            self.A_lip @ x[0:3] + self.B_lip * u[0],
            self.A_lip @ x[3:6] + self.B_lip * u[1],
            self.A_lip @ x[6:9] + self.B_lip * u[2] + np.array([[0], [-params['g']], [0]]),
        )

        self.opt = cs.Opti('conic')
        p_opts = {"expand": True}
        s_opts = {"max_iter": 3000, "verbose": False, "adaptive_rho": True}
        self.opt.solver("osqp", p_opts, s_opts)

        # weight parameters
        self.weight_f_param = self.opt.parameter()
        self.weight_h_param = self.opt.parameter()
        self.weight_v_param = self.opt.parameter()
        self.foot_sign_param = self.opt.parameter(self.N_f + 1)

        # variables
        self.U = self.opt.variable(3, self.N)
        self.X = self.opt.variable(9, self.N + 1)

        # new decision variable
        self.F = self.opt.variable(2, self.N_f + 1)

        # new ZMP reference
        self.mc_x = self.opt.variable(self.N)
        self.mc_y = self.opt.variable(self.N)
        self.mc_z = self.opt.variable(self.N)

        # relaxed ZMP constraints
        self.slack_x = self.opt.variable(self.N)
        self.slack_y = self.opt.variable(self.N)

        # parameters
        self.x0_param = self.opt.parameter(9)
        self.zmp_z_mid_param = self.opt.variable(self.N)
        self.fixed_prev_pos_param = self.opt.parameter(2)
        self.sigma_param = self.opt.parameter(self.N, self.N_f + 1)
        self.nominal_F_param = self.opt.parameter(2, self.N_f + 1)
        # lock mask
        self.lock_mask_param = self.opt.parameter(self.N_f + 1)

        # internal state
        self.optimized_steps = {}

        init_x = (initial['lfoot']['pos'][3] + initial['rfoot']['pos'][3]) / 2.
        init_y = (initial['lfoot']['pos'][4] + initial['rfoot']['pos'][4]) / 2.
        self.optimized_steps[0] = np.array([init_x, init_y])

        self.x = np.zeros(9)

        self.lip_state = {
            'com': {
                'pos': np.zeros(3),
                'vel': np.zeros(3),
                'acc': np.zeros(3)
            },
            'zmp': {
                'pos': np.zeros(3),
                'vel': np.zeros(3)
            }
        }

        self.last_valid_x = np.zeros(9)
        self.last_valid_u = np.zeros(3)
        self.last_valid_F = np.zeros((2, self.N_f + 1))

        # LIP dynamics
        for i in range(self.N):
            self.opt.subject_to(
                self.X[:, i + 1] == self.X[:, i] + self.delta * self.f(self.X[:, i], self.U[:, i])
            )

        self.opt.subject_to(self.X[:, 0] == self.x0_param)
        self.opt.subject_to(self.F[:, 0] == self.fixed_prev_pos_param)

        mc_x_sym = self.fixed_prev_pos_param[0]
        mc_y_sym = self.fixed_prev_pos_param[1]

        for k in range(1, self.N_f + 1):
            prev_x = self.fixed_prev_pos_param[0] if k == 0 else self.F[0, k - 1]
            prev_y = self.fixed_prev_pos_param[1] if k == 0 else self.F[1, k - 1]

            # sigma = 0 previous support
            # sigma = 1 new step
            mc_x_sym = mc_x_sym + self.sigma_param[:, k] * (self.F[0, k] - prev_x)
            mc_y_sym = mc_y_sym + self.sigma_param[:, k] * (self.F[1, k] - prev_y)

            # anti-split constraints
            self.opt.subject_to(self.F[0, k] - self.nominal_F_param[0, k] <= 0.30)
            self.opt.subject_to(self.F[0, k] - self.nominal_F_param[0, k] >= -0.30)

            S = self.foot_sign_param[k]
            delta_y = self.F[1, k] - self.nominal_F_param[1, k]

            self.opt.subject_to(S * delta_y <= 0.20)
            self.opt.subject_to(S * delta_y >= -0.01)

        
        self.opt.subject_to(self.mc_x == mc_x_sym)
        self.opt.subject_to(self.mc_y == mc_y_sym)
        self.opt.subject_to(self.mc_z == self.zmp_z_mid_param)
        self.opt.subject_to(self.zmp_z_mid_param == 0)

        # cost function
        cost = 10 * cs.sumsqr(self.U)
        cost += 100 * cs.sumsqr(self.X[2, 1:].T - self.mc_x)
        cost += 100 * cs.sumsqr(self.X[5, 1:].T - self.mc_y)
        cost += 100 * cs.sumsqr(self.X[8, 1:].T - self.mc_z)

        # anti-squatting
        cost += self.weight_h_param * cs.sumsqr(self.X[6, 1:].T - self.h)

        # vertical damping
        cost += 1e3 * cs.sumsqr(self.X[7, 1:].T)

        # velocity contraint
        cost += self.weight_v_param * cs.sumsqr(self.X[1, 1:])
        cost += self.weight_v_param * cs.sumsqr(self.X[4, 1:])

        for k in range(1, self.N_f + 1):
            cost += self.weight_f_param * cs.sumsqr(
                self.F[:, k] - self.nominal_F_param[:, k]
            )

            cost += 1e7 * self.lock_mask_param[k] * cs.sumsqr(
                self.F[:, k] - self.nominal_F_param[:, k]
            )

        cost += 1e6 * cs.sumsqr(self.slack_x)
        cost += 1e6 * cs.sumsqr(self.slack_y)

        self.opt.minimize(cost)

        # soft ZMP constraints
        self.opt.subject_to(
            self.X[2, 1:].T <= self.mc_x + self.foot_size / 2. + self.slack_x
        )
        self.opt.subject_to(
            self.X[2, 1:].T >= self.mc_x - self.foot_size / 2. - self.slack_x
        )

        self.opt.subject_to(
            self.X[5, 1:].T <= self.mc_y + self.foot_size / 2. + self.slack_y
        )
        self.opt.subject_to(
            self.X[5, 1:].T >= self.mc_y - self.foot_size / 2. - self.slack_y
        )

        self.opt.subject_to(self.slack_x >= 0)
        self.opt.subject_to(self.slack_y >= 0)

        # stability constraints - periodic tail
        self.opt.subject_to(
            self.X[1, 0] + self.eta * (self.X[0, 0] - self.X[2, 0]) ==
            self.X[1, self.N] + self.eta * (self.X[0, self.N] - self.X[2, self.N])
        )

        self.opt.subject_to(
            self.X[4, 0] + self.eta * (self.X[3, 0] - self.X[5, 0]) ==
            self.X[4, self.N] + self.eta * (self.X[3, self.N] - self.X[5, self.N])
        )

        self.opt.subject_to(
            self.X[7, 0] + self.eta * (self.X[6, 0] - self.X[8, 0]) ==
            self.X[7, self.N] + self.eta * (self.X[6, self.N] - self.X[8, self.N])
        )

    def solve(self, current, t, push_active=False):
        try:
            c_p = np.nan_to_num(
                current['com']['pos'],
                nan=0.0,
                posinf=0.0,
                neginf=0.0
            )
            c_v = np.nan_to_num(
                current['com']['vel'],
                nan=0.0,
                posinf=0.0,
                neginf=0.0
            )
            z_p = np.nan_to_num(
                current['zmp']['pos'],
                nan=0.0,
                posinf=0.0,
                neginf=0.0
            )

            c_p = np.clip(c_p, -10.0, 10.0)
            c_v = np.clip(c_v, -10.0, 10.0)
            z_p = np.clip(z_p, -10.0, 10.0)

            self.x = np.array([
                c_p[0], c_v[0], z_p[0],
                c_p[1], c_v[1], z_p[1],
                c_p[2], c_v[2], z_p[2]
            ])

        except Exception:
            self.x = self.last_valid_x

        # weight adjust
        if push_active:
            w_f = 50.0
            w_h = 100.0
            w_v = 1000.0
        else:
            w_f = 5000.0
            w_h = 1e5
            w_v = 0.0

        fixed_prev, sigma_matrix, nominal_F, lock_mask, mc_z, foot_signs = \
            self.generate_moving_constraint(t)

        self.opt.set_value(self.x0_param, self.x)
        self.opt.set_value(self.fixed_prev_pos_param, fixed_prev)
        self.opt.set_value(self.sigma_param, sigma_matrix)
        self.opt.set_value(self.nominal_F_param, nominal_F)
        self.opt.set_value(self.lock_mask_param, lock_mask)
        self.opt.set_initial(self.zmp_z_mid_param, mc_z)
        self.opt.set_value(self.weight_f_param, w_f)
        self.opt.set_value(self.weight_h_param, w_h)
        self.opt.set_value(self.weight_v_param, w_v)
        self.opt.set_value(self.foot_sign_param, foot_signs)

        try:
            sol = self.opt.solve()

            self.x = sol.value(self.X[:, 1])
            self.u = sol.value(self.U[:, 0])
            f_val = sol.value(self.F)

            self.opt.set_initial(self.U, sol.value(self.U))
            self.opt.set_initial(self.X, sol.value(self.X))
            self.opt.set_initial(self.F, f_val)
            self.opt.set_initial(self.mc_x, sol.value(self.mc_x))
            self.opt.set_initial(self.mc_y, sol.value(self.mc_y))
            self.opt.set_initial(self.mc_z, sol.value(self.mc_z))
            self.opt.set_initial(self.slack_x, sol.value(self.slack_x))
            self.opt.set_initial(self.slack_y, sol.value(self.slack_y))

            self.last_valid_x = self.x
            self.last_valid_u = self.u
            self.last_valid_F = f_val

            # footstep planner update
            idx = self.footstep_planner.get_step_index_at_time(t)

            for k in range(1, self.N_f + 1):
                target_idx = idx + k 

                if target_idx < len(self.footstep_planner.plan):
                    if lock_mask[k] == 0.0:
                        self.footstep_planner.plan[target_idx]['pos'][0] = f_val[0, k]
                        self.footstep_planner.plan[target_idx]['pos'][1] = f_val[1, k]
                        self.optimized_steps[target_idx] = f_val[:, k]

        except Exception as e:
            print(f"[t={t}] Solver failed: {type(e).__name__}: {e}")

            self.x = self.last_valid_x
            self.u = self.last_valid_u

            self.opt.set_initial(
                self.X,
                np.tile(self.last_valid_x, (self.N + 1, 1)).T
            )
            self.opt.set_initial(
                self.U,
                np.tile(self.last_valid_u, (self.N, 1)).T
            )
            self.opt.set_initial(self.F, self.last_valid_F)
            self.opt.set_initial(self.mc_x, np.zeros(self.N))
            self.opt.set_initial(self.mc_y, np.zeros(self.N))
            self.opt.set_initial(self.mc_z, np.zeros(self.N))
            self.opt.set_initial(self.slack_x, np.zeros(self.N))
            self.opt.set_initial(self.slack_y, np.zeros(self.N))

        self.lip_state['com']['pos'] = np.array([
            self.x[0],
            self.x[3],
            self.x[6]
        ])

        self.lip_state['com']['vel'] = np.array([
            self.x[1],
            self.x[4],
            self.x[7]
        ])

        self.lip_state['zmp']['pos'] = np.array([
            self.x[2],
            self.x[5],
            self.x[8]
        ])

        self.lip_state['zmp']['vel'] = self.u

        self.lip_state['com']['acc'] = (
            self.eta ** 2 *
            (self.lip_state['com']['pos'] - self.lip_state['zmp']['pos'])
            + np.hstack([0, 0, -self.params['g']])
        )

        contact = self.footstep_planner.get_phase_at_time(t)

        if contact == 'ss':
            contact = self.footstep_planner.plan[
                self.footstep_planner.get_step_index_at_time(t)
            ]['foot_id']

        return self.lip_state, contact

    def generate_moving_constraint(self, t):
        idx = self.footstep_planner.get_step_index_at_time(t)
        phase = self.footstep_planner.get_phase_at_time(t)

        if idx in self.optimized_steps:
            fixed_prev_pos = self.optimized_steps[idx]
        else:
            fixed_prev_pos = np.array(self.footstep_planner.plan[idx]['pos'][:2])
            self.optimized_steps[idx] = fixed_prev_pos

        sigma_matrix = np.zeros((self.N, self.N_f + 1))
        nominal_F = np.zeros((2, self.N_f + 1))
        lock_mask = np.zeros(self.N_f + 1)
        foot_signs = np.zeros(self.N_f + 1)

        nominal_F[:, 0] = fixed_prev_pos
        lock_mask[0] = 1.0

        if self.footstep_planner.plan[idx]['foot_id'] == 'lfoot':
            foot_signs[0] = 1.0
        else:
            foot_signs[0] = -1.0

        time_array = np.arange(t, t + self.N)

        for k in range(1, self.N_f + 1):
            target_idx = idx + k

            if target_idx < len(self.original_plan):
                fs_start_time = self.footstep_planner.get_start_time(target_idx - 1)

                ds_start_time = (
                    fs_start_time +
                    self.footstep_planner.plan[target_idx - 1]['ss_duration']
                )

                fs_end_time = (
                    ds_start_time +
                    self.footstep_planner.plan[target_idx - 1]['ds_duration']
                )

                sigma_matrix[:, k] = self.sigma(
                    time_array,
                    ds_start_time,
                    fs_end_time
                )

                # original plan coordinates
                nominal_F[0, k] = self.original_plan[target_idx]['pos'][0]
                nominal_F[1, k] = self.original_plan[target_idx]['pos'][1]

                if self.original_plan[target_idx]['foot_id'] == 'lfoot':
                    foot_signs[k] = 1.0
                else:
                    foot_signs[k] = -1.0

                # lock step if in ds
                if k == 1 and phase == 'ds':
                    lock_mask[k] = 1.0

                    nominal_F[0, k] = self.footstep_planner.plan[target_idx]['pos'][0]
                    nominal_F[1, k] = self.footstep_planner.plan[target_idx]['pos'][1]

            else:
                if k > 1:
                    nominal_F[:, k] = nominal_F[:, k - 1]
                    foot_signs[k] = foot_signs[k - 1]
                else:
                    nominal_F[:, k] = fixed_prev_pos
                    foot_signs[k] = -foot_signs[0]

        mc_z = np.zeros(self.N)

        return fixed_prev_pos, sigma_matrix, nominal_F, lock_mask, mc_z, foot_signs